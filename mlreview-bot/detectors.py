"""All 8 ML pipeline bug detectors, implemented as ast.NodeVisitor subclasses.

Every detector emits Finding objects. Every visit_* override that needs to keep
walking child nodes MUST call self.generic_visit(node), otherwise traversal
stops at that subtree.
"""

import ast
from dataclasses import dataclass


@dataclass
class Finding:
    line: int                 # Where in the file
    rule_id: str               # e.g. "pre_split_leakage"
    severity: str               # "critical" | "warning"
    stage: str                # e.g. "Stage 1 — Preprocessing"
    message: str               # Short description
    code_context: str = ""    # +/-3 lines around the issue
    explanation: str = ""     # Filled by Gemini later
    suggestion: str = ""      # Filled by Gemini later


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _call_short_name(node: ast.Call):
    """Return the trailing identifier of a call's func: f(...) -> 'f',
    obj.method(...) -> 'method'. None if it's not a simple Name/Attribute.
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _dotted_name(node):
    """Render a Name/Attribute chain as a dotted string, e.g. np.random.seed.
    Returns None if the chain contains anything else (calls, subscripts, ...).
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def _has_kwarg(node: ast.Call, name: str) -> bool:
    return any(kw.arg == name for kw in node.keywords)


def _get_kwarg(node: ast.Call, name: str):
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _receiver_name(node: ast.Call):
    """For obj.method(...), return obj's simple Name id, else None."""
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return node.func.value.id
    return None


def _arg_name(node):
    """If `node` is a simple ast.Name, return its id, else None."""
    return node.id if isinstance(node, ast.Name) else None


# ---------------------------------------------------------------------------
# Stage 1 - Preprocessing
# ---------------------------------------------------------------------------

SCALER_CLASSES = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
    "Normalizer", "QuantileTransformer", "PowerTransformer", "Binarizer",
    "OneHotEncoder", "OrdinalEncoder", "LabelEncoder", "SimpleImputer",
    "KNNImputer", "PolynomialFeatures",
}


class PreSplitLeakageDetector(ast.NodeVisitor):
    """Stage 1 - Preprocessing | critical | pre_split_leakage

    Flags scaler/transformer .fit()/.fit_transform() calls that occur before
    train_test_split, i.e. the transformer learned statistics from the full
    (pre-split) dataset.
    """

    def __init__(self):
        self.findings = []
        self._scaler_vars = set()
        self._fit_lines = []
        self._split_lines = []

    def visit_Assign(self, node):
        if (
            isinstance(node.value, ast.Call)
            and _call_short_name(node.value) in SCALER_CLASSES
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._scaler_vars.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        name = _call_short_name(node)

        if name == "train_test_split":
            self._split_lines.append(node.lineno)

        elif name in ("fit", "fit_transform"):
            receiver = _receiver_name(node)
            if receiver in self._scaler_vars:
                first_arg_name = _arg_name(node.args[0]) if node.args else None
                if first_arg_name != "X_train":
                    self._fit_lines.append(node.lineno)

        self.generic_visit(node)

    def finalize(self):
        if self._fit_lines and self._split_lines:
            if min(self._fit_lines) < min(self._split_lines):
                self.findings.append(
                    Finding(
                        line=min(self._fit_lines),
                        rule_id="pre_split_leakage",
                        severity="critical",
                        stage="Stage 1 — Preprocessing",
                        message=(
                            "Scaler/transformer fitted on data before "
                            "train_test_split — test-set statistics leak into training."
                        ),
                    )
                )


# ---------------------------------------------------------------------------
# Stage 2 - Train/Test Split
# ---------------------------------------------------------------------------

class UnseededSplitDetector(ast.NodeVisitor):
    """Stage 2 - Split | warning | unseeded_split

    Flags train_test_split(...) calls missing the random_state kwarg.
    """

    def __init__(self):
        self.findings = []

    def visit_Call(self, node):
        if _call_short_name(node) == "train_test_split" and not _has_kwarg(node, "random_state"):
            self.findings.append(
                Finding(
                    line=node.lineno,
                    rule_id="unseeded_split",
                    severity="warning",
                    stage="Stage 2 — Split",
                    message="train_test_split() is missing random_state — results aren't reproducible.",
                )
            )
        self.generic_visit(node)


class TestSizeTooSmallDetector(ast.NodeVisitor):
    """Stage 2 - Split | warning | test_size_too_small

    Flags train_test_split(...) calls with test_size < 0.1 (float) or < 30 (int).
    """

    def __init__(self):
        self.findings = []

    def visit_Call(self, node):
        if _call_short_name(node) == "train_test_split":
            test_size = _get_kwarg(node, "test_size")
            if isinstance(test_size, ast.Constant) and isinstance(test_size.value, (int, float)):
                value = test_size.value
                too_small = (
                    (isinstance(value, float) and value < 0.1)
                    or (isinstance(value, int) and not isinstance(value, bool) and value < 30)
                )
                if too_small:
                    self.findings.append(
                        Finding(
                            line=node.lineno,
                            rule_id="test_size_too_small",
                            severity="warning",
                            stage="Stage 2 — Split",
                            message=(
                                "test_size is very small — evaluation metrics may be statistically unreliable."
                            ),
                        )
                    )
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Stage 3 - Model Training
# ---------------------------------------------------------------------------

class UnseededModelDetector(ast.NodeVisitor):
    """Stage 3 - Training | warning | unseeded_model

    Flags stochastic sklearn model constructors missing random_state.
    Allowlist: RandomForestClassifier/Regressor, GradientBoostingClassifier/Regressor,
    ExtraTreesClassifier/Regressor, DecisionTreeClassifier/Regressor, LogisticRegression,
    SVC, KMeans, MLPClassifier/Regressor, AdaBoostClassifier/Regressor,
    BaggingClassifier/Regressor.
    Exclude (deterministic, no false positives): LinearRegression, Ridge, Lasso,
    ElasticNet, KNeighborsClassifier/Regressor.
    """

    STOCHASTIC_MODELS = {
        "RandomForestClassifier", "RandomForestRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "ExtraTreesClassifier", "ExtraTreesRegressor",
        "DecisionTreeClassifier", "DecisionTreeRegressor",
        "LogisticRegression", "SVC", "KMeans",
        "MLPClassifier", "MLPRegressor",
        "AdaBoostClassifier", "AdaBoostRegressor",
        "BaggingClassifier", "BaggingRegressor",
    }

    def __init__(self):
        self.findings = []

    def visit_Call(self, node):
        name = _call_short_name(node)
        if name in self.STOCHASTIC_MODELS and not _has_kwarg(node, "random_state"):
            self.findings.append(
                Finding(
                    line=node.lineno,
                    rule_id="unseeded_model",
                    severity="warning",
                    stage="Stage 3 — Training",
                    message=f"{name}() is missing random_state — results vary across runs.",
                )
            )
        self.generic_visit(node)


class SilentFailureDetector(ast.NodeVisitor):
    """Stage 3 - Training | warning | silent_failure

    Flags broad except blocks (bare except: or except Exception/BaseException:)
    whose body has neither a logging call nor a raise statement.
    """

    LOGGING_ATTRS = {"error", "exception", "warning", "warn", "critical", "info", "debug", "log"}

    def _is_broad(self, node: ast.ExceptHandler) -> bool:
        if node.type is None:
            return True
        if isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException"):
            return True
        return False

    def _body_has_log_or_raise(self, node: ast.ExceptHandler) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                return True
            if isinstance(child, ast.Call):
                name = _call_short_name(child)
                if name == "print":
                    return True
                if isinstance(child.func, ast.Attribute) and child.func.attr in self.LOGGING_ATTRS:
                    return True
        return False

    def __init__(self):
        self.findings = []

    def visit_ExceptHandler(self, node):
        if self._is_broad(node) and not self._body_has_log_or_raise(node):
            self.findings.append(
                Finding(
                    line=node.lineno,
                    rule_id="silent_failure",
                    severity="warning",
                    stage="Stage 3 — Training",
                    message="Broad except block swallows exceptions silently — failures go unnoticed.",
                )
            )
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Stage 4 - Evaluation
# ---------------------------------------------------------------------------

class MetricMismatchDetector(ast.NodeVisitor):
    """Stage 4 - Evaluation | warning | metric_mismatch

    Flags accuracy_score(...) calls (potentially misleading on imbalanced data).
    Always a "flag for review", not a strict bug.
    """

    def __init__(self):
        self.findings = []

    def visit_Call(self, node):
        if _call_short_name(node) == "accuracy_score":
            self.findings.append(
                Finding(
                    line=node.lineno,
                    rule_id="metric_mismatch",
                    severity="warning",
                    stage="Stage 4 — Evaluation",
                    message=(
                        "accuracy_score() can be misleading on imbalanced data — verify class "
                        "balance or consider f1_score/roc_auc_score/balanced_accuracy_score."
                    ),
                )
            )
        self.generic_visit(node)


class ValNotTransformedDetector(ast.NodeVisitor):
    """Stage 4 - Evaluation | critical | val_not_transformed

    Flags cases where X_train is transformed via .transform()/.fit_transform()
    but X_test/X_val is later used in .predict()/.score() without being
    transformed the same way.
    """

    TRANSFORM_METHODS = {"transform", "fit_transform"}
    PREDICT_METHODS = {"predict", "score"}

    def __init__(self):
        self.findings = []
        self._transformed_vars = set()
        self._train_transformed = False
        self._candidates = []  # (lineno, var_name)

    @staticmethod
    def _looks_like_train(name: str) -> bool:
        lowered = name.lower()
        return "train" in lowered

    @staticmethod
    def _looks_like_test_or_val(name: str) -> bool:
        lowered = name.lower()
        return "test" in lowered or "val" in lowered

    def visit_Assign(self, node):
        if (
            isinstance(node.value, ast.Call)
            and _call_short_name(node.value) in self.TRANSFORM_METHODS
        ):
            # The output of a transform/fit_transform call is itself
            # transformed data, regardless of what it's named.
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._transformed_vars.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        method = _call_short_name(node)

        if method in self.TRANSFORM_METHODS and node.args:
            arg_name = _arg_name(node.args[0])
            if arg_name:
                self._transformed_vars.add(arg_name)
                if self._looks_like_train(arg_name):
                    self._train_transformed = True

        elif method in self.PREDICT_METHODS and node.args:
            arg_name = _arg_name(node.args[0])
            if arg_name and self._looks_like_test_or_val(arg_name):
                self._candidates.append((node.lineno, arg_name))

        self.generic_visit(node)

    def finalize(self):
        if not self._train_transformed:
            return
        for lineno, var_name in self._candidates:
            if var_name not in self._transformed_vars:
                self.findings.append(
                    Finding(
                        line=lineno,
                        rule_id="val_not_transformed",
                        severity="critical",
                        stage="Stage 4 — Evaluation",
                        message=(
                            f"'{var_name}' is used for prediction/scoring without being "
                            "transformed the same way as the training data."
                        ),
                    )
                )


# ---------------------------------------------------------------------------
# Stage 5 - Reproducibility
# ---------------------------------------------------------------------------

class NumpySeedMissingDetector(ast.NodeVisitor):
    """Stage 5 - Reproducibility | warning | numpy_seed_missing

    If numpy is imported but np.random.seed()/numpy.random.seed()/
    np.random.default_rng() is never called anywhere in the file, flag at line 1.
    """

    SEED_SUFFIXES = (".random.seed", ".random.default_rng")

    def __init__(self):
        self.findings = []
        self._numpy_aliases = set()
        self._seeded = False

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "numpy":
                self._numpy_aliases.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.generic_visit(node)

    def visit_Call(self, node):
        dotted = _dotted_name(node.func)
        if dotted:
            for alias in self._numpy_aliases:
                if any(dotted == f"{alias}{suffix}" for suffix in self.SEED_SUFFIXES):
                    self._seeded = True
        self.generic_visit(node)

    def finalize(self):
        if self._numpy_aliases and not self._seeded:
            self.findings.append(
                Finding(
                    line=1,
                    rule_id="numpy_seed_missing",
                    severity="warning",
                    stage="Stage 5 — Reproducibility",
                    message=(
                        "numpy is imported but never seeded — NumPy-driven randomness "
                        "(KFold shuffling, permutation_importance, ...) isn't reproducible."
                    ),
                )
            )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_DETECTOR_CLASSES = [
    PreSplitLeakageDetector,
    UnseededSplitDetector,
    TestSizeTooSmallDetector,
    UnseededModelDetector,
    SilentFailureDetector,
    MetricMismatchDetector,
    ValNotTransformedDetector,
    NumpySeedMissingDetector,
]


def _attach_code_context(findings, source_lines, context=3):
    for finding in findings:
        start = max(0, finding.line - 1 - context)
        end = min(len(source_lines), finding.line - 1 + context + 1)
        finding.code_context = "\n".join(source_lines[start:end])


def run_detectors(code: str) -> list:
    """Parse `code` with ast and run all 8 detectors, returning combined Findings.

    If code has a SyntaxError, ast.parse() raises; this function catches that
    and returns [] rather than crashing (the developer's own CI catches syntax
    errors; MLReviewBot stays out of the way).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    findings = []
    for detector_cls in _DETECTOR_CLASSES:
        detector = detector_cls()
        detector.visit(tree)
        if hasattr(detector, "finalize"):
            detector.finalize()
        findings.extend(detector.findings)

    _attach_code_context(findings, code.splitlines())
    return findings
