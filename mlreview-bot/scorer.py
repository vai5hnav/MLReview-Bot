"""Severity-weighted ML Quality Score (0-100) and pass/fail gate."""

SEVERITY_WEIGHTS = {"critical": 25, "warning": 10}


def compute_score(findings):
    """Return 100 minus severity-weighted deductions, floored at 0.

    deduction = sum(SEVERITY_WEIGHTS[f.severity] for f in findings)
    score = max(0, 100 - deduction)
    """
    deduction = sum(SEVERITY_WEIGHTS.get(f.severity, 0) for f in findings)
    return max(0, 100 - deduction)


def is_passing(score, threshold=70):
    """Return True if score >= threshold (default SCORE_THRESHOLD)."""
    return score >= threshold
