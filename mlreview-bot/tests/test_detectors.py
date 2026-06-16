"""16+ tests: one positive + one negative case per detector.

Positive tests prove a detector fires on a real bug. Negative tests prove it
doesn't fire on look-alike clean code -- the precision guarantee that keeps
false-positive rate low.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors import run_detectors


def _rule_ids(code: str) -> list:
    findings = run_detectors(code)
    return [f.rule_id for f in findings]


# --- pre_split_leakage ---

def test_pre_split_leakage_fires():
    code = """
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, y)
"""
    assert "pre_split_leakage" in _rule_ids(code)


def test_fit_on_train_is_clean():
    code = """
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
"""
    assert "pre_split_leakage" not in _rule_ids(code)


# --- unseeded_split ---

def test_unseeded_split_fires():
    code = """
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""
    assert "unseeded_split" in _rule_ids(code)


def test_seeded_split_is_clean():
    code = """
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
    assert "unseeded_split" not in _rule_ids(code)


# --- test_size_too_small ---

def test_test_size_too_small_fires():
    code = """
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, y, test_size=0.05)
"""
    assert "test_size_too_small" in _rule_ids(code)


def test_test_size_reasonable_is_clean():
    code = """
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, y, test_size=0.2)
"""
    assert "test_size_too_small" not in _rule_ids(code)


# --- unseeded_model ---

def test_unseeded_model_fires():
    code = """
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
"""
    assert "unseeded_model" in _rule_ids(code)


def test_seeded_model_is_clean():
    code = """
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
"""
    assert "unseeded_model" not in _rule_ids(code)


# --- silent_failure ---

def test_silent_failure_fires():
    code = """
try:
    model.fit(X_train, y_train)
except Exception:
    pass
"""
    assert "silent_failure" in _rule_ids(code)


def test_logged_exception_is_clean():
    code = """
try:
    model.fit(X_train, y_train)
except Exception as e:
    logger.error(e)
"""
    assert "silent_failure" not in _rule_ids(code)


# --- metric_mismatch ---

def test_metric_mismatch_fires():
    code = """
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, predictions)
"""
    assert "metric_mismatch" in _rule_ids(code)


def test_f1_score_is_clean():
    code = """
from sklearn.metrics import f1_score
score = f1_score(y_test, predictions)
"""
    assert "metric_mismatch" not in _rule_ids(code)


# --- val_not_transformed ---

def test_val_not_transformed_fires():
    code = """
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test)
"""
    assert "val_not_transformed" in _rule_ids(code)


def test_val_transformed_is_clean():
    code = """
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
"""
    assert "val_not_transformed" not in _rule_ids(code)


# --- numpy_seed_missing ---

def test_numpy_seed_missing_fires():
    code = """
import numpy as np
import sklearn
"""
    assert "numpy_seed_missing" in _rule_ids(code)


def test_numpy_seeded_is_clean():
    code = """
import numpy as np
np.random.seed(42)
"""
    assert "numpy_seed_missing" not in _rule_ids(code)
