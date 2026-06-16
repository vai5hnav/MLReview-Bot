"""ML-file gate and per-file analysis adapter."""

import ast

from detectors import run_detectors

ML_PACKAGES = {"sklearn", "numpy"}


def is_ml_file(file_content: str) -> bool:
    """Return True if the file imports sklearn or numpy.

    Used to skip non-ML files entirely so the bot stays quiet on unrelated
    changes in mixed repositories. A file with a syntax error isn't an ML
    file we can analyze, so it's treated as False here too -- run_detectors
    handles the SyntaxError case on its own for files that do pass the gate.
    """
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] in ML_PACKAGES for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in ML_PACKAGES:
                return True
    return False


def analyze_file(path: str, content: str) -> list:
    """Gate `content` via is_ml_file, then run all detectors if it's an ML file."""
    if not is_ml_file(content):
        return []
    return run_detectors(content)
