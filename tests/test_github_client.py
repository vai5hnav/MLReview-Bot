"""Regression tests for the pure-logic parts of github_client.py.

parse_diff_files is plain string processing with no I/O, so it's tested
directly against synthetic unified diffs -- no GitHub API access needed.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from github_client import parse_diff_files


def test_single_py_file_reconstructed():
    diff = """diff --git a/train.py b/train.py
index abc123..def456 100644
--- a/train.py
+++ b/train.py
@@ -1,5 +1,6 @@
 import sklearn
 from sklearn.model_selection import train_test_split
+import numpy as np
 X_train, X_test = train_test_split(X, y)
-model = RandomForestClassifier()
+model = RandomForestClassifier(random_state=42)
"""
    files = parse_diff_files(diff)
    assert list(files.keys()) == ["train.py"]
    expected = (
        "import sklearn\n"
        "from sklearn.model_selection import train_test_split\n"
        "import numpy as np\n"
        "X_train, X_test = train_test_split(X, y)\n"
        "model = RandomForestClassifier(random_state=42)"
    )
    assert files["train.py"] == expected


def test_non_py_files_are_excluded():
    diff = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,1 +1,1 @@
-x = 1
+x = 2
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -1,1 +1,1 @@
 y = 3
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1,2 +1,2 @@
-old line
+new line
"""
    files = parse_diff_files(diff)
    assert files == {"a.py": "x = 2", "b.py": "y = 3"}


def test_no_newline_marker_and_file_boundary_reset():
    diff = """diff --git a/c.txt b/c.txt
+++ b/c.txt
@@ -1,1 +1,1 @@
+hello
\\ No newline at end of file
diff --git a/d.py b/d.py
+++ b/d.py
@@ -1,1 +1,1 @@
+import numpy
"""
    files = parse_diff_files(diff)
    assert files == {"d.py": "import numpy"}


def test_empty_diff_returns_empty_dict():
    assert parse_diff_files("") == {}
