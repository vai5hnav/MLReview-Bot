"""Regression tests for the pure-logic parts of gemini_client.py.

_JSON_RE / _extract_json_array are tested directly against sample LLM-style
response strings -- no real Gemini API call needed.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_client import _extract_json_array


def test_extracts_fenced_json_array():
    text = """Here is the result:
```json
[{"explanation": "a", "suggestion": "b"}]
```
"""
    assert _extract_json_array(text) == [{"explanation": "a", "suggestion": "b"}]


def test_extracts_bare_json_array():
    text = 'leading text [{"explanation": "a2", "suggestion": "b2"}] trailing text'
    assert _extract_json_array(text) == [{"explanation": "a2", "suggestion": "b2"}]


def test_extracts_fenced_array_without_json_tag():
    text = """```
[{"explanation": "a3", "suggestion": "b3"}]
```"""
    assert _extract_json_array(text) == [{"explanation": "a3", "suggestion": "b3"}]


def test_multiple_findings_in_one_array():
    text = '[{"explanation": "e1", "suggestion": "s1"}, {"explanation": "e2", "suggestion": "s2"}]'
    result = _extract_json_array(text)
    assert result == [
        {"explanation": "e1", "suggestion": "s1"},
        {"explanation": "e2", "suggestion": "s2"},
    ]


def test_no_json_array_returns_none():
    assert _extract_json_array("no json here at all") is None


def test_malformed_json_raises():
    # _extract_json_array only guards against "no array found"; a syntactically
    # broken array is the caller's (enrich_findings's) problem to catch.
    import json
    import pytest

    with pytest.raises(json.JSONDecodeError):
        _extract_json_array('[{"explanation": "unterminated string]')
