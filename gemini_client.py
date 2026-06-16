"""Gemini enrichment: adds plain-English explanation + corrected code snippet
to each AST-confirmed Finding.

The enrichment prompt is grounded — it describes confirmed findings and asks
only for explanation + fix, with the response array keyed to each finding's
index. It cannot invent new findings. If GEMINI_API_KEY is missing, malformed,
or the API is down, findings are returned un-enriched rather than crashing
the pipeline.
"""

import json
import logging
import os
import re

MODEL = "gemini-2.5-flash-lite"

# Matches both fenced (```json [...] ```) and bare ([...]) JSON arrays.
_JSON_RE = re.compile(r'```(?:json)?\s*(\[.*?\])\s*```|(\[.*?\])', re.DOTALL)

logger = logging.getLogger(__name__)


def _get_client():
    """Lazily import google.genai and build a client from GEMINI_API_KEY.

    Lazy import so the google.genai package isn't required at import time;
    the bot still posts un-enriched findings if the package is missing.
    """
    from google import genai
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _numbered_context(code_context: str) -> str:
    lines = code_context.splitlines() if code_context else []
    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))


def _build_prompt(findings: list) -> str:
    """Build the batched enrichment prompt for all findings.

    Each finding is described by its pipeline stage, rule, severity, and
    +/-10 lines of numbered code context. The model is asked to respond
    with a single JSON array, one {"explanation": ..., "suggestion": ...}
    object per finding, in the same order — i.e. the response is keyed to
    each finding's index, so the model can only annotate confirmed findings,
    never invent new ones.
    """
    sections = []
    for i, finding in enumerate(findings):
        sections.append(
            f"Finding {i}:\n"
            f"Pipeline Stage: {finding.stage}\n"
            f"Rule: {finding.rule_id}\n"
            f"Severity: {finding.severity}\n"
            f"Code context (numbered):\n{_numbered_context(finding.code_context)}"
        )

    findings_block = "\n\n".join(sections)
    return (
        f"{findings_block}\n\n"
        "Task:\n"
        "1. For each finding above, explain in plain English why it breaks the ML pipeline.\n"
        "2. Provide a corrected code snippet for each finding.\n\n"
        "Respond with a single JSON array with one object per finding, in the same "
        "order as the findings above. Each object must have exactly this shape: "
        '{"explanation": "...", "suggestion": "..."}'
    )


def _extract_json_array(text: str):
    match = _JSON_RE.search(text)
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    return json.loads(raw)


def enrich_findings(findings: list) -> list:
    """Enrich each Finding with `explanation` and `suggestion` via Gemini.

    Graceful degradation: if GEMINI_API_KEY is unset, return findings
    unmodified. If the API call fails or the response can't be parsed via
    _JSON_RE, log it and fall back to un-enriched findings rather than
    crashing.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        return findings  # un-enriched, but pipeline continues

    if not findings:
        return findings

    try:
        client = _get_client()
        prompt = _build_prompt(findings)
        response = client.models.generate_content(model=MODEL, contents=prompt)
        enrichments = _extract_json_array(response.text)
    except Exception:
        logger.exception("Gemini enrichment failed — falling back to un-enriched findings.")
        return findings

    if enrichments is None:
        logger.warning("Could not extract a JSON array from Gemini's response — falling back to un-enriched findings.")
        return findings

    for finding, enrichment in zip(findings, enrichments):
        if isinstance(enrichment, dict):
            finding.explanation = enrichment.get("explanation", "")
            finding.suggestion = enrichment.get("suggestion", "")

    return findings
