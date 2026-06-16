"""Entry point. Orchestrates the full MLReviewBot pipeline and exits with the
appropriate status code (0 = passed/merge allowed, 1 = failed/merge blocked).

End-to-end flow:
  1. github_client.get_pr_diff       -> fetch diff, reconstruct changed files
  2. analyzer.is_ml_file / analyze_file -> filter to ML files, run detectors
  3. gemini_client.enrich_findings   -> add explanation + suggestion per finding
  4. scorer.compute_score/is_passing -> 0-100 ML Quality Score + pass/fail
  5. github_client.post/update_review_comment -> one living PR comment
  6. github_client.create_check_run  -> green/red merge gate
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()  # no-ops in CI where vars are already injected
except ImportError:
    pass

import analyzer
import gemini_client
import github_client
import scorer

SEVERITY_ORDER = [("critical", "🔴", "Critical"), ("warning", "🟡", "Warning")]


def collect_findings(files: dict) -> list:
    """Run analyzer.analyze_file over each changed .py file, tagging each
    resulting Finding with the file path it came from (Finding itself has
    no path field -- the path is per-file, not per-detector).
    """
    findings = []
    for path, content in files.items():
        for finding in analyzer.analyze_file(path, content):
            finding.path = path
            findings.append(finding)
    return findings


def build_comment_body(findings: list, score: int, passed: bool, threshold: int) -> str:
    """Build the Markdown PR comment per Section 11's format."""
    status_emoji = "✅" if passed else "🔴"
    status_text = "PASSED" if passed else "FAILED"
    stage_count = len({f.stage for f in findings})

    lines = [
        f"{status_emoji} **MLReviewBot — ML Quality Score: {score}/100 ({status_text})**",
        f"Threshold: {threshold} | {len(findings)} findings across {stage_count} pipeline stages",
        "",
        "---",
    ]

    for severity, emoji, label in SEVERITY_ORDER:
        group = [f for f in findings if f.severity == severity]
        if not group:
            continue

        lines.append("")
        lines.append(f"### {emoji} {label} ({len(group)})")
        for finding in group:
            path = getattr(finding, "path", "?")
            lines.append("")
            lines.append(f"**`{path}:{finding.line}` — {finding.stage} — {finding.message}**")
            if finding.explanation:
                lines.append(f"> {finding.explanation}")
            if finding.suggestion:
                lines.append("```python")
                lines.append(finding.suggestion)
                lines.append("```")

    lines.append("")
    lines.append("---")
    lines.append(github_client.SIGNATURE)
    return "\n".join(lines)


def main() -> int:
    repo = os.environ["MLREVIEW_REPO"]
    pr = os.environ["MLREVIEW_PR"]
    sha = os.environ["MLREVIEW_SHA"]
    threshold = int(os.environ.get("SCORE_THRESHOLD", "70"))

    diff = github_client.get_pr_diff(repo, pr)
    files = github_client.parse_diff_files(diff)

    findings = collect_findings(files)
    findings = gemini_client.enrich_findings(findings)

    score = scorer.compute_score(findings)
    passed = scorer.is_passing(score, threshold)

    body = build_comment_body(findings, score, passed, threshold)

    existing_id = github_client.find_bot_comment(repo, pr)
    if existing_id is not None:
        github_client.update_review_comment(repo, existing_id, body)
    else:
        github_client.post_review_comment(repo, pr, body)

    github_client.create_check_run(repo, sha, score, passed, summary=body)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
