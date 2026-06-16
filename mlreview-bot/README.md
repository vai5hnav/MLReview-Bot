# MLReviewBot

A GitHub Actions bot that reviews scikit-learn ML pipeline code on every pull
request, catching silent semantic bugs (data leakage, unseeded randomness,
train/test contamination, metric mismatches, exception-swallowing) that
general-purpose tools like Copilot, CodeRabbit, SonarQube, and Pylint miss.

## Install

Add this to `.github/workflows/mlreview.yml` in your repo (template at
[`workflow-template/mlreview.yml`](workflow-template/mlreview.yml)):

```yaml
name: MLReviewBot
on:
  pull_request:
    paths: ['**.py']
jobs:
  ml-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: vai5hnav/mlreviewbot@v1
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          gemini-key: ${{ secrets.GEMINI_API_KEY }}
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | No* | — | Google AI Studio key. If missing, findings post un-enriched |
| `GITHUB_TOKEN` | Yes | — | PAT with `repo` + `pull-requests:write` (+ optional `checks:write`) |
| `SCORE_THRESHOLD` | No | `70` | Min score to pass |
| `MLREVIEW_REPO` | Yes | — | `owner/repo` (injected by Actions) |
| `MLREVIEW_PR` | Yes | — | PR number (injected by Actions) |
| `MLREVIEW_SHA` | Yes | — | Head commit SHA (injected by Actions) |

\* Optional in the sense that the bot still works without it; you just lose
the Gemini explanations and fix suggestions.

For local development, copy [`.env.example`](.env.example) to `.env`.

## How `uses: vai5hnav/mlreviewbot@v1` works

This repo is itself the GitHub Action referenced by the workflow template,
via [`action.yml`](action.yml) at the repo root. It's a composite action: it
installs `requirements.txt` and runs `runner.py`, mapping the action's
`github-token`/`gemini-key`/`score-threshold` inputs to `GITHUB_TOKEN`/
`GEMINI_API_KEY`/`SCORE_THRESHOLD`, and deriving `MLREVIEW_REPO`/
`MLREVIEW_PR`/`MLREVIEW_SHA` from the GitHub Actions event context
(`github.repository`, `github.event.pull_request.number`,
`github.event.pull_request.head.sha`). No separate deployment or build step —
publishing a tagged release of this repo (e.g. `v1`) is what makes
`uses: vai5hnav/mlreviewbot@v1` resolve.

## Development

```bash
pip install -r requirements.txt
pytest tests/ -v
```

See `MLReviewBot v1.0_Documentation.docx` (parent directory) for full design
docs: detector internals, scoring formula, architecture, and known limitations.
