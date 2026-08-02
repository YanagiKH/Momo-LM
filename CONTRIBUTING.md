# Contributing to Momo-LM

Contributions should be small enough to review, include tests for behavior changes, and preserve local-first operation without requiring a hosted AI API.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
ruff check .
```

Use an isolated `MOMO_HOME` for manual tests. Do not commit personal databases, generated output, credentials, third-party model weights, or training data without a compatible license.

Pull requests should explain the user impact, implementation, limitations, and exact validation performed. New Mods must document their permissions and external dependencies.
