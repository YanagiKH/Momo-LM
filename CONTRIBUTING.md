# Contributing to Momo-LM

Contributions should be small enough to review, include tests for behavior changes, and preserve local-first operation without requiring a hosted AI API.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/build_native.py --release
python -m unittest discover -s tests -v
ruff check .
```

Changes to `native/` must pass CMake/CTest on Windows and Linux, `cargo test`, Clippy with warnings denied, and the Python numerical parity tests. Keep the public C ABI backward-compatible within an ABI version; increment the ABI version and document migration when a breaking change is unavoidable. Do not add Rust crate dependencies or native libraries without documenting their license, platform support and reproducible build behavior.

Use an isolated `MOMO_HOME` for manual tests. Do not commit personal databases, generated output, credentials, third-party model weights, or training data without a compatible license.

Pull requests should explain the user impact, implementation, limitations, and exact validation performed. New Mods must document their permissions and external dependencies.
