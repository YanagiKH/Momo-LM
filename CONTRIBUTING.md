# Contributing to Momo-LM

Changes should be reviewable, tested, and usable without a hosted AI API. Describe what changed, what remains limited, and the exact commands you ran.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/build_native.py --release
```

Windows PowerShell uses `.\.venv\Scripts\Activate.ps1`.

Use an isolated home for manual tests:

```bash
momo --home ./tmp/dev-home init
momo --home ./tmp/dev-home serve
```

Do not commit that home, personal databases, generated output, credentials, third-party weights, or training data without documented redistribution rights.

## Required checks

Run the checks relevant to your change; before a release, run the complete set:

```bash
python -m compileall -q momo_lm scripts tests
ruff check .
python scripts/validate_repo.py
python -m unittest discover -s tests -v
python -m build
```

Native changes also require:

```bash
cmake -S native -B build/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=ON
cmake --build build/native --config Release
ctest --test-dir build/native -C Release --output-on-failure
cargo fmt --manifest-path native/rust/Cargo.toml --all -- --check
cargo test --manifest-path native/rust/Cargo.toml --release --locked
cargo clippy --manifest-path native/rust/Cargo.toml --all-targets --locked -- -D warnings
```

GitHub Actions runs Windows／Linux × Python 3.10／3.12, ASan／UBSan, a clean sdist install with missing compilers, PyInstaller smoke tests, Inno Setup compilation, and CodeQL Python／C++.

## Pull requests

Include:

- user-visible behavior and compatibility impact
- files and trust boundaries changed
- test inputs, commands, results, and any skipped platform
- benchmark method and raw data for performance claims
- migration steps for config, database, ABI, API, or checkpoint changes
- new dependencies, their license, why they are needed, and reproducible version constraints
- known failure cases that remain

Do not claim model-quality improvement from train loss alone. Model changes need a separate held-out set, leakage check, baseline, complete metrics, and representative failures. UI screenshots do not replace HTTP and server tests.

## Native ABI

Keep ABI v2 backward-compatible within the version. Before adding or changing an entrypoint:

1. Document tensor layout, dtype, dimensions, aliasing, allocation ownership, status codes, and numeric range.
2. Validate every size and multiplication before pointer arithmetic.
3. Add invalid, overflow, non-finite, empty, and allocation-failure tests.
4. Compare C++ and Rust output with NumPy using an explicit tolerance.
5. Update the CPython bridge and [docs/NATIVE_CORE.md](docs/NATIVE_CORE.md).

Increment the ABI version for an incompatible contract. Never silently reinterpret an existing function.

## Checkpoints and training data

Checkpoint format changes need strict tensor manifests, bounded loading, migration tests for every supported old format, and rejection tests for unknown or malformed archives.

Any committed training asset must include source, author／generator, license, creation method, SHA-256, and a reason redistribution is allowed. Split near-duplicates before train／validation separation. Do not include private prompts or scraped personal data.

When bundled weights change, update `evals/base-model-report.json` with:

- whole-file SHA-256
- shape and parameter count
- training command／function inputs
- seed, steps, examples and optimizer settings
- initial／final metrics and definitions
- reproduction environment
- explicit statement about overlap and non-held-out data

## Agents and tools

New agent tools must have a narrow capability, explicit input bounds, path confinement, structured output, audit events, cancellation behavior, and tests. Mutating tools require exact one-use approval. Network, arbitrary shell, process execution, external messages, sensors, vehicles, and physical-control tools are outside the built-in registry.

Mods remain trusted Python code and must not be described as sandboxed.

## Security reports

Do not submit exploitable details through a pull request or public issue. Follow [SECURITY.md](SECURITY.md).

## Release changes

Only a reviewed semantic-version `v*` tag should start a release. The workflow validates the tag, rebuilds Windows and Linux installers from tagged source, smoke-tests them, publishes SHA-256 checksums, and writes the Release with least-privilege permissions. Follow [docs/BRANCH_PROTECTION.md](docs/BRANCH_PROTECTION.md) before granting direct push or release authority.
