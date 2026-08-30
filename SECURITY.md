# Security policy

## Supported versions

Security fixes target the current `main` branch and latest published release. Older releases may not receive backports.

## Report a vulnerability

Do not publish exploit details, secrets, personal data, private checkpoints, or affected user content in a public issue.

Use [GitHub private vulnerability reporting](https://github.com/YanagiKH/Momo-LM/security/advisories/new). Include:

- affected version or full commit SHA
- operating system, Python version, active backend, and installation method
- minimal reproduction without real credentials or private data
- security impact and required attacker access
- whether Mods, non-default host binding, custom checkpoints, or `MOMO_RUST_LIBRARY` are involved
- a suggested mitigation, if known

If private reporting is unavailable, open a public issue that only asks the maintainer for a private contact channel. Do not include the vulnerability details there.

## Trust boundaries

### HTTP service

The default server listens on `127.0.0.1`. A loopback service can still be reached by other local processes and by a browser visiting a malicious page, so the server checks `Host` and `Origin`. Every `/api/` request and private `/generated/` or `/speech/` artifact also requires `X-Momo-Token`. When no loopback token is configured, the process creates an ephemeral session token and gives it to the opened workbench through a URL fragment; the browser removes that fragment and keeps the token in session storage.

Binding to a non-loopback address requires an explicit access token. Tokens must contain 1–1024 visible ASCII characters. Configure them through `MOMO_ACCESS_TOKEN` or a protected config file; do not pass them in URLs, prompts, issues, or shell history.

Application checks do not provide TLS, user accounts, network isolation, denial-of-service protection, or host patching. A remote deployment needs a maintained reverse proxy, HTTPS, firewall／ACL, rate limits, log retention policy, and a dedicated low-privilege OS account.

### Checkpoints

Text v3 and image v2 checkpoints use NPZ with `allow_pickle=False`, bounded archive／metadata sizes, exact tensor names／shapes／dtypes, finite-value checks, and per-tensor SHA-256. This prevents Python pickle execution and rejects malformed known formats; it does not prove that a weight file is benign, accurate, licensed, or free of memorized data.

Only load checkpoints from a trusted source and verify the published whole-file SHA-256. Never publish weights trained on secrets, personal data, proprietary material, or data without redistribution rights.

### Native libraries

Wheels and installers build C/C++ and Rust code from this repository. ABI entrypoints validate pointers, dimensions, multiplication overflow, numeric inputs, and allocation failure before calculation. External C callers still own buffer allocation and must provide the documented lengths and aliasing behavior.

`MOMO_RUST_LIBRARY` intentionally loads an operator-selected shared library into the process. It is equivalent to executing native code and must never point to an untrusted file.

### Agents

Built-in agents use a fixed tool registry, profile capabilities, exact one-use approvals, budgets, a confined workspace, and persistent event records. They do not have tools for network access, arbitrary commands, email, cameras, microphones, vehicles, or other physical devices.

This is an application-layer restriction, not a kernel sandbox. Run Momo-LM with a low-privilege account when processing untrusted goals. Common credential strings are redacted from agent records, but the redactor is not a complete secret detector.

### Mods

Mods are ordinary Python code with the full permissions of the Momo-LM process. They can bypass agent restrictions, read files, access the network, start programs, and inspect environment variables. Only install code you wrote or audited. A Mod load error being isolated does not make the Mod safe.

### Image-training manifests

The loader confines relative image paths to the manifest directory, rejects symlink traversal, checks hashes and sizes, and decodes only PNG／JPEG／WebP. `source` and `license` fields are unverified declarations. Operators remain responsible for consent, copyright, privacy, dataset poisoning, and downstream model rights.

### Web learning

The crawler runs only after an explicit user request, follows same-origin HTTP／HTTPS links, observes `robots.txt`, and limits page count, response size, and timeout. These controls do not grant permission to copy or train on content. Review site terms, copyright, personal data, and prompt-like instructions in downloaded text before using `--train`.

### Local data and speech

SQLite databases, logs, generated files, agent events, speech, and checkpoints are stored under the selected Momo home. File permissions, disk encryption, backups, deletion, and multi-user isolation are supplied by the operating system. System TTS may be governed by separate platform privacy and licensing terms.

## Secrets

Momo-LM does not require a hosted AI API key. Do not put any credential in:

- training text, prompts, negative prompts, or manifests
- config files committed to Git
- agent goals, approvals, events, or workspace files
- Mods, issues, logs, screenshots, or generated artifacts

If a secret is exposed, revoke or rotate it first. Removing it from the latest commit does not remove it from Git history, caches, logs, databases, or checkpoints.

## Dependency and build security

GitHub Actions uses least-privilege `GITHUB_TOKEN` permissions and pins each Action to a full commit SHA. CI runs Python tests, native tests, rustfmt, Clippy, compiler warnings as errors, ASan／UBSan, clean-sdist fallback installation, executable smoke tests, Inno Setup compilation, and CodeQL for Python and C/C++.

These checks reduce known failure modes but cannot guarantee that the repository has no vulnerabilities. Release consumers should verify `SHA256SUMS.txt`, review the tagged source, and apply their own supply-chain policy.
