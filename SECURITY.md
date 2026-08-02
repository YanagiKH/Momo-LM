# Security Policy

## Supported versions

Security fixes are applied to the latest release and the `main` branch.

## Reporting a vulnerability

Do not publish exploitable details in a public issue. Use GitHub's private vulnerability reporting for this repository when available, including affected version, reproduction steps, impact, and a suggested mitigation.

## Trust boundaries

- The default web server listens only on `127.0.0.1`. Exposing it to a network requires authentication, TLS, rate limiting, and a firewall supplied by the operator.
- Files in the Mods directory are trusted native Python code, not sandboxed plugins.
- Web learning downloads user-selected pages. Operators remain responsible for authorization, copyright, privacy, and the content learned by the model.
- Checkpoints may reproduce sensitive training text. Do not publish weights trained on secrets or personal data.
- Momo-LM never requests an AI API key. Never place credentials in training data, prompts, config files, Mods, or issues.
