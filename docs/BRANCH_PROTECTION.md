# GitHub branch and release protection

Repository rules are GitHub-side settings and cannot be enforced by files in this repository. An administrator should apply the following rules to `main`, then verify them through GitHub Settings or the rulesets API.

## `main` ruleset

Target the default branch and enable:

- Require a pull request before merging.
- Require at least one approving review.
- Dismiss stale approvals after new commits.
- Require review from Code Owners.
- Require conversation resolution.
- Require branches to be up to date before merging.
- Block force pushes and branch deletion.
- Apply rules to administrators unless an emergency process explicitly documents an exception.
- Restrict bypass to named maintainers; audit every bypass.

Do not enable `pull_request_target` workflows that check out or execute pull-request code with write permissions or secrets.

## Required checks

Select the check names GitHub reports after the first run. For the current workflows they are:

```text
Python 3.10 · ubuntu-latest
Python 3.12 · ubuntu-latest
Python 3.10 · windows-latest
Python 3.12 · windows-latest
C/C++/Rust core · ubuntu-latest
C/C++/Rust core · windows-latest
Sanitizers · ubuntu-latest
Clean sdist · NumPy fallback
Executable smoke · ubuntu-latest
Executable smoke · windows-latest
CodeQL · python
CodeQL · c-cpp
```

GitHub may prefix these with a workflow name in the settings UI. Select the exact contexts emitted by a successful commit; do not create similarly named manual status checks.

If a job is renamed, update the ruleset before merging the rename. A stale required name can block every merge, while omitting a new name can leave a test optional.

## Tag and Release protection

Create a second ruleset for `v*` tags:

- Restrict tag creation, update, and deletion to release maintainers.
- Block force updates.
- Require the tag commit to be reachable from protected `main`.
- Use annotated, signed tags if the maintainer signing process supports them.

The Release workflow accepts semantic-version tags shaped like `v1.2.3` with an optional pre-release／build suffix. It validates and smoke-tests tagged source before publishing installers. Publishing uses `contents: write` only in the final job; validation and build jobs remain read-only.

For higher assurance, place the `publish` job behind a GitHub Environment named `release` with required reviewers. If you add an environment to the workflow, document it here and test a release candidate before relying on it.

## Repository settings

Recommended settings:

- Enable private vulnerability reporting.
- Enable Dependabot alerts and security updates.
- Enable secret scanning and push protection when available.
- Enable CodeQL default alerts for Python and C/C++.
- Disable Actions from untrusted publishers, or allow only GitHub-owned and explicitly approved pinned Actions.
- Require approval for workflows from first-time external contributors.
- Set the default `GITHUB_TOKEN` permission to read-only.
- Prevent GitHub Actions from creating or approving pull requests unless a documented automation requires it.
- Keep branch deletion after merge as a maintainer preference; it does not affect `main` protection.

## CODEOWNERS

`.github/CODEOWNERS` assigns repository-wide ownership and adds explicit entries for workflows, native code, and bundled weights. GitHub enforces those owners only when branch protection requires Code Owner review.

If ownership changes, update CODEOWNERS and repository team membership together. A nonexistent username or team silently weakens the intended review path.

## Verification checklist

After changing settings:

1. Open a test pull request from a branch without approval.
2. Confirm direct merge is blocked.
3. Confirm all matrix, sanitizer, fallback, package, and CodeQL checks appear.
4. Modify a workflow file and confirm Code Owner review is required.
5. Confirm a non-release maintainer cannot create or move a `v*` tag.
6. Confirm a failed installer job prevents `publish`.
7. Confirm Release assets include both installers and `SHA256SUMS.txt`.
8. Record the ruleset ID, reviewers, bypass actors, and verification date in the maintainer runbook.

Rulesets, organization policy, installed GitHub Apps, secrets, and environment reviewers are external state. CI success cannot prove those settings are enabled.
