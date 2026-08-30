## Change

Describe the user-visible behavior and the smallest relevant implementation detail.

## Limits and compatibility

List known failures, migration needs, and any API, ABI, config, database, or checkpoint impact.

## Validation

Paste the exact commands and results. Mark checks that could not run locally and explain why.

- [ ] Python unit and HTTP integration tests
- [ ] Ruff and repository validator
- [ ] Windows／Linux behavior affected by this change
- [ ] CMake／CTest and Cargo checks, if native code changed
- [ ] Clean-sdist NumPy fallback, if packaging changed
- [ ] Checkpoint hashes, data provenance, leakage note, and metrics, if weights changed
- [ ] Security／privacy boundary reviewed

## Dependencies and data

Name every new dependency, dataset, weight, Mod, or generated asset. Include version, source, license, SHA-256 where applicable, and redistribution basis. Write `None` if this change adds none.

Do not include credentials, private prompts, personal data, proprietary checkpoints, or vulnerability details in this pull request.
