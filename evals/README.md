# Bundled checkpoint evaluations

`base-model-report.json` records the weights committed under `momo_lm/assets/weights/`. The repository validator recomputes file hashes, parameter counts, format versions, fixed text metrics, and the untrained baseline before accepting a change.

## What the report measures

### Text

- mean negative log-likelihood over a fixed sequence of byte／BOS／EOS targets
- perplexity, calculated as `exp(mean_nll)`
- top-1 token accuracy over the same targets
- training steps and target count stored in checkpoint metadata

The text validation sequence comes from the repository starter corpus and overlaps the material used to train the bundled checkpoint. It is deterministic regression data, not an independent held-out set. The numbers do not measure factuality, instruction following, dialogue coherence, human preference, or safety, and should not be used to compare Momo-LM with other models.

### Image

- mean RGB reconstruction loss on fixed sampled coordinates
- per-style reconstruction loss for `anime`, `manga`, `illustration`, and `realistic`
- training steps, processed examples, and manifest SHA-256

The four procedural reference images are used for both training and evaluation. With style labels supplied as inputs, the recorded run lowers reconstruction loss on these known examples. Each label is confounded with one image, so the report does not isolate or establish the effect of style conditioning. These are not FID, prompt alignment, human preference, or held-out quality tests.

## Reproduce

Use a clean checkout and record the toolchain shown by each command:

```bash
python --version
python -c "import numpy, PIL; print(numpy.__version__, PIL.__version__)"
python scripts/validate_repo.py
python -m unittest discover -s tests -v
sha256sum momo_lm/assets/weights/momo-text-base.npz
sha256sum momo_lm/assets/weights/momo-image-base.npz
```

To regenerate bundled weights, use the deterministic bootstrap path documented by `python scripts/bootstrap_weights.py --help`, then rerun the validator. Weight regeneration is an intentional repository change: review the binary diff through hashes, metadata, metrics, and tests rather than line-oriented Git output.

Floating-point reductions can differ slightly across CPU, BLAS, NumPy, and compiler versions. The report records the reference environment and the validator uses explicit tolerances for recomputed metrics. Whole-file SHA-256 only matches a byte-identical checkpoint.

## Adding an evaluation

Keep raw, machine-readable outputs under `evals/`. State:

- exact commit and checkpoint SHA-256
- dataset source, license, split method, deduplication, and overlap
- metric formula and aggregation
- seed, model shape, optimizer, steps, and examples
- Python, NumPy, Pillow, OS, CPU, and native backend
- representative failures

Do not label a split held-out if the same source, near-duplicate, template, or generated variant appears in training.
