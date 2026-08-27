# Blind MRI Model Study

Reproducible, privacy-gated pipeline for exploratory blind comparison of MRI-capable AI
models. Original DICOM data remains local. Only a reviewed, de-identified copy can be sent to
external providers.

This software is not a medical device, does not produce a clinical diagnosis, and must not be
used to make treatment decisions.

## Quick Start

```bash
./install.sh --mode=native
./bin/mristudy doctor
./bin/mristudy demo run
```

For a real study:

```bash
./bin/mristudy init-study --input /absolute/path/to/dicom --profile knee-meniscus-v1
./bin/mristudy run --study <study-id> --through privacy-review
./bin/mristudy status --study <study-id>
```

Inspect `reports/02-anonymization-report.html` and the exported review material before approving:

```bash
./bin/mristudy approve privacy-review --study <study-id> --note "Reviewed locally"
./bin/mristudy export --study <study-id>
```

No model or cloud call is made by these commands.

## Safety Gates

- `privacy-review`: required before canonical image export and any future egress.
- `provider-egress`: bound to the exact image hashes, prompt hash, models, and batches.
- `cloud-upload`: bound to the exact anonymous DICOM files and target DICOM Store.
- `cloud-destroy`: bound to the exact uploaded instances.

An approval becomes invalid when its operation plan changes.

## Data Layout

Data is stored outside Git under `${MRI_STUDY_DATA_ROOT:-$HOME/mri-study-data}`. Each study has
separate `original`, `anonymized-dicom`, `exports`, `manifests`, `runs`, `reports`, `logs`, and
`state` directories. The original source path is referenced, not copied, unless the operator
copies it there explicitly.

## Documentation

- `docs/tutorials/01-install.md`
- `docs/tutorials/02-synthetic-run.md`
- `docs/tutorials/03-new-mri.md`
- `docs/tutorials/04-routera.md`
- `docs/tutorials/05-google-cloud.md`
- `docs/tutorials/05b-scaleway.md`
- `docs/tutorials/06-cleanup.md`
- `docs/llm-operator-runbook.md`
- `docs/security-model.md`
- `docs/clinical-reporting.md`
- `docs/implementation-issues.md`
