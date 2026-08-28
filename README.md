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

## Study Profiles

A study profile decides the body region, which series are selected and under which role, the
clinical schema, the windowing constraints, and how findings are grouped during integration.
Everything else, including coverage, evidence identity, provenance and the safety boundaries, is
invariant and lives in code.

```bash
./bin/mristudy init-study --input /absolute/path/to/dicom --profile knee-meniscus-v1
```

Shipped profiles:

- `configs/profiles/knee-meniscus-v1.yaml` targets `knee-meniscus-finding-v2`.
- `configs/profiles/brain-unknown-v1.yaml` targets `mri-finding-core-v1` and demonstrates a second
  body region with unknown pathology.

A new region needs a profile, not a code change. `mri-finding-core-v1` is anatomy-independent: its
`observation_class` includes `unexpected_finding` and its `localization` carries free text, so an
unanticipated finding can be reported instead of being forced into another organ vocabulary.

## Evidence Traceability

Findings are graded `verified`, `recovered`, `cross_window` or `unverified`. A citation the model
misplaced into `finding_id` is recovered into the evidence list and recorded; an identifier absent
from the approved plan is refused rather than invented. A finding that cannot be traced is kept and
labelled, because losing a produced clinical result is worse than reporting it with a caveat.

```bash
./bin/mristudy evidence --study <id> --id image-000211
./bin/mristudy evidence --study <id> --from-result <result.json> --export-images
./bin/mristudy viewer-bundle --study <id>
```

`evidence` resolves an identifier to its anonymous UIDs, the de-identified instance on disk, that
file digest, and the windows the slice belongs to. Resolving the original acquisition UID is opt-in
because it re-identifies the study.

## Private Report Viewer

`report-webapp/` serves a local, Tailnet-only review page. `viewer-bundle` writes
`reports/viewer/report-bundle.json` and the cited images in the shape the viewer consumes. Each
model is presented independently and never merged into a shared verdict.

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
