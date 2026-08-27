# LLM Operator Runbook

## Start

```bash
./bin/mristudy doctor
./bin/mristudy status --study <id>
./bin/mristudy next --study <id>
```

Read `AGENTS.md`. Do not infer the current step from old chat messages.

## Local Processing

```bash
./bin/mristudy run --study <id> --through privacy-review
```

Stop and ask the human to inspect `reports/02-anonymization-report.html` and pixel review images.
Do not approve on the human's behalf for real clinical data.

After approval:

```bash
./bin/mristudy export --study <id>
```

## External Providers

Provider discovery may use synthetic data. Medical-image egress requires a separate exact plan
and human approval. Never substitute a different alias after approval.

## Google

Use ADC. Create or select the Healthcare dataset, anonymous DICOM Store, and Vertex endpoint only
after a cloud plan is reviewed. Upload, verify exact SOP Instance UID coverage, run MedGemma, then
destroy resources and verify absence.

## Failure

Do not delete state or rerun into an existing result directory. Preserve the error and provenance,
fix configuration, and resume from the recorded stage. Never use destructive Git commands to
recover study data.
