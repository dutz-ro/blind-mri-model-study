# Scaleway Hybrid DICOM-Aware Backend

The local server runs Orthanc, PostgreSQL, and OHIF. Scaleway supplies one temporary GPU instance
for MedGemma. Original DICOM never enters Orthanc or Scaleway; only the reviewed anonymous copy is
used.

## Local DICOM stack

```bash
./scripts/local-dicom-stack.sh up
./scripts/local-dicom-stack.sh status
```

The services bind only to loopback:

```text
Orthanc DICOMweb: http://127.0.0.1:18042/dicom-web
OHIF:             http://127.0.0.1:13000
```

Access OHIF from a workstation with:

```bash
ssh -L 13000:127.0.0.1:13000 openclaw@flow.dutz.ro
```

## Import anonymous DICOM

Load the generated stack environment without printing it:

```bash
set -a
. "$HOME/mri-study-data/services/orthanc/stack.env"
set +a
export MRI_STUDY_DATA_ROOT="$HOME/mri-study-data"

./bin/mristudy backend orthanc-probe --study <id>
./bin/mristudy backend orthanc-plan --study <id>
./bin/mristudy backend orthanc-import --study <id>
```

## Scaleway authentication

```bash
./scripts/install-scaleway-cli.sh
scw init
./bin/mristudy backend scaleway-audit
```

Do not paste the Scaleway secret key into chat, Git, plans, or logs.

## Plan migration

```bash
./bin/mristudy backend migrate-plan \
  --study <id> \
  --source-public-ip <server-ip>

./bin/mristudy approve backend-plan-migration --study <id>
./bin/mristudy backend apply-plan-migration --study <id>
```

Migration leaves the study at `CLOUD_PLAN_READY`; it does not authorize egress or create a GPU.

## Model access

Accept the gated terms for `google/medgemma-1.5-4b-it` on Hugging Face, create a read-only token,
and authenticate with `hf auth login`. The pinned model revision and container digest must be
written into a new backend plan before GPU provisioning can be approved.

The preferred runtime does not expose the Hugging Face token to the GPU. Download the pinned
snapshot on the trusted SSH server and transfer the verified snapshot after provisioning:

```bash
./scripts/download-medgemma-model.sh
```

## Full-coverage MedGemma plan

Generate the immutable plan after the final runtime digest is known:

```bash
./bin/mristudy backend medgemma-plan \
  --study <id> \
  --runtime-digest sha256:<final-container-digest>
```

The plan must report `coverage_percent: 100.0`, no missing or unexpected UIDs, windows no larger
than 32, and at least 8 overlapping sections between adjacent windows. The generated evidence map
is stored at `manifests/medgemma-evidence-map.json`.

The final Scaleway plan must then bind both the MedGemma run-plan SHA-256 and evidence-map SHA-256.
After approval and private runtime startup, run the required synthetic probes:

```bash
./bin/mristudy backend medgemma-probes --study <id>
```

This executes 1, 8, 16, and 32-image probes plus a 32-object synthetic DICOM probe. Medical DICOM
is forbidden unless the 32-image probe completes without OOM and every result validates against
`knee-meniscus-finding-v2`.

The probe sequence also includes a model-backed constrained-JSON probe. It must prove that the
pinned Transformers pipeline forwards the LM Format Enforcer callback through `generate_kwargs`,
that output starts with a JSON object, and that the result passes canonical schema and evidence
validation. Capacity-marker probes alone do not authorize medical access.

After verification:

```bash
./bin/mristudy backend verify-dicom-store --study <id>
./bin/mristudy backend run-medgemma --study <id>
```

Resume an interrupted run without overwriting completed windows:

```bash
./bin/mristudy backend run-medgemma \
  --study <id> \
  --resume-run-id medgemma-schema-v2-<run-id>
```
