# Google Cloud DICOM Store and MedGemma

Authenticate interactively when required:

```bash
gcloud auth login
gcloud auth application-default login
```

Configure the existing or separately bootstrapped resources:

```bash
export GOOGLE_CLOUD_PROJECT=<project>
export GOOGLE_CLOUD_REGION=<region>
export GOOGLE_HEALTHCARE_DATASET=<dataset>
export GOOGLE_DICOM_STORE=<anonymous-store>
export GOOGLE_VERTEX_ENDPOINT_ID=<endpoint-id>
```

Plan, inspect, and approve the exact anonymous DICOM upload:

```bash
./bin/mristudy cloud plan --study <id>
./bin/mristudy approve cloud-upload --study <id>
./bin/mristudy cloud apply --study <id>
```

The apply command uploads with STOW-RS and verifies the expected SOP Instance UIDs using QIDO-RS.
Run MedGemma only with a verified series UID:

```bash
./bin/mristudy cloud run-medgemma --study <id> --series-uid <anonymous-series-uid>
```

Resource creation and Vertex deployment remain organization-specific because billing, quota,
Model Garden terms, IAM, and region policy require explicit owner decisions.
