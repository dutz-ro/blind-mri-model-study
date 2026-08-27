# Security Policy

## Protected Data

Original DICOM, anonymous DICOM, exports, UID maps, prompts containing clinical context, model
responses, and reports are protected study artifacts and stay outside the repository.

## Reporting

Report vulnerabilities privately to the repository owner. Do not attach clinical data, API keys,
access tokens, or screenshots containing patient identity.

## Operational Rules

- Use a dedicated Unix account and a data root with mode `0700`.
- Keep original inputs read-only.
- Prefer Google Application Default Credentials over service-account key files.
- Use minimal IAM and a Vertex serving identity with read-only access to the anonymous store.
- Enable Cloud Data Access audit logs before upload.
- Do not assert legal compliance solely because metadata fields were removed.
- Pixel PHI review is mandatory before external transfer.
