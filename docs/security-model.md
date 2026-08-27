# Security Model

## Trust Boundaries

The local SSH server is the source-of-truth boundary. Routera/LiteLLM, upstream model providers,
Google Cloud Healthcare API, and Vertex AI are separate external boundaries.

The project does not upload original DICOM. Local de-identification removes private tags, remaps
UIDs coherently, shifts dates, rewrites identity declarations, and emits a review report. This is
not sufficient by itself: burned-in annotations and unusual vendor content require human review.

## Secrets

Secrets are read from environment variables or Application Default Credentials. They are never
included in operation plans or provenance. Error messages expose error classes, not provider
response bodies that may contain request content.

## Approvals

Approvals include a digest of the exact artifact being approved. Modifying paths, hashes, model
names, prompts, batches, or targets invalidates approval.

## Residual Risks

- OCR-free pixel review can miss small burned-in identity.
- DICOM private data may use unexpected encodings.
- A provider may transform or retain inputs according to its contract.
- Models can hallucinate plausible but false findings and evidence references.
- Agreement among models is not medical validation.
- Conventional MRI does not directly measure meniscal vascularity. `red-red`, `red-white`, and
  `white-white` are reported only as anatomical zone estimates, with confidence and limitations.
- Imaging factors may support specialist review, but the pipeline does not determine repairability
  and does not recommend conservative, repair, resection, or surgical treatment.
