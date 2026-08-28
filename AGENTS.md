# LLM Operator Contract

1. Run `mristudy status --study <id>` and `mristudy next --study <id>` before acting.
2. Never display patient metadata, filenames containing identity, raw prompts, credentials, or
   DICOM pixels unless the user explicitly needs a local privacy review.
3. Never alter original DICOM files. Never write into the configured input directory.
4. Never bypass a state transition or fabricate an approval file.
5. Stop at `PRIVACY_REVIEW_REQUIRED` and ask the user to inspect the local report and images.
6. Do not send medical images to LiteLLM/Routera before `PROVIDER_EGRESS_APPROVED`.
7. Do not send DICOM or model inputs to any external DICOM-aware backend before
   `CLOUD_UPLOAD_APPROVED`.
8. Only upload files from `anonymized-dicom` that are listed in the approved plan.
9. Treat aliases as unverified until the provider probe records the returned model identity.
10. Never include the radiology report in prompts before all model outputs are locked.
11. Create a new run ID instead of overwriting a model result.
12. After MedGemma, create, approve, execute, and verify the destroy plan. Cleanup is incomplete
    until the provider inventory proves every temporary resource is absent.
13. Never commit DICOM, generated images, study state, UID maps, reports, logs, or secrets.
14. Ask the user only for credentials, interactive login, legal authorization, privacy approval,
    cost approval, region choice, and Git hosting choices that cannot be safely automated.
15. MedGemma medical runs must use the approved full-coverage window plan. Never reintroduce
    uniform slice sampling or select slices inside the runtime.
16. Do not provision Scaleway unless the provision plan binds the exact MedGemma run-plan digest,
    evidence-map digest, container digest, model snapshot manifest, prompt and runtime environment.
17. The 32-image and 32-DICOM synthetic probes must pass before medical access is authorized.
18. Never treat overlapping window memberships as independent evidence. Preserve all window and
    evidence provenance through series and multiplanar integration.
19. Never invent an evidence identifier. A citation may be recovered from `finding_id` only when
    that identifier exists in the approved plan, and the repair must be recorded.
20. Never discard a produced clinical finding because its evidence is weak. Keep it and label it
    `unverified`. Traceability is reported, never used to fail a run.
21. Never resolve an original acquisition UID unless explicitly asked. The UID map also holds the
    de-identification secret and subject identifier, which must never be returned or logged.
22. Change study behaviour through a profile in `configs/profiles`, not by hard-coding anatomy.
    A profile is authoritative; do not restate the clinical vocabulary that belongs to the schema.
23. Keep the canonical schema byte-identical unless the change is intended. Editing it alters the
    approved decoder policy digest and invalidates the plan and its approval.

The system is exploratory and non-diagnostic. Preserve model disagreement and uncertainty.
