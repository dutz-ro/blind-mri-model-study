# Clinical Reporting Contract

New knee-meniscus model runs use `knee-meniscus-finding-v2`.

For each finding, the contract records meniscus, segment, tear status, pattern, orientation,
articular-surface extension, meniscocapsular relationship, displacement features, root involvement,
extrusion, cyst, tissue-quality appearance, chronicity appearance, and traceable evidence.

## Vascular-zone terminology

`red_red`, `red_white`, `white_white`, and `junctional` describe an estimated anatomical location.
They do not state that blood flow was visualized or measured. Every result must set:

```json
{
  "directly_observed_vascularity": false
}
```

Distance from the suspected finding to the meniscocapsular margin is included only when measurable.
The result distinguishes calibrated geometry, visual estimation, and a non-measurable relationship.

## Treatment boundary

The report may list imaging factors a qualified specialist could consider. It cannot determine
repairability or recommend a treatment. Symptoms, clinical locking, alignment, activity level,
cartilage status, tissue quality at arthroscopy, patient preferences, and operative findings remain
outside the model contract.

Locked GPT/Grok runs created before v2 remain immutable and are labeled `legacy_schema`. They are
not silently converted to the new schema. MedGemma and any future runs must validate against v2
before their result lock is accepted.

An optional GPT/Grok supplemental run can be planned with `models supplement-plan`. It creates a
new run ID, validates every batch against v2, and never overwrites the original blind result lock.

## MedGemma full coverage

Windowing applies only to the MedGemma arm. GPT and Grok inputs and locked results are unchanged.
The MedGemma planner orders each relevant series from DICOM geometry and divides it into balanced,
contiguous windows of at most 32 sections with at least 8 sections shared by adjacent windows.

The five relevant series contain 617 sections in this study. Every section must occur in at least
one window and every overlap membership is recorded. The same section always retains one global
`evidence_id`, even when it occurs in two neighboring windows. Overlap is repeated context, not
independent corroboration.

Each window is validated independently. Series integration deduplicates only evidence-linked
observations from overlapping windows, retains source finding IDs, and preserves contradictory
claim variants. Multiplanar integration retains all plane-specific variants and classifies them as:

```text
confirmed_multiplanar
supported_single_plane
window_local_only
contradicted_between_planes
indeterminate
not_comparable
```

Agreement between windows or planes is not medical validation. The report continues to prohibit
treatment recommendations, repairability determinations, and claims that vascularity was directly
measured.
