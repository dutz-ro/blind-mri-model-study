# Architecture

The project uses a local DICOM-first core and isolated external adapters.

```text
original DICOM -> inventory/QC -> local de-identification -> privacy review
                                                   |
                         canonical PNG -> LiteLLM generalist arms
                                                   |
                         anonymous DICOM -> DICOMweb -> MedGemma
                                                   |
                         locked results -> comparison/reporting
```

Cloud and provider actions are plan/approve/apply operations. State is stored in
`state/status.json`; safe operator guidance is available through `mristudy next`.

The default backend is hybrid: Orthanc, DICOMweb, and OHIF stay on the existing SSH server while
a temporary Scaleway GPU runs MedGemma through SSH tunnels. Google remains an optional adapter.

## MedGemma window pipeline

```text
geometrically ordered anonymous DICOM series
    -> global evidence map
    -> balanced contiguous windows (max 32, overlap >= 8)
    -> exact UID analyze-window requests
    -> per-window validation and immutable manifests
    -> per-series evidence-linked integration
    -> multiplanar integration preserving contradictions
    -> coverage report (must be 100%)
```

The runtime does not select or uniformly sample medical sections. It receives exact approved SOP
Instance UID and evidence ID lists, verifies them against the mounted immutable run plan, and
retrieves only those instances from Orthanc.

Historical model outputs are hardened locally through a separate replay path. Replay enumerates
complete JSON candidates, rejects duplicate keys and ambiguous distinct valid objects, applies only
versioned conservative transformations, proves that every structural diff has an audit action, and
never reads DICOM or changes a historical result lock.
