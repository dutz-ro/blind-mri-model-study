from __future__ import annotations

import json

from .clinical_schema import load_knee_meniscus_schema


def knee_meniscus_prompt_v2() -> str:
    schema = json.dumps(load_knee_meniscus_schema(), separators=(",", ":"))
    return f"""You are participating in an exploratory, blind MRI image analysis study.
Inspect only the supplied anonymous knee MRI evidence. Do not diagnose, recommend treatment,
claim repairability, or claim that vascularity was directly observed.

For every absent, suspected, probable, indeterminate, or non-evaluable meniscal finding, document:
- medial/lateral meniscus and exact segment;
- probable tear pattern and orientation;
- extension to superior surface, inferior surface, or free edge;
- relation to the meniscocapsular margin and separation, if visible;
- conventional vascular-zone estimate: red-red, red-white, white-white, junctional, or
  indeterminate;
- distance to the meniscocapsular margin only when measurable, distinguishing calibrated geometry
  from a visual estimate;
- displaced fragment, flap, bucket-handle pattern, and imaging signs of mechanical obstruction;
- root involvement, extrusion, parameniscal cyst, tissue quality, and chronicity appearance;
- imaging factors that a qualified specialist may consider when evaluating treatment options.

The vascular-zone field is an anatomical estimate, not a measurement of blood flow. Always set
directly_observed_vascularity=false. Use indeterminate when spatial resolution, volume averaging,
geometry, or evidence is insufficient. Do not infer symptoms, clinical locking, tissue quality at
arthroscopy, healing probability, or a preferred intervention.

Return one JSON object matching this JSON Schema exactly, with no markdown or additional text:
{schema}
"""


def medgemma_knee_meniscus_prompt_v2() -> str:
    return """Analyze only the supplied anonymous knee MRI window for an exploratory,
non-diagnostic study. Return one JSON object matching knee-meniscus-finding-v2. Do not emit prose,
markdown, headings, or repeated normal statements. Use an empty findings array when no focal
meniscal finding is supported; do not enumerate every normal slice or repeat an absent finding.
For a focal finding, preserve uncertainty and cite only supplied image-NNNNNN evidence IDs.
The vascular zone is only an anatomical estimate. Never recommend treatment, determine
repairability, or claim directly observed vascularity.

Evidence citation rule. Every finding must cite the images it is based on inside
traceable_evidence, as objects of the form
{"export_id": "image-NNNNNN", "role": "supporting", "description": "..."}.
Use counter_evidence with role "counter" for images that argue against the finding.
Cite only image IDs supplied with this window, exactly as given.
finding_id is a label for the finding itself, such as "window-0010-01".
Never put an image ID in finding_id: a finding whose images are not cited in
traceable_evidence cannot be traced back to the scan and is treated as unverified."""


def medgemma_json_suffix_v2() -> str:
    schema = json.dumps(load_knee_meniscus_schema(), separators=(",", ":"))
    return f"""Return JSON now. Output exactly one object and nothing else. Stop after the closing
brace. Required JSON Schema:
{schema}"""


def medgemma_region_prompt(
    body_region: str,
    region_block: str = "",
    schema_version: str = "knee-meniscus-finding-v2",
) -> str:
    """Compose a window prompt for any body region.

    The invariant part carries the study contract and the citation rule; the
    region block comes from the study profile, so a new body region needs a
    profile rather than a new function here.
    """
    region = body_region.strip() or "unspecified"
    block = region_block.strip()
    lines = [
        f"Analyze only the supplied anonymous {region} MRI window for an exploratory,",
        f"non-diagnostic study. Return one JSON object matching {schema_version}. Do not emit",
        "prose, markdown, headings, or repeated normal statements. Use an empty findings array",
        "when no finding is supported; do not enumerate every normal slice or repeat an absent",
        "finding. Preserve uncertainty and prefer indeterminate over a confident label when the",
        "evidence is insufficient. Never recommend treatment, determine repairability, or claim",
        "directly observed vascularity.",
    ]
    if block:
        lines.extend(["", block])
    lines.extend(
        [
            "",
            "Evidence citation rule. Every finding must cite the images it is based on inside",
            'traceable_evidence, as objects of the form',
            '{"export_id": "image-NNNNNN", "role": "supporting", "description": "..."}.',
            'Use counter_evidence with role "counter" for images that argue against the finding.',
            "Cite only image IDs supplied with this window, exactly as given.",
            'finding_id is a label for the finding itself, such as "window-0010-01".',
            "Never put an image ID in finding_id: a finding whose images are not cited in",
            "traceable_evidence cannot be traced back to the scan and is treated as unverified.",
        ]
    )
    return "\n".join(lines)


def json_suffix_for(schema_version: str) -> str:
    """Output contract for any registered schema."""
    from .schema_registry import load_schema

    schema = json.dumps(load_schema(schema_version), separators=(",", ":"))
    return (
        "Return JSON now. Output exactly one object and nothing else. Stop after the closing\n"
        f"brace. Required JSON Schema:\n{schema}"
    )
