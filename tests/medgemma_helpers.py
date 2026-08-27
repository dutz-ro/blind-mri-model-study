from __future__ import annotations

from copy import deepcopy

from tests.test_clinical_schema import valid_result


def result_for_evidence(evidence_id: str, *, status: str = "suspected") -> dict:
    result = deepcopy(valid_result())
    finding = result["findings"][0]
    finding["tear_status"] = status
    finding["traceable_evidence"][0]["export_id"] = evidence_id
    return result
