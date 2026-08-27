from mristudy.medgemma.integration import integrate_multiplanar, integrate_series
from tests.test_clinical_schema import valid_result


def window(window_id: str, status: str, evidence_id: str) -> dict:
    result = valid_result()
    result["findings"][0]["finding_id"] = "finding-1"
    result["findings"][0]["tear_status"] = status
    result["findings"][0]["traceable_evidence"][0]["export_id"] = evidence_id
    return {"window_id": window_id, "result": result}


def test_overlap_duplicate_is_deduplicated_without_double_counting_evidence() -> None:
    integrated, matrix = integrate_series(
        "primary-volume",
        "series-1",
        [window("w1", "probable", "image-000010"), window("w2", "probable", "image-000010")],
    )
    assert len(integrated["findings"]) == 1
    assert matrix["source_observations"] == 2
    assert matrix["finding_provenance"][0]["repeated_across_windows"]
    assert not matrix["finding_provenance"][0]["repeated_with_distinct_evidence"]
    assert matrix["finding_provenance"][0]["evidence_ids"] == ["image-000010"]


def test_overlap_contradiction_is_preserved() -> None:
    integrated, matrix = integrate_series(
        "primary-volume",
        "series-1",
        [window("w1", "probable", "image-000010"), window("w2", "absent", "image-000010")],
    )
    assert len(integrated["findings"]) == 2
    assert all(item["contradiction_preserved"] for item in matrix["finding_provenance"])


def test_same_shared_evidence_supporting_and_counter_stays_separate() -> None:
    supporting = window("w1", "probable", "image-000010")
    counter = window("w2", "probable", "image-000010")
    evidence = counter["result"]["findings"][0]["traceable_evidence"].pop()
    evidence["role"] = "counter"
    counter["result"]["findings"][0]["counter_evidence"] = [evidence]
    integrated, matrix = integrate_series(
        "primary-volume", "series-1", [supporting, counter]
    )
    assert len(integrated["findings"]) == 2
    assert all(item["contradiction_preserved"] for item in matrix["finding_provenance"])


def test_same_anatomy_without_shared_evidence_stays_separate() -> None:
    integrated, _ = integrate_series(
        "primary-volume",
        "series-1",
        [window("w1", "probable", "image-000010"), window("w2", "probable", "image-000020")],
    )
    assert len(integrated["findings"]) == 2


def test_distinct_consecutive_evidence_can_support_single_plane() -> None:
    first = window("w1", "probable", "image-000010")
    second = window("w2", "probable", "image-000011")
    series_result, provenance = integrate_series(
        "primary-volume",
        "series-1",
        [first, second],
    )
    # No shared evidence means conservative separation rather than false deduplication.
    assert len(series_result["findings"]) == 2
    assert all(
        not item["repeated_with_distinct_evidence"]
        for item in provenance["finding_provenance"]
    )


def test_multiplanar_positive_negative_disagreement_is_preserved() -> None:
    positive = valid_result()
    negative = valid_result()
    negative["findings"][0]["tear_status"] = "absent"
    final, matrix = integrate_multiplanar(
        [
            {"role": "sagittal-meniscus", "result": positive},
            {"role": "coronal-confirmation", "result": negative},
        ]
    )
    assert len(final["findings"]) == 2
    assert matrix["groups"][0]["classification"] == "contradicted_between_planes"
    assert matrix["groups"][0]["contradictions_preserved"]
    assert len(matrix["groups"][0]["output_finding_provenance"]) == 2


def test_integration_is_deterministic() -> None:
    inputs = [
        window("w2", "probable", "image-000010"),
        window("w1", "probable", "image-000010"),
    ]
    first = integrate_series("primary-volume", "series-1", inputs)
    second = integrate_series("primary-volume", "series-1", list(reversed(inputs)))
    assert first == second


def test_different_segments_are_never_merged() -> None:
    posterior = valid_result()
    body = valid_result()
    body["findings"][0]["segment"] = "body"
    final, matrix = integrate_multiplanar(
        [
            {"role": "sagittal-meniscus", "result": posterior},
            {"role": "coronal-confirmation", "result": body},
        ]
    )
    assert len(final["findings"]) == 2
    assert {(item["meniscus"], item["segment"]) for item in matrix["groups"]} == {
        ("medial", "posterior_horn"),
        ("medial", "body"),
    }
