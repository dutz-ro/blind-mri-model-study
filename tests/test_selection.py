from mristudy.models.selection import select_generalist_exports


def test_generalist_selection_excludes_dense_source_volume() -> None:
    descriptions = [
        ("s1", "MPR SAG DP FS"),
        ("s2", "MPR COR DP FS"),
        ("s3", "MPR AX DP FS"),
        ("s4", "SAG T1 TSE"),
        ("s5", "SAG 3D DP FS"),
    ]
    inventory = {
        "instances": [
            {
                "series_uid": uid,
                "relative_path": f"series/{uid}.dcm",
                "series_description": description,
                "protocol_name": description,
            }
            for uid, description in descriptions
        ]
    }
    exports = {
        "exports": [
            {
                "export_id": f"image-{index}",
                "series_uid": f"anonymous-{uid}",
                "source_relative_path": f"series/{uid}.dcm",
            }
            for index, (uid, _) in enumerate(descriptions, start=1)
        ]
    }
    result = select_generalist_exports(inventory, exports)
    assert result["total_exports"] == 4
    assert "image-5" not in result["selected_exports"]
    assert result["excluded_exports"] == 1


def test_generalist_selection_keeps_every_instance_in_selected_series() -> None:
    inventory = {
        "instances": [
            {
                "series_uid": "s1",
                "relative_path": f"mpr/slice-{index}.dcm",
                "series_description": "MPR SAG DP FS",
                "protocol_name": "MPR SAG DP FS",
            }
            for index in range(3)
        ]
        + [
            {
                "series_uid": "s2",
                "relative_path": "cor/slice.dcm",
                "series_description": "MPR COR DP FS",
                "protocol_name": "MPR COR DP FS",
            },
            {
                "series_uid": "s3",
                "relative_path": "ax/slice.dcm",
                "series_description": "MPR AX DP FS",
                "protocol_name": "MPR AX DP FS",
            },
            {
                "series_uid": "s4",
                "relative_path": "t1/slice.dcm",
                "series_description": "SAG T1 TSE",
                "protocol_name": "SAG T1 TSE",
            },
        ]
    }
    exports = {
        "exports": [
            {
                "export_id": f"image-{index}",
                "source_relative_path": item["relative_path"],
            }
            for index, item in enumerate(inventory["instances"], start=1)
        ]
    }
    result = select_generalist_exports(inventory, exports)
    assert result["counts_by_role"]["mpr_sagittal"] == 3
    assert result["total_exports"] == 6
