from scripts.compare_g1b_formal import validate_pairs


def run(mode: str, parameters: int) -> dict:
    return {
        "frames": {(1, 1): {}},
        "config": {"mode": mode, "seed": 1, "epochs": 50},
        "provenance": {
            "manifest_sha256": "m",
            "scene_split_sha256": "s",
            "normalization_sha256": "n",
            "git_commit": "g",
            "torch_version": "t",
            "device": "NVIDIA H200 NVL",
            "model_parameter_count": parameters,
        },
    }


def test_validate_g1b_pairs_allows_only_bounded_matched_candidate() -> None:
    baseline = {1: run("rae_max", 10_000)}
    candidate = {1: run("rae_circular_harmonics", 10_050)}

    report = validate_pairs(baseline, candidate, "rae_circular_harmonics")

    assert report["passed"] is True
    candidate[1]["provenance"]["model_parameter_count"] = 10_200
    try:
        validate_pairs(baseline, candidate, "rae_circular_harmonics")
    except ValueError as error:
        assert "1% parameter budget" in str(error)
    else:
        raise AssertionError("Over-budget G1B candidate was accepted")
