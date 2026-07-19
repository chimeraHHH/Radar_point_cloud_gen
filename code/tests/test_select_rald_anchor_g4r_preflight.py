from scripts.select_rald_anchor_g4r_preflight import select


def run(score: float, confidence: float = 1.0, coverage: float = 100.0) -> dict:
    return {
        "score": score,
        "confidence": confidence,
        "coverage": coverage,
        "parent_checkpoint_sha256": "parent",
        "parent_prediction_manifest_sha256": "cache",
    }


def test_preflight_selects_best_eligible_rald_level() -> None:
    runs = {
        "token": run(3.0),
        "latent": run(2.0),
        "query": run(1.0, confidence=0.5),
    }

    assert select(runs) == "latent"


def test_preflight_tie_prefers_rald_mixed_latent() -> None:
    runs = {mode: run(1.0) for mode in ("token", "latent", "query")}

    assert select(runs) == "latent"
