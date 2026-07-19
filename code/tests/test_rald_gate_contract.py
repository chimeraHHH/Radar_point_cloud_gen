import json
from pathlib import Path

from g1b_contract import FROZEN_G1B_SEEDS, sha256
from rald_gate_contract import validate_g3r_selected_runs


def test_g3r_contract_binds_selected_full_runs(tmp_path: Path) -> None:
    source = "source"
    runs = {}
    hashes = {}
    for seed in FROZEN_G1B_SEEDS:
        run = tmp_path / f"seed{seed}"
        run.mkdir()
        (run / "config.json").write_text(
            json.dumps(
                {
                    "config": {
                        "seed": seed,
                        "cycle_variant": "full",
                        "doppler_head_mode": "distribution",
                    },
                    "provenance": {"git_commit": source},
                }
            ),
            encoding="utf-8",
        )
        (run / "best.pt").write_bytes(f"checkpoint-{seed}".encode())
        runs[str(seed)] = str(run)
        hashes[str(seed)] = {
            "config_sha256": sha256(run / "config.json"),
            "best_checkpoint_sha256": sha256(run / "best.pt"),
        }
    comparison = tmp_path / "g3r.json"
    comparison.write_text(
        json.dumps(
            {
                "decision": {"g3r_passed": True},
                "runs": {"full": runs},
                "run_hashes": {"full": hashes},
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "status": "g3r_passed",
        "source_commit": source,
        "seeds": list(FROZEN_G1B_SEEDS),
        "selected_arm": "full",
        "selected_runs": runs,
        "selected_run_hashes": hashes,
        "g3r_comparison": str(comparison),
        "g3r_comparison_sha256": sha256(comparison),
    }

    assert validate_g3r_selected_runs(summary, source) == {
        seed: Path(runs[str(seed)]).resolve() for seed in FROZEN_G1B_SEEDS
    }
    (Path(runs[str(FROZEN_G1B_SEEDS[0])]) / "best.pt").write_bytes(b"changed")
    try:
        validate_g3r_selected_runs(summary, source)
    except ValueError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("Changed selected checkpoint was accepted")

