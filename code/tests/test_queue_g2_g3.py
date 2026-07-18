from pathlib import Path

from scripts.queue_g2_g3 import g1_parent_runs


def test_g1_parents_are_resolved_beside_comparison() -> None:
    comparison = Path("/runs/formal_g1/g1_comparison_a7d06db1.json")

    parents = g1_parent_runs(
        comparison,
        "a7d06db1abcc69c20dfed381f0c2909b1a89f026",
        [20260716, 20260717, 20260718],
    )

    assert parents == {
        20260716: Path(
            "/runs/formal_g1/g1_full_raed_seed20260716_a7d06db1"
        ),
        20260717: Path(
            "/runs/formal_g1/g1_full_raed_seed20260717_a7d06db1"
        ),
        20260718: Path(
            "/runs/formal_g1/g1_full_raed_seed20260718_a7d06db1"
        ),
    }
