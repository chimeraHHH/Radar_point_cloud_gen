from pathlib import Path
from types import SimpleNamespace

from scripts.queue_rald_anchor_g4r import train_command


def test_g4r_train_command_uses_rald_native_trainer() -> None:
    args = SimpleNamespace(
        repo_root=Path("/repo"),
        data_root=Path("/data"),
        cache_root=Path("/cache"),
        manifest=Path("/manifest.json"),
        scene_split=Path("/split.json"),
        normalization=Path("/normalization.json"),
        dense_cache_report=Path("/dense.json"),
        g3r_summary=Path("/g3r.json"),
        g3r_source_commit="g3r-source",
        source_commit="source",
    )

    command = train_command(
        Path("/python"),
        args,
        Path("/parent"),
        Path("/parent-cache"),
        Path("/output"),
        "latent",
        20260716,
        20,
    )

    assert command[2] == "/repo/code/scripts/train_rald_anchor_temporal.py"
    assert command[command.index("--fusion-mode") + 1] == "latent"
    assert command[command.index("--g3r-source-commit") + 1] == "g3r-source"
    assert command[command.index("--temporal-warmup-epochs") + 1] == "5"
    assert "static" not in " ".join(command).lower()
