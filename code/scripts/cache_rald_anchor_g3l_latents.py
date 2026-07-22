#!/usr/bin/env python3
"""Cache train-only posterior means from a passing G3L-1 physical VAE."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from models.rald_anchor import normalize_rae_coordinates  # noqa: E402
from models.cube_doppler import query_cube_spectrum  # noqa: E402
from models.rald_anchor_ldm import RaLDPointStatePosteriorEncoder  # noqa: E402


PROTOCOL = "rald_anchor_g3l2_train_latent_cache_v1"
G3L1_GATE_PROTOCOL = "rald_anchor_g3l1_physical_vae_gate_v1"
SCHEMA_VERSION = 1
LATENT_COUNT = 512
LATENT_DIM = 32
SPECTRUM_BINS = 64
LATENT_SHAPE = (LATENT_COUNT, LATENT_DIM)
LATENT_ARRAY_KEY = "latent_mean"
LATENT_METADATA_KEYS = (
    "cache_source_commit",
    "manifest_sha256",
    "scene_split_sha256",
    "normalization_sha256",
    "g3l1_report_sha256",
    "g3l1_config_sha256",
    "g3l1_checkpoint_sha256",
    "g3l1_source_commit",
    "parent_config_sha256",
    "parent_checkpoint_sha256",
)
LATENT_KEYS = {
    "schema_version",
    "sequence",
    "radar_index",
    "partition",
    LATENT_ARRAY_KEY,
    *LATENT_METADATA_KEYS,
}
G3L1_CONFIG_KEYS = (
    "protocol",
    "latent_count",
    "latent_dim",
    "model_dim",
    "seed",
    "decoder_depth",
    "denoiser_depth",
    "edm_steps",
)
G3L1_PROVENANCE_KEYS = (
    "git_commit",
    "manifest_sha256",
    "scene_split_sha256",
    "normalization_sha256",
    "g3r_selected_config",
    "g3r_selected_config_sha256",
    "g3r_selected_checkpoint",
    "g3r_selected_checkpoint_sha256",
    "g3r_geometry_parent_checkpoint",
    "g3r_geometry_parent_checkpoint_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _require_keys(document: dict, keys: tuple[str, ...], description: str) -> None:
    missing = [key for key in keys if key not in document]
    if missing:
        raise ValueError(f"{description} lacks required keys: {missing}")


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return document


def _resolved_report_run(report: dict, seed: int) -> Path:
    selected_runs = report.get("selected_runs")
    if not isinstance(selected_runs, dict):
        raise ValueError("G3L-1 gate report lacks selected_runs")
    selected = selected_runs.get(str(seed))
    if not isinstance(selected, str) or not selected:
        raise ValueError(f"G3L-1 gate report lacks selected run for seed {seed}")
    return Path(selected).expanduser().resolve()


def validate_g3l1_run(
    run: Path,
    report_path: Path,
    *,
    manifest_hash: str,
    scene_split_hash: str,
    normalization_hash: str,
    g3l1_source_commit: str,
) -> dict:
    """Validate a passing G3L-1 run and every live parent artifact hash."""

    run = run.expanduser().resolve()
    config_path = (run / "config.json").resolve()
    checkpoint_path = (run / "best.pt").resolve()
    report_path = report_path.expanduser().resolve()
    config_document = _load_json(config_path)
    report = _load_json(report_path)
    if report.get("protocol") != G3L1_GATE_PROTOCOL:
        raise ValueError("G3L-1 gate report protocol differs")
    if report.get("decision", {}).get("g3l1_passed") is not True:
        raise ValueError("G3L-2 accepts only a passing G3L-1 VAE run")
    if report.get("best_of_k") is not False:
        raise ValueError("G3L-1 gate must attest that best-of-k was not used")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    config = config_document.get("config")
    provenance = config_document.get("provenance")
    if not isinstance(config, dict) or not isinstance(provenance, dict):
        raise ValueError("G3L-1 config must contain config and provenance objects")
    _require_keys(config, G3L1_CONFIG_KEYS, "G3L-1 config")
    _require_keys(provenance, G3L1_PROVENANCE_KEYS, "G3L-1 provenance")
    seed = int(config["seed"])
    if _resolved_report_run(report, seed) != run:
        raise ValueError("G3L-1 gate-selected run differs from requested run")
    if config["protocol"] != "rald_anchor_g3l_physical_vae_v1":
        raise ValueError("G3L-1 VAE training protocol differs")
    if (int(config["latent_count"]), int(config["latent_dim"])) != LATENT_SHAPE:
        raise ValueError("G3L-1 latent shape must be exactly 512x32")
    if int(config["decoder_depth"]) != 24:
        raise ValueError("G3L-1 anchor decoder must use the official 24 layers")
    if int(config["denoiser_depth"]) != 24 or int(config["edm_steps"]) != 18:
        raise ValueError("G3L-1 must preserve the official EDM architecture contract")
    if provenance["git_commit"] != g3l1_source_commit:
        raise ValueError("G3L-1 source commit mismatch")
    expected_hashes = {
        "manifest_sha256": manifest_hash,
        "scene_split_sha256": scene_split_hash,
        "normalization_sha256": normalization_hash,
    }
    for key, expected in expected_hashes.items():
        if provenance[key] != expected:
            raise ValueError(f"G3L-1 {key} mismatch")
    config_hash = sha256(config_path)
    checkpoint_hash = sha256(checkpoint_path)
    selected_hashes_by_seed = report.get("selected_run_hashes")
    if not isinstance(selected_hashes_by_seed, dict):
        raise ValueError("G3L-1 gate report lacks selected_run_hashes")
    selected_hashes = selected_hashes_by_seed.get(str(seed))
    if not isinstance(selected_hashes, dict):
        raise ValueError(f"G3L-1 gate report lacks hashes for seed {seed}")
    if selected_hashes.get("config_sha256") != config_hash:
        raise ValueError("G3L-1 selected config hash mismatch")
    if selected_hashes.get("best_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("G3L-1 selected checkpoint hash mismatch")

    parent_config_path = Path(provenance["g3r_selected_config"]).expanduser().resolve()
    parent_checkpoint_path = Path(
        provenance["g3r_selected_checkpoint"]
    ).expanduser().resolve()
    parent_config_hash = sha256(parent_config_path)
    parent_checkpoint_hash = sha256(parent_checkpoint_path)
    if parent_config_hash != provenance["g3r_selected_config_sha256"]:
        raise ValueError("G3L-1 parent config hash mismatch")
    if parent_checkpoint_hash != provenance["g3r_selected_checkpoint_sha256"]:
        raise ValueError("G3L-1 parent checkpoint hash mismatch")
    geometry_parent = Path(
        provenance["g3r_geometry_parent_checkpoint"]
    ).expanduser().resolve()
    if sha256(geometry_parent) != provenance["g3r_geometry_parent_checkpoint_sha256"]:
        raise ValueError("G3L-1 frozen geometry-parent checkpoint hash mismatch")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config") != config:
        raise ValueError("G3L-1 checkpoint configuration differs from config.json")
    if checkpoint.get("provenance") != provenance:
        raise ValueError("G3L-1 checkpoint provenance differs from config.json")
    vae_state = checkpoint.get("g3l_vae")
    if not isinstance(vae_state, dict) or set(vae_state) != {
        "posterior_encoder",
        "anchor_decoder",
        "physical_head",
    }:
        raise ValueError("G3L-1 checkpoint lacks posterior encoder weights")

    return {
        "run": run,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "report_path": report_path,
        "config": config,
        "provenance": provenance,
        "checkpoint": checkpoint,
        "g3l1_report_sha256": sha256(report_path),
        "g3l1_config_sha256": config_hash,
        "g3l1_checkpoint_sha256": checkpoint_hash,
        "parent_config": parent_config_path,
        "parent_config_sha256": parent_config_hash,
        "parent_checkpoint": parent_checkpoint_path,
        "parent_checkpoint_sha256": parent_checkpoint_hash,
    }


def build_posterior_encoder(config: dict) -> RaLDPointStatePosteriorEncoder:
    return RaLDPointStatePosteriorEncoder(
        latent_count=int(config["latent_count"]),
        latent_dim=int(config["latent_dim"]),
        model_dim=int(config["model_dim"]),
        spectrum_bins=int(config["spectrum_bins"]),
    )


def load_posterior_state(model: RaLDPointStatePosteriorEncoder, checkpoint: dict) -> None:
    vae_state = checkpoint.get("g3l_vae")
    if isinstance(vae_state, dict) and isinstance(
        vae_state.get("posterior_encoder"), dict
    ):
        model.load_state_dict(vae_state["posterior_encoder"], strict=True)
        return
    direct = checkpoint.get("posterior_encoder")
    if isinstance(direct, dict):
        model.load_state_dict(direct, strict=True)
        return
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("G3L-1 checkpoint has no model state")
    for prefix in ("posterior_encoder.", "ldm.posterior_encoder."):
        extracted = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if extracted:
            model.load_state_dict(extracted, strict=True)
            return
    raise ValueError("G3L-1 checkpoint has no recognizable posterior encoder prefix")


def validate_data_contract(
    manifest_path: Path,
    scene_split_path: Path,
    normalization_path: Path,
) -> dict:
    manifest = _load_json(manifest_path)
    scene_split = _load_json(scene_split_path)
    normalization = _load_json(normalization_path)
    manifest_hash = sha256(manifest_path)
    scene_split_hash = sha256(scene_split_path)
    normalization_hash = sha256(normalization_path)
    if scene_split.get("gate_pass") is not True:
        raise ValueError("Scene split did not pass its leakage gate")
    if normalization.get("manifest_sha256") != manifest_hash:
        raise ValueError("Normalization manifest hash mismatch")
    if normalization.get("scene_split_sha256") != scene_split_hash:
        raise ValueError("Normalization scene-split hash mismatch")
    if normalization.get("partitions") != ["train"]:
        raise ValueError("G3L-2 normalization must be train-only")
    if normalization.get("frame_limit") is not None:
        raise ValueError("G3L-2 normalization must cover all train frames")
    if "log10_power_plus_one" not in normalization:
        raise ValueError("G3L-2 requires Full-RAED Cube normalization")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Training manifest contains no frames")
    train_frames = [frame for frame in frames if frame.get("partition") == "train"]
    if len(train_frames) != int(normalization.get("frame_count", -1)):
        raise ValueError("Train frame count differs from normalization")
    keys = [(int(frame["sequence"]), int(frame["radar_index"])) for frame in train_frames]
    if len(keys) != len(set(keys)):
        raise ValueError("Training manifest contains duplicate train frames")
    normalization_frames = normalization.get("frames")
    if normalization_frames is not None:
        normalized_keys = {
            (int(frame["sequence"]), int(frame["radar_index"]))
            for frame in normalization_frames
        }
        if set(keys) != normalized_keys:
            raise ValueError("Train frames differ from normalization frame set")
    return {
        "manifest": manifest,
        "scene_split": scene_split,
        "normalization": normalization,
        "train_frames": train_frames,
        "manifest_sha256": manifest_hash,
        "scene_split_sha256": scene_split_hash,
        "normalization_sha256": normalization_hash,
    }


def latent_metadata(configuration: dict) -> dict[str, str]:
    return {key: str(configuration[key]) for key in LATENT_METADATA_KEYS}


def latent_path(root: Path, sequence: int, radar_index: int) -> Path:
    return root / "latents" / f"seq{sequence:02d}_radar_{radar_index:05d}.npz"


def write_latent_file(
    path: Path,
    latent_mean: np.ndarray | torch.Tensor,
    metadata: dict[str, str],
    *,
    sequence: int,
    radar_index: int,
) -> None:
    array = (
        latent_mean.detach().float().cpu().numpy()
        if isinstance(latent_mean, torch.Tensor)
        else np.asarray(latent_mean, dtype=np.float32)
    )
    if array.shape != LATENT_SHAPE or not np.isfinite(array).all():
        raise ValueError("G3L-2 latent mean must be finite with shape 512x32")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int16),
        "sequence": np.asarray(sequence, dtype=np.int16),
        "radar_index": np.asarray(radar_index, dtype=np.int32),
        "partition": np.asarray("train"),
        LATENT_ARRAY_KEY: array.astype(np.float32, copy=False),
        **{key: np.asarray(metadata[key]) for key in LATENT_METADATA_KEYS},
    }
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def validate_latent_file(
    path: Path,
    metadata: dict[str, str],
    *,
    sequence: int | None = None,
    radar_index: int | None = None,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as cache:
        keys = set(cache.files)
        if keys != LATENT_KEYS:
            raise ValueError(
                f"G3L-2 latent schema differs in {path}: "
                f"missing={sorted(LATENT_KEYS - keys)}, "
                f"unexpected={sorted(keys - LATENT_KEYS)}"
            )
        if int(cache["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(f"G3L-2 latent schema version differs in {path}")
        if str(np.asarray(cache["partition"]).item()) != "train":
            raise ValueError(f"G3L-2 latent is not train-only: {path}")
        for key in LATENT_METADATA_KEYS:
            if str(np.asarray(cache[key]).item()) != metadata[key]:
                raise ValueError(f"G3L-2 latent metadata {key} differs in {path}")
        cached_sequence = int(cache["sequence"])
        cached_radar_index = int(cache["radar_index"])
        if sequence is not None and cached_sequence != sequence:
            raise ValueError(f"G3L-2 latent sequence differs in {path}")
        if radar_index is not None and cached_radar_index != radar_index:
            raise ValueError(f"G3L-2 latent radar index differs in {path}")
        latent = cache[LATENT_ARRAY_KEY]
        if latent.dtype != np.float32:
            raise ValueError(f"G3L-2 latent dtype must be float32 in {path}")
        if latent.shape != LATENT_SHAPE:
            raise ValueError(f"G3L-2 latent shape differs in {path}: {latent.shape}")
        if not np.isfinite(latent).all():
            raise ValueError(f"G3L-2 latent contains non-finite values in {path}")
        return latent.copy()


def validate_latent_record(
    root: Path,
    record: dict,
    metadata: dict[str, str],
) -> np.ndarray:
    path = Path(record["path"])
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != record.get("sha256"):
        raise ValueError(f"G3L-2 latent file hash differs: {path}")
    return validate_latent_file(
        path,
        metadata,
        sequence=int(record["sequence"]),
        radar_index=int(record["radar_index"]),
    )


def validate_latent_cache_manifest(
    path: Path,
    *,
    expected_configuration: dict | None = None,
    validate_files: bool = True,
) -> dict:
    document = _load_json(path)
    if document.get("protocol") != PROTOCOL:
        raise ValueError("G3L-2 latent cache protocol differs")
    if int(document.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("G3L-2 latent cache schema version differs")
    if document.get("partitions") != ["train"]:
        raise ValueError("G3L-2 latent cache must contain train only")
    if document.get("latent_shape") != list(LATENT_SHAPE):
        raise ValueError("G3L-2 latent cache shape differs")
    if document.get("test_accessed") is not False:
        raise ValueError("G3L-2 latent cache must attest no test access")
    configuration = document.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("G3L-2 latent cache lacks configuration")
    if expected_configuration is not None and configuration != expected_configuration:
        raise ValueError("G3L-2 latent cache configuration or source mismatch")
    records = document.get("records")
    if not isinstance(records, list) or len(records) != int(
        document.get("frame_count", -1)
    ):
        raise ValueError("G3L-2 latent cache frame count differs")
    identities = [
        (int(record["sequence"]), int(record["radar_index"])) for record in records
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("G3L-2 latent cache contains duplicate frames")
    if any(record.get("partition") != "train" for record in records):
        raise ValueError("G3L-2 latent cache contains a non-train record")
    if validate_files:
        root = path.resolve().parent
        metadata = latent_metadata(configuration)
        for record in records:
            validate_latent_record(root, record, metadata)
    return document


def require_h200_cuda(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("G3L-2 latent caching requires CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("G3L-2 latent caching is CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"G3L-2 latent caching requires H200, got {resolved}")
    return device, resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--g3l1-run", type=Path, required=True)
    parser.add_argument("--g3l1-report", type=Path, required=True)
    parser.add_argument("--g3l1-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    output_nonempty = args.output.exists() and any(args.output.iterdir())
    if output_nonempty and args.overwrite:
        shutil.rmtree(args.output)
        output_nonempty = False
    if output_nonempty and not args.resume:
        raise FileExistsError(
            f"Cache directory is not empty: {args.output}; use --resume or --overwrite"
        )
    if args.resume and not output_nonempty:
        raise FileNotFoundError(f"No G3L-2 cache to resume: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    device, gpu_name = require_h200_cuda(args.device)
    source_commit = args.source_commit or git_commit(Path(__file__).resolve().parents[2])
    if source_commit is None:
        raise RuntimeError("Cache source commit is required")
    data = validate_data_contract(args.manifest, args.scene_split, args.normalization)
    g3l1 = validate_g3l1_run(
        args.g3l1_run,
        args.g3l1_report,
        manifest_hash=data["manifest_sha256"],
        scene_split_hash=data["scene_split_sha256"],
        normalization_hash=data["normalization_sha256"],
        g3l1_source_commit=args.g3l1_source_commit,
    )
    configuration = {
        "protocol": PROTOCOL,
        "cache_source_commit": source_commit,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": data["manifest_sha256"],
        "scene_split": str(args.scene_split.resolve()),
        "scene_split_sha256": data["scene_split_sha256"],
        "normalization": str(args.normalization.resolve()),
        "normalization_sha256": data["normalization_sha256"],
        "g3l1_run": str(g3l1["run"]),
        "g3l1_report": str(g3l1["report_path"]),
        "g3l1_report_sha256": g3l1["g3l1_report_sha256"],
        "g3l1_config": str(g3l1["config_path"]),
        "g3l1_config_sha256": g3l1["g3l1_config_sha256"],
        "g3l1_checkpoint": str(g3l1["checkpoint_path"]),
        "g3l1_checkpoint_sha256": g3l1["g3l1_checkpoint_sha256"],
        "g3l1_source_commit": args.g3l1_source_commit,
        "parent_config": str(g3l1["parent_config"]),
        "parent_config_sha256": g3l1["parent_config_sha256"],
        "parent_checkpoint": str(g3l1["parent_checkpoint"]),
        "parent_checkpoint_sha256": g3l1["parent_checkpoint_sha256"],
        "latent_shape": list(LATENT_SHAPE),
        "partitions": ["train"],
        "cfar_query_helper": False,
        "best_of_k": False,
        "test_accessed": False,
    }
    config_path = args.output / "cache_config.json"
    if args.resume:
        if _load_json(config_path) != configuration:
            raise ValueError("Resume cache configuration or source mismatch")
    else:
        atomic_json(config_path, configuration)

    progress_path = args.output / "cache_progress.json"
    completed_records: dict[tuple[int, int], dict] = {}
    if args.resume and progress_path.is_file():
        progress = _load_json(progress_path)
        if progress.get("configuration") != configuration:
            raise ValueError("Resume cache progress configuration or source mismatch")
        for record in progress.get("records", []):
            identity = (int(record["sequence"]), int(record["radar_index"]))
            if identity in completed_records:
                raise ValueError("Resume cache progress contains duplicate frames")
            completed_records[identity] = record

    posterior = build_posterior_encoder(g3l1["config"]).to(device)
    load_posterior_state(posterior, g3l1["checkpoint"])
    posterior.eval().requires_grad_(False)
    dataset = KRadarCubeDataset(
        args.data_root, args.cache_root, args.manifest, ("train",)
    )
    metadata = latent_metadata(configuration)
    records = []
    for position in range(len(dataset)):
        item = dataset[position]
        sequence = int(item["sequence"])
        radar_index = int(item["radar_index"])
        path = latent_path(args.output, sequence, radar_index)
        identity = (sequence, radar_index)
        prior_record = completed_records.get(identity)
        if prior_record is not None:
            if sha256(path) != prior_record.get("sha256"):
                raise ValueError(f"Resume detected a tampered G3L-2 latent: {path}")
            validate_latent_file(
                path, metadata, sequence=sequence, radar_index=radar_index
            )
        else:
            if path.exists():
                # A file without an atomic progress record may be a crash remnant.
                path.unlink()
            cube = item["cube_drae"].unsqueeze(0).to(device, non_blocking=True)
            target = item["target_xyz_confidence"].to(device, non_blocking=True)
            target_index = item["target_rae_index"].to(device, non_blocking=True)
            normalized_rae = normalize_rae_coordinates(
                target_index.to(cube), tuple(int(size) for size in cube.shape[2:])
            )
            probability = query_cube_spectrum(cube, target_index)
            confidence = target[:, 3].clamp(0.0, 1.0)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = posterior(
                    normalized_rae.unsqueeze(0),
                    probability.unsqueeze(0),
                    confidence.unsqueeze(0),
                )
            write_latent_file(
                path,
                encoded.mean[0],
                metadata,
                sequence=sequence,
                radar_index=radar_index,
            )
            validate_latent_file(
                path, metadata, sequence=sequence, radar_index=radar_index
            )
            del cube, target, target_index, normalized_rae, probability, encoded
        record = {
            "sequence": sequence,
            "radar_index": radar_index,
            "partition": "train",
            "path": str(path.relative_to(args.output)),
            "sha256": sha256(path),
        }
        records.append(record)
        completed_records[identity] = record
        atomic_json(
            progress_path,
            {
                "protocol": PROTOCOL,
                "configuration": configuration,
                "records": records,
            },
        )
        print(
            json.dumps(
                {
                    "completed": position + 1,
                    "total": len(dataset),
                    "sequence": sequence,
                    "radar_index": radar_index,
                }
            ),
            flush=True,
        )

    summary = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "partitions": ["train"],
        "frame_count": len(records),
        "latent_shape": list(LATENT_SHAPE),
        "test_accessed": False,
        "device": gpu_name,
        "records": records,
    }
    summary_path = args.output / "g3l2_latent_cache_manifest.json"
    atomic_json(summary_path, summary)
    validate_latent_cache_manifest(
        summary_path, expected_configuration=configuration, validate_files=True
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
