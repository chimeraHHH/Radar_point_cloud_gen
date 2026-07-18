#!/usr/bin/env python3
"""Select one static-Doppler SNR slice using train-only evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_QUANTILES = (0.0, 0.5, 0.75, 0.9)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_record(path: Path, report: dict) -> dict:
    protocol = report["protocol"]
    hypothesis = report["train"]["selected_hypothesis"]
    train_error = report["train"]["hypotheses"][hypothesis][
        "frame_median_error_median_mps"
    ]
    validation_error = report["validation"]["hypotheses"][hypothesis][
        "frame_median_error_median_mps"
    ]
    random_baseline = protocol["random_circular_median_baseline_mps"]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "background_snr_quantile": float(protocol["background_snr_quantile"]),
        "train_selected_hypothesis": hypothesis,
        "train_selected_error_mps": float(train_error),
        "train_selection_margin_mps": float(
            report["train"]["selected_margin_to_second_mps"]
        ),
        "train_background_point_count": int(
            report["train"]["background_point_count"]
        ),
        "validation_frozen_error_mps": float(validation_error),
        "random_circular_median_baseline_mps": float(random_baseline),
        "validation_beats_random": bool(validation_error < random_baseline),
        "data_checks_passed": bool(
            report["checks"]["required_frame_count"]
            and report["checks"]["no_frame_errors"]
        ),
        "report_passed": bool(report["passed"]),
    }


def select_train_only(candidates: list[dict], minimum_margin_mps: float) -> dict:
    eligible = [
        candidate
        for candidate in candidates
        if candidate["data_checks_passed"]
        and candidate["train_selection_margin_mps"] >= minimum_margin_mps
    ]
    if not eligible:
        raise ValueError("No SNR slice meets the frozen train selection margin")
    return min(
        eligible,
        key=lambda candidate: (
            candidate["train_selected_error_mps"],
            candidate["background_snr_quantile"],
        ),
    )


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-markdown", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (
        args.output.exists() or args.decision_markdown.exists()
    ) and not args.overwrite:
        raise FileExistsError("Static-Doppler selection output already exists")
    documents = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.reports
    ]
    common_source = {document["source_commit"] for document in documents}
    common_manifest = {document["manifest_sha256"] for document in documents}
    fixed_protocols = {
        json.dumps(
            {
                key: value
                for key, value in document["protocol"].items()
                if key != "background_snr_quantile"
            },
            sort_keys=True,
        )
        for document in documents
    }
    if (
        len(common_source) != 1
        or len(common_manifest) != 1
        or len(fixed_protocols) != 1
    ):
        raise ValueError("Static-Doppler candidates have mismatched provenance")
    candidates = [
        candidate_record(path, document)
        for path, document in zip(args.reports, documents)
    ]
    quantiles = sorted(
        candidate["background_snr_quantile"] for candidate in candidates
    )
    if quantiles != list(EXPECTED_QUANTILES):
        raise ValueError(f"Expected SNR quantiles {EXPECTED_QUANTILES}, found {quantiles}")
    minimum_margins = {
        document["protocol"]["minimum_selection_margin_mps"]
        for document in documents
    }
    if len(minimum_margins) != 1:
        raise ValueError("Static-Doppler candidates use different train margins")
    minimum_margin = float(minimum_margins.pop())
    selected = select_train_only(candidates, minimum_margin)
    checks = {
        "exact_candidate_grid": True,
        "candidate_provenance_matches": True,
        "selected_by_train_only_error": True,
        "selected_data_checks_passed": selected["data_checks_passed"],
        "selected_train_margin_passed": selected["train_selection_margin_mps"]
        >= minimum_margin,
        "selected_validation_beats_random": selected["validation_beats_random"],
        "selected_report_passed": selected["report_passed"],
    }
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "candidate_source_commit": common_source.pop(),
        "manifest_sha256": common_manifest.pop(),
        "selection_partition": "train",
        "selection_rule": (
            "minimum train selected-hypothesis frame-median error among slices "
            "meeting the frozen train selection margin; ties prefer lower quantile"
        ),
        "minimum_selection_margin_mps": minimum_margin,
        "candidates": sorted(
            candidates, key=lambda candidate: candidate["background_snr_quantile"]
        ),
        "selected": selected,
        "checks": checks,
        "passed": all(checks.values()),
        "completed": True,
    }
    lines = [
        "# Static Doppler SNR recovery decision",
        "",
        f"- Selected quantile: `{selected['background_snr_quantile']}`",
        f"- Train-selected hypothesis: `{selected['train_selected_hypothesis']}`",
        f"- Train error: `{selected['train_selected_error_mps']:.6f} m/s`",
        f"- Validation error: `{selected['validation_frozen_error_mps']:.6f} m/s`",
        f"- Random baseline: `{selected['random_circular_median_baseline_mps']:.6f} m/s`",
        f"- Passed: `{report['passed']}`",
        "",
        "Selection used train metrics only. Validation was read only after the "
        "fixed rule selected a slice.",
    ]
    atomic_text(args.output, json.dumps(report, indent=2) + "\n")
    atomic_text(args.decision_markdown, "\n".join(lines) + "\n")
    print(json.dumps({"selected": selected, "checks": checks}, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
