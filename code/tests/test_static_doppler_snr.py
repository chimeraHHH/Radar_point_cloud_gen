from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from scripts.audit_kradar_static_doppler import (  # noqa: E402
    filter_background_by_snr_quantile,
)
from scripts.select_static_doppler_snr import select_train_only  # noqa: E402


def test_background_snr_quantile_keeps_high_snr_points() -> None:
    cfar = np.zeros((4, 6), dtype=np.float64)
    cfar[:, 5] = [1.0, 2.0, 3.0, 4.0]
    selected, threshold = filter_background_by_snr_quantile(
        cfar, np.ones(4, dtype=bool), 0.5
    )

    assert threshold == 2.5
    assert selected.tolist() == [False, False, True, True]


def test_train_only_selection_ignores_validation_outcome() -> None:
    candidates = [
        {
            "background_snr_quantile": 0.0,
            "train_selected_error_mps": 0.4,
            "train_selection_margin_mps": 0.1,
            "data_checks_passed": True,
            "validation_beats_random": False,
        },
        {
            "background_snr_quantile": 0.5,
            "train_selected_error_mps": 0.5,
            "train_selection_margin_mps": 0.1,
            "data_checks_passed": True,
            "validation_beats_random": True,
        },
    ]

    selected = select_train_only(candidates, minimum_margin_mps=0.05)

    assert selected["background_snr_quantile"] == 0.0
