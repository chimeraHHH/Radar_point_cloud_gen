import pytest

from scripts.preflight_g1g_max_target import select_max_target_index


def test_max_target_selection_is_deterministic_and_first_on_ties() -> None:
    assert select_max_target_index([10, 30, 20, 30]) == 1


@pytest.mark.parametrize("counts", ([], [1, 0], [-1]))
def test_max_target_selection_rejects_invalid_counts(counts: list[int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        select_max_target_index(counts)
