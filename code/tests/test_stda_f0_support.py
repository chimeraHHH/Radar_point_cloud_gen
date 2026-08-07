from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import pytest

import eval.stda_f0_support as support_module
from eval.stda_f0_support import (
    BUILDER_SPACING_SELF_CHECK_AUTHORITATIVE,
    CAPACITY_NO_GO_STATUS,
    FORMAL_SPACING_VERIFIER_MODULE,
    PROTOCOL_FREEZE_COMMIT,
    PROTOCOL_SHA256,
    SUPPORT_COLUMN_SCHEMA,
    SpacingViolationError,
    SupportSerializationError,
    build_packed_support,
    builder_self_check_serialized_support_spacing,
    builder_self_check_serialized_xyz_spacing,
    candidate_field_digest_sha256,
    commit_packed_support,
    deserialize_packed_support,
    exact_float32_cells,
    exact_float32_floor_20,
    packed_support_digest_sha256,
    serialize_packed_support,
    serialize_xyz_float32,
    stable_candidate_ids,
    support_serialization_schema,
)


EXPECTED_FREEZE_COMMIT = "eb839e1e806c44dd1668051085a307faf4fe83a0"
EXPECTED_PROTOCOL_SHA256 = (
    "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
)


def _permutation_candidates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(
        [
            [0.001, 0.0, 0.0],
            [0.002, 0.0, 0.0],
            [0.125, 0.0, 0.0],
            [0.0, 0.125, 0.0],
            [0.0, 0.0, 0.125],
            [0.0625, 0.0, 0.0],
            [0.0, 0.0625, 0.0],
            [0.0, 0.0, 0.0625],
        ],
        dtype="<f4",
    )
    confidence = np.asarray(
        [0.9, 0.9, 0.7, 0.8, 0.6, 0.5, 0.4, 0.3],
        dtype="<f4",
    )
    candidate_ids = np.asarray([80, 10, 70, 30, 50, 20, 60, 40], dtype="<i8")
    return xyz, confidence, candidate_ids


def test_freeze_constants_and_target_free_api_are_bound() -> None:
    assert PROTOCOL_FREEZE_COMMIT == EXPECTED_FREEZE_COMMIT
    assert PROTOCOL_SHA256 == EXPECTED_PROTOCOL_SHA256
    assert tuple(inspect.signature(build_packed_support).parameters) == (
        "candidate_xyz_m",
        "base_confidence",
        "candidate_ids",
    )
    assert all(
        "target" not in name
        for name in inspect.signature(build_packed_support).parameters
    )
    assert "candidate_model" not in inspect.signature(
        build_packed_support
    ).parameters


def test_builder_spacing_check_is_explicitly_non_authoritative() -> None:
    assert BUILDER_SPACING_SELF_CHECK_AUTHORITATIVE is False
    assert FORMAL_SPACING_VERIFIER_MODULE == "eval.stda_f0_verify"
    assert not hasattr(support_module, "verify_serialized_support_spacing")


def test_stable_candidate_ids_are_little_endian_immutable_rows() -> None:
    ids = stable_candidate_ids(5)

    np.testing.assert_array_equal(ids, np.arange(5, dtype="<i8"))
    assert ids.dtype.str == "<i8"
    assert not ids.flags.writeable
    with pytest.raises(ValueError):
        stable_candidate_ids(-1)
    with pytest.raises(ValueError):
        stable_candidate_ids(True)


def test_exact_float32_floor_handles_negative_boundary_and_ulps() -> None:
    positive = np.float32(0.25)
    positive_below = np.nextafter(positive, np.float32(-np.inf))
    positive_above = np.nextafter(positive, np.float32(np.inf))
    negative = np.float32(-0.25)
    negative_below = np.nextafter(negative, np.float32(-np.inf))
    negative_above = np.nextafter(negative, np.float32(np.inf))

    assert exact_float32_floor_20(positive_below) == 4
    assert exact_float32_floor_20(positive) == 5
    assert exact_float32_floor_20(positive_above) == 5
    assert exact_float32_floor_20(negative_below) == -6
    assert exact_float32_floor_20(negative) == -5
    assert exact_float32_floor_20(negative_above) == -5
    assert exact_float32_floor_20(np.float32(0.0)) == 0
    assert exact_float32_floor_20(np.float32(-0.0)) == 0

    five_cm = np.float32(0.05)
    five_cm_below = np.nextafter(five_cm, np.float32(-np.inf))
    assert exact_float32_floor_20(five_cm_below) == 0
    assert exact_float32_floor_20(five_cm) == 1
    assert exact_float32_floor_20(np.float32(-0.05)) == -2


def test_exact_float32_cells_preserve_signed_floor_semantics() -> None:
    xyz = np.asarray(
        [
            [-0.25, -0.0, 0.25],
            [-0.05, 0.05, 0.0],
        ],
        dtype="<f4",
    )

    cells = exact_float32_cells(xyz)

    np.testing.assert_array_equal(cells, [[-5, 0, 5], [-2, 1, 0]])
    assert cells.dtype.str == "<i8"
    assert not cells.flags.writeable


def test_within_cell_winner_uses_confidence_then_smallest_stable_id() -> None:
    xyz = np.asarray(
        [
            [0.001, 0.001, 0.001],
            [0.002, 0.001, 0.001],
            [0.003, 0.001, 0.001],
            [0.004, 0.001, 0.001],
        ],
        dtype="<f4",
    )
    confidence = np.asarray([0.8, 0.9, 0.9, 0.7], dtype="<f4")
    candidate_ids = np.asarray([1, 9, 4, 2], dtype="<i8")

    support = build_packed_support(xyz, confidence, candidate_ids)

    np.testing.assert_array_equal(support.stable_candidate_id, [4])
    np.testing.assert_array_equal(support.base_confidence, [np.float32(0.9)])
    assert support.color_cardinalities == (1, 0, 0, 0, 0, 0, 0, 0)


def test_largest_color_tie_uses_smallest_binary_color_id() -> None:
    xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0625],
            [0.0, 0.0625, 0.0],
            [0.0625, 0.0, 0.0],
        ],
        dtype="<f4",
    )
    confidence = np.ones(4, dtype="<f4")
    candidate_ids = np.asarray([8, 6, 4, 2], dtype="<i8")

    support = build_packed_support(xyz, confidence, candidate_ids)

    assert support.color_cardinalities == (1, 1, 1, 0, 1, 0, 0, 0)
    assert support.selected_color_id == 0
    np.testing.assert_array_equal(support.stable_candidate_id, [8])
    np.testing.assert_array_equal(support.color_id, [0])


def test_support_and_candidate_hashes_are_input_permutation_invariant() -> None:
    xyz, confidence, candidate_ids = _permutation_candidates()
    permutation = np.asarray([4, 0, 7, 2, 6, 1, 5, 3])

    first = build_packed_support(xyz, confidence, candidate_ids)
    second = build_packed_support(
        xyz[permutation],
        confidence[permutation],
        candidate_ids[permutation],
    )

    assert first.candidate_field_sha256 == second.candidate_field_sha256
    assert first.digest_sha256 == second.digest_sha256
    assert serialize_packed_support(first) == serialize_packed_support(second)
    np.testing.assert_array_equal(
        first.stable_candidate_id,
        np.sort(first.stable_candidate_id),
    )
    assert candidate_field_digest_sha256(
        xyz,
        confidence,
        candidate_ids,
    ) == candidate_field_digest_sha256(
        xyz[permutation],
        confidence[permutation],
        candidate_ids[permutation],
    )


def test_serialization_exposes_canonical_little_endian_schema() -> None:
    xyz, confidence, candidate_ids = _permutation_candidates()
    support = build_packed_support(xyz, confidence, candidate_ids)
    serialized = serialize_packed_support(support)
    header_bytes, payload = serialized.split(b"\n", 1)
    header = json.loads(header_bytes.decode("ascii"))
    schema = support_serialization_schema()

    assert header_bytes + b"\n" == (
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    assert header["columns"] == [
        {"name": name, "dtype": dtype} for name, dtype in SUPPORT_COLUMN_SCHEMA
    ]
    assert header["framing"] == schema["framing"]
    assert schema["bytes_per_row"] == 49
    assert len(payload) == support.support_count * schema["bytes_per_row"]
    assert hashlib.sha256(payload).hexdigest() == header["payload_sha256"]


def test_serialization_roundtrip_rehash_and_builder_commitment() -> None:
    xyz, confidence, candidate_ids = _permutation_candidates()
    support = build_packed_support(xyz, confidence, candidate_ids)
    serialized = serialize_packed_support(support)

    reloaded = deserialize_packed_support(
        serialized,
        expected_sha256=support.digest_sha256,
    )
    commitment = commit_packed_support(
        serialized,
        expected_sha256=support.digest_sha256,
    )

    assert serialize_packed_support(reloaded) == serialized
    assert packed_support_digest_sha256(serialized) == support.digest_sha256
    np.testing.assert_array_equal(reloaded.stable_candidate_id, [10, 30, 50, 70])
    assert all(
        not values.flags.writeable
        for values in (
            reloaded.stable_candidate_id,
            reloaded.cell_xyz,
            reloaded.xyz_m,
            reloaded.base_confidence,
            reloaded.color_id,
        )
    )
    assert commitment.support_sha256 == support.digest_sha256
    assert commitment.builder_spacing_self_check_passed
    assert not commitment.formal_independent_spacing_verified
    assert not commitment.capacity_status.capacity_sufficient


def test_payload_and_full_hash_corruption_are_rejected() -> None:
    xyz, confidence, candidate_ids = _permutation_candidates()
    support = build_packed_support(xyz, confidence, candidate_ids)
    serialized = serialize_packed_support(support)
    corrupted = bytearray(serialized)
    corrupted[-1] ^= 1

    with pytest.raises(SupportSerializationError, match="payload SHA-256"):
        deserialize_packed_support(bytes(corrupted))
    with pytest.raises(SupportSerializationError, match="SHA-256 mismatch"):
        deserialize_packed_support(serialized, expected_sha256="00" * 32)

    header_bytes, payload = serialized.split(b"\n", 1)
    header = json.loads(header_bytes.decode("ascii"))
    header["capacity_sufficient"] = 0
    mistyped_header = (
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
        + payload
    )
    with pytest.raises(SupportSerializationError, match="semantics changed"):
        deserialize_packed_support(mistyped_header)


def test_builder_self_check_rejects_4p9999cm_and_ulp_below_5cm() -> None:
    four_point_9999_cm = np.asarray(
        [[0.0, 0.0, 0.0], [0.049999, 0.0, 0.0]],
        dtype="<f4",
    )
    five_cm = np.float32(0.05)
    ulp_below = np.nextafter(five_cm, np.float32(-np.inf))
    ulp_pair = np.asarray(
        [[0.0, 0.0, 0.0], [ulp_below, 0.0, 0.0]],
        dtype="<f4",
    )

    with pytest.raises(SpacingViolationError):
        builder_self_check_serialized_xyz_spacing(
            serialize_xyz_float32(four_point_9999_cm)
        )
    with pytest.raises(SpacingViolationError):
        builder_self_check_serialized_xyz_spacing(
            serialize_xyz_float32(ulp_pair)
        )


def test_builder_self_check_accepts_serialized_5cm_and_exact_equality() -> None:
    positive = np.asarray(
        [[0.0, 0.0, 0.0], [np.float32(0.05), 0.0, 0.0]],
        dtype="<f4",
    )
    negative = np.asarray(
        [[np.float32(-0.05), 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype="<f4",
    )
    positive_bytes = serialize_xyz_float32(positive)

    positive_check = builder_self_check_serialized_xyz_spacing(
        positive_bytes,
        expected_sha256=hashlib.sha256(positive_bytes).hexdigest(),
    )
    negative_check = builder_self_check_serialized_xyz_spacing(
        serialize_xyz_float32(negative)
    )
    origin = ((0, 1), (0, 1), (0, 1))
    exact_five_cm = ((1, 20), (0, 1), (0, 1))
    below_five_cm = ((49_999, 1_000_000), (0, 1), (0, 1))

    assert positive_check.passed and not positive_check.authoritative
    assert negative_check.passed and not negative_check.authoritative
    assert not support_module._exact_squared_distance_below_minimum(
        origin,
        exact_five_cm,
    )
    assert support_module._exact_squared_distance_below_minimum(
        origin,
        below_five_cm,
    )


def test_builder_self_check_enumerates_all_27_neighbor_cells() -> None:
    xyz = np.asarray(
        [[0.049, 0.049, 0.049], [0.051, 0.051, 0.051]],
        dtype="<f4",
    )

    with pytest.raises(SpacingViolationError) as error:
        builder_self_check_serialized_xyz_spacing(serialize_xyz_float32(xyz))

    assert (error.value.first_row, error.value.second_row) == (0, 1)


def test_packed_support_builder_self_check_runs_only_after_serialization() -> None:
    xyz, confidence, candidate_ids = _permutation_candidates()
    support = build_packed_support(xyz, confidence, candidate_ids)
    serialized = serialize_packed_support(support)

    result = builder_self_check_serialized_support_spacing(
        serialized,
        expected_sha256=support.digest_sha256,
    )

    assert result.input_sha256 == support.digest_sha256
    assert result.point_count == support.support_count
    assert result.passed
    assert result.authoritative is False


def test_small_support_reports_only_packed_support_capacity_no_go() -> None:
    xyz, confidence, candidate_ids = _permutation_candidates()
    support = build_packed_support(xyz, confidence, candidate_ids)
    status = support.capacity_status

    assert status.required_count == 10_000
    assert status.support_count == support.support_count
    assert not status.capacity_sufficient
    assert not status.graph_construction_allowed
    assert status.terminal_status == CAPACITY_NO_GO_STATUS
    assert status.conclusion_scope == (
        "exact_section_5_largest_color_parity_support_only"
    )
    assert status.as_dict()["terminal_status"] == CAPACITY_NO_GO_STATUS


def test_nonfinite_candidates_duplicate_ids_and_wrong_shapes_fail_closed() -> None:
    xyz = np.zeros((2, 3), dtype="<f4")
    confidence = np.ones(2, dtype="<f4")

    with pytest.raises(ValueError, match="finite"):
        build_packed_support(
            np.asarray([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]], dtype="<f4"),
            confidence,
        )
    with pytest.raises(ValueError, match="finite"):
        build_packed_support(xyz, np.asarray([1.0, np.inf], dtype="<f4"))
    with pytest.raises(ValueError, match="unique"):
        build_packed_support(
            xyz,
            confidence,
            np.asarray([3, 3], dtype="<i8"),
        )
    with pytest.raises(ValueError, match="shape"):
        build_packed_support(np.zeros((2, 4), dtype="<f4"), confidence)
