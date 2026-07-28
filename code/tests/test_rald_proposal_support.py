import torch

from models.rald_proposal_support import latent_only_proposal_support


class CoordinateOnlyDecoder:
    def __init__(self) -> None:
        self.seen_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def prepare_decoder_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def decode_queries(
        self,
        prepared_latent: torch.Tensor,
        normalized_queries: torch.Tensor,
    ) -> torch.Tensor:
        self.seen_shapes.append(
            (tuple(prepared_latent.shape), tuple(normalized_queries.shape))
        )
        return normalized_queries.sum(dim=-1)


def test_latent_only_support_has_frozen_cardinalities() -> None:
    cube = torch.full((1, 64, 7, 7, 5), 1e-3)
    cube[:, :, 1, 1, 1] = 2.0
    cube[:, :, 5, 5, 3] = 1.0
    decoder = CoordinateOnlyDecoder()
    latent = torch.randn(1, 4, 3)

    support = latent_only_proposal_support(
        decoder,
        latent,
        cube,
        base_seed_count=2,
        selected_coarse_count=3,
        nms_kernel=(3, 3, 3),
        decode_chunk_size=7,
    )

    assert support.proposal_flat_index.shape == (1, 2)
    assert support.coarse_coordinates_rae.shape == (1, 64, 3)
    assert support.coarse_occupancy_logit.shape == (1, 64)
    assert support.selected_coarse_coordinates_rae.shape == (1, 3, 3)
    assert support.final_coordinates_rae.shape == (1, 12, 3)
    assert decoder.seen_shapes
    assert all(latent_shape == (1, 4, 3) for latent_shape, _ in decoder.seen_shapes)


def test_proposal_support_selection_is_deterministic() -> None:
    torch.manual_seed(17)
    cube = torch.rand(1, 64, 7, 7, 5)
    latent = torch.randn(1, 4, 3)

    first = latent_only_proposal_support(
        CoordinateOnlyDecoder(),
        latent,
        cube,
        base_seed_count=3,
        selected_coarse_count=4,
        nms_kernel=(3, 3, 3),
        decode_chunk_size=11,
    )
    second = latent_only_proposal_support(
        CoordinateOnlyDecoder(),
        latent,
        cube,
        base_seed_count=3,
        selected_coarse_count=4,
        nms_kernel=(3, 3, 3),
        decode_chunk_size=13,
    )

    assert torch.equal(first.proposal_flat_index, second.proposal_flat_index)
    assert torch.equal(first.selected_coarse_index, second.selected_coarse_index)
    assert torch.equal(first.final_coordinates_rae, second.final_coordinates_rae)
