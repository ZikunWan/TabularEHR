import math
from types import SimpleNamespace

import torch

from pretraining.phenotype_triplet_learning import (
    PhenotypeTripletModel,
    _parse_reference_range,
    select_clinical_feature_indices,
)


def test_reference_range_parser_accepts_common_two_sided_ranges():
    assert _parse_reference_range("3.5-5.5 mmol/L") == (3.5, 5.5)
    assert _parse_reference_range("-2 to -1") == (-2.0, -1.0)
    assert _parse_reference_range("< 5") is None


def test_clinical_feature_selection_prefers_full_latest_per_measurement():
    specs = [
        SimpleNamespace(
            item="Heart Rate",
            unit="bpm",
            window_name="first24h",
            statistic="latest",
            key="hr_24h_latest",
        ),
        SimpleNamespace(
            item="Heart Rate",
            unit="bpm",
            window_name="full",
            statistic="mean",
            key="hr_full_mean",
        ),
        SimpleNamespace(
            item="Heart Rate",
            unit="bpm",
            window_name="full",
            statistic="latest",
            key="hr_full_latest",
        ),
        SimpleNamespace(
            item="Sodium",
            unit="mmol/L",
            window_name="full",
            statistic="latest",
            key="sodium_full_latest",
        ),
    ]

    assert select_clinical_feature_indices(specs).tolist() == [2, 3]


def test_clinical_distance_uses_reference_scale_and_shared_dimensions():
    distances = PhenotypeTripletModel._clinical_distance_matrix(
        local_values=torch.tensor([[2.0, 4.0]]),
        global_values=torch.tensor([[4.0, 8.0], [4.0, 999.0]]),
        local_mask=torch.tensor([[True, True]]),
        global_mask=torch.tensor([[True, True], [True, False]]),
        center=torch.tensor([0.0, 0.0]),
        scale=torch.tensor([2.0, 4.0]),
        precision_factor=torch.eye(2),
        min_shared=1,
    )

    assert torch.isclose(distances[0, 0], torch.tensor(math.sqrt(2.0)))
    assert torch.isclose(distances[0, 1], torch.tensor(math.sqrt(2.0)))


def test_semi_hard_mining_uses_hardest_clinical_neighbor_as_positive():
    positive, negative, valid, semi_hard = (
        PhenotypeTripletModel._mine_semi_hard_triplets(
            clinical_distances=torch.tensor([[0.0, 1.0, 2.0, 4.0]]),
            embedding_distances=torch.tensor([[0.0, 0.3, 0.6, 0.8]]),
            self_indices=torch.tensor([0]),
            positive_k=2,
            min_clinical_gap=0.5,
            alpha=0.2,
        )
    )

    assert positive.item() == 2
    assert negative.item() == 3
    assert valid.item()
    assert semi_hard.item()


def test_soft_triplet_margin_scales_with_clinical_distance_gap():
    terms, margins, gaps = PhenotypeTripletModel._soft_triplet_terms(
        d_emb_ap=torch.tensor([0.6, 0.2]),
        d_emb_an=torch.tensor([0.8, 0.9]),
        d_clin_ap=torch.tensor([1.0, 1.0]),
        d_clin_an=torch.tensor([3.0, 2.0]),
        alpha=0.2,
    )

    assert torch.allclose(gaps, torch.tensor([2.0, 1.0]))
    assert torch.allclose(margins, torch.tensor([0.4, 0.2]))
    assert torch.allclose(terms, torch.tensor([0.2, 0.0]))
