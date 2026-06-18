import torch
from torch.utils.data import Dataset

from pretraining.pretrain import (
    CycledMultiTaskDataset,
    MultiTaskCollator,
    WeightedLossCombiner,
)


class _RangeDataset(Dataset):
    def __init__(self, length, prefix):
        self.length = length
        self.prefix = prefix

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return f"{self.prefix}{index}"


def test_cycled_multitask_dataset_uses_longest_length_and_cycles_shorter_tasks():
    dataset = CycledMultiTaskDataset(
        ntp_dataset=_RangeDataset(5, "n"),
        task_dataset=_RangeDataset(2, "t"),
        ranking_dataset=_RangeDataset(3, "r"),
    )

    assert len(dataset) == 5
    assert dataset[4] == {"ntp": "n4", "task": "t0", "ranking": "r1"}


def test_multitask_collator_dispatches_each_task_batch():
    collator = MultiTaskCollator(
        ntp_collator=lambda values: ("ntp", values),
        task_collator=lambda values: ("task", values),
        ranking_collator=lambda values: ("ranking", values),
    )

    output = collator(
        [
            {"ntp": 1, "task": 2, "ranking": 3},
            {"ntp": 4, "task": 5, "ranking": 6},
        ]
    )

    assert output["ntp"] == ("ntp", [1, 4])
    assert output["task"] == ("task", [2, 5])
    assert output["ranking"] == ("ranking", [3, 6])


def test_weighted_loss_combiner_averages_equal_weights():
    combiner = WeightedLossCombiner(weights=[1.0, 1.0, 1.0])
    total, weighted_losses = combiner(
        {
            "ntp": torch.tensor(3.0),
            "task": torch.tensor(6.0),
            "ranking": torch.tensor(9.0),
        }
    )

    assert torch.allclose(weighted_losses, torch.tensor([3.0, 6.0, 9.0]))
    assert torch.isclose(total, torch.tensor(6.0))


def test_weighted_loss_combiner_applies_explicit_task_weights():
    combiner = WeightedLossCombiner(weights=[2.0, 1.0, 1.0])
    total, weighted_losses = combiner(
        {
            "ntp": torch.tensor(4.0),
            "task": torch.tensor(2.0),
            "ranking": torch.tensor(6.0),
        }
    )

    assert torch.allclose(weighted_losses, torch.tensor([8.0, 2.0, 6.0]))
    assert torch.isclose(total, torch.tensor(4.0))


def test_weighted_loss_combiner_supports_disabling_a_task():
    combiner = WeightedLossCombiner(weights=[1.0, 1.0, 0.0])
    total, weighted_losses = combiner(
        {
            "ntp": torch.tensor(4.0),
            "task": torch.tensor(2.0),
            "ranking": torch.tensor(100.0),
        }
    )

    assert torch.allclose(weighted_losses, torch.tensor([4.0, 2.0, 0.0]))
    assert torch.isclose(total, torch.tensor(3.0))
