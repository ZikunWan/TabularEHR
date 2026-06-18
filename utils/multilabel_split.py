import random
from typing import Dict, Hashable, List, Mapping, Optional, Sequence


def select_balanced_multilabel_groups(
    labels_by_group: Mapping[Hashable, Sequence[Optional[int]]],
    holdout_size: int,
    seed: int,
) -> List[Hashable]:
    """Select groups while covering positive and negative examples per label."""
    group_ids = sorted(labels_by_group, key=str)
    if holdout_size <= 0:
        return []
    if holdout_size >= len(group_ids):
        raise ValueError("holdout_size must leave at least one group for training")

    num_labels = len(labels_by_group[group_ids[0]])
    features: Dict[Hashable, List[tuple[int, int]]] = {}
    totals = [[0, 0] for _ in range(num_labels)]

    for group_id in group_ids:
        values = labels_by_group[group_id]
        if len(values) != num_labels:
            raise ValueError("All multilabel vectors must have the same length")

        group_features = []
        for label_idx, value in enumerate(values):
            if value is None:
                continue
            class_idx = int(value)
            if class_idx not in (0, 1):
                raise ValueError(f"Expected binary labels, got {value!r}")
            group_features.append((label_idx, class_idx))
            totals[label_idx][class_idx] += 1
        features[group_id] = group_features

    target_per_class = max(1, holdout_size // 2)
    targets = [[0, 0] for _ in range(num_labels)]
    for label_idx, (negative_count, positive_count) in enumerate(totals):
        if negative_count > 0 and positive_count > 0:
            balanced_target = min(negative_count, positive_count, target_per_class)
            targets[label_idx] = [balanced_target, balanced_target]

    rng = random.Random(seed)
    tie_breakers = {group_id: rng.random() for group_id in group_ids}
    selected_counts = [[0, 0] for _ in range(num_labels)]
    remaining = set(group_ids)
    selected = []

    while len(selected) < holdout_size:
        weights = [[0.0, 0.0] for _ in range(num_labels)]
        has_deficit = False
        for label_idx in range(num_labels):
            for class_idx in (0, 1):
                deficit = targets[label_idx][class_idx] - selected_counts[label_idx][class_idx]
                if deficit > 0:
                    has_deficit = True
                    weights[label_idx][class_idx] = deficit / totals[label_idx][class_idx]

        if not has_deficit:
            fill = sorted(remaining, key=lambda group_id: tie_breakers[group_id], reverse=True)
            selected.extend(fill[: holdout_size - len(selected)])
            break

        best_group = max(
            remaining,
            key=lambda group_id: (
                sum(weights[label_idx][class_idx] for label_idx, class_idx in features[group_id]),
                tie_breakers[group_id],
            ),
        )
        selected.append(best_group)
        remaining.remove(best_group)
        for label_idx, class_idx in features[best_group]:
            selected_counts[label_idx][class_idx] += 1

    return selected


def select_stratified_multilabel_groups(
    labels_by_group: Mapping[Hashable, Sequence[Optional[int]]],
    holdout_size: int,
    seed: int,
) -> List[Hashable]:
    """Select groups while approximately preserving each binary label rate."""
    group_ids = sorted(labels_by_group, key=str)
    if holdout_size <= 0:
        return []
    if holdout_size >= len(group_ids):
        raise ValueError("holdout_size must leave at least one group for training")

    num_labels = len(labels_by_group[group_ids[0]])
    features: Dict[Hashable, List[tuple[int, int]]] = {}
    totals = [[0, 0] for _ in range(num_labels)]
    for group_id in group_ids:
        values = labels_by_group[group_id]
        if len(values) != num_labels:
            raise ValueError("All multilabel vectors must have the same length")
        group_features = []
        for label_idx, value in enumerate(values):
            if value is None:
                continue
            class_idx = int(value)
            if class_idx not in (0, 1):
                raise ValueError(f"Expected binary labels, got {value!r}")
            group_features.append((label_idx, class_idx))
            totals[label_idx][class_idx] += 1
        features[group_id] = group_features

    holdout_fraction = holdout_size / len(group_ids)
    targets = [[0, 0] for _ in range(num_labels)]
    for label_idx in range(num_labels):
        for class_idx in (0, 1):
            total = totals[label_idx][class_idx]
            if total >= 2:
                targets[label_idx][class_idx] = min(
                    total - 1,
                    max(1, round(total * holdout_fraction)),
                )

    rng = random.Random(seed)
    tie_breakers = {group_id: rng.random() for group_id in group_ids}
    selected_counts = [[0, 0] for _ in range(num_labels)]
    remaining = set(group_ids)
    selected = []

    while len(selected) < holdout_size:
        def score(group_id):
            contribution = 0.0
            for label_idx, class_idx in features[group_id]:
                target = targets[label_idx][class_idx]
                deficit = target - selected_counts[label_idx][class_idx]
                if deficit > 0:
                    contribution += deficit / target
            return contribution, tie_breakers[group_id]

        best_group = max(remaining, key=score)
        selected.append(best_group)
        remaining.remove(best_group)
        for label_idx, class_idx in features[best_group]:
            selected_counts[label_idx][class_idx] += 1

    return selected
