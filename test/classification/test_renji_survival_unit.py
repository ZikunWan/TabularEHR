import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from dataset.renji.renji_dataset import (
    build_piecewise_survival_target,
    load_patient_subset,
)
from utils.metrics import (
    compute_piecewise_survival_metrics,
    harrell_c_index,
)
from train.tte.train_renji_survival import split_by_patient


def test_piecewise_target_places_integer_event_in_preceding_bin():
    exposure, event_bins, stage_mask = build_piecewise_survival_target(
        time_to_event=1.0,
        event_observed=True,
        num_bins=31,
    )

    assert exposure[0] == 1.0
    assert exposure[1:].sum() == 0.0
    assert event_bins[0] == 1.0
    assert event_bins[1:].sum() == 0.0
    assert stage_mask[:31].sum() == 31
    assert stage_mask[31:].sum() == 0


def test_piecewise_target_supports_fractional_censoring():
    exposure, event_bins, _ = build_piecewise_survival_target(
        time_to_event=2.25,
        event_observed=False,
        num_bins=31,
    )

    np.testing.assert_allclose(exposure[:4], [1.0, 1.0, 0.25, 0.0])
    assert event_bins.sum() == 0.0


def test_patient_subset_normalizes_csv_filenames():
    with tempfile.TemporaryDirectory() as temp_dir:
        subset_path = Path(temp_dir) / "patients.json"
        subset_path.write_text(
            '["patient-a.csv", "/some/path/patient-b.csv"]',
            encoding="utf-8",
        )

        assert load_patient_subset(subset_path) == {"patient-a", "patient-b"}


def test_harrell_c_index_rewards_correct_ranking():
    times = np.array([1.0, 2.0, 3.0])
    events = np.array([True, True, False])
    risks = np.array([3.0, 2.0, 1.0])

    assert harrell_c_index(times, events, risks) == 1.0


def test_survival_metrics_are_finite_for_each_stage():
    horizons = (31, 150, 185)
    labels = np.zeros((12, 4, 185), dtype=np.float32)
    logits = np.full((12, 185), -4.0, dtype=np.float32)
    reference_stage = []
    reference_time = []
    reference_event = []
    reference_horizon = []

    for stage_id, horizon in enumerate(horizons):
        first_row = stage_id * 4
        for offset, (time, event) in enumerate(
            ((2, True), (4, True), (5, False), (7, False))
        ):
            row = first_row + offset
            labels[row, 0, :time] = 1.0
            labels[row, 2, :horizon] = 1.0
            labels[row, 3, 0] = horizon
            if event:
                labels[row, 1, time - 1] = 1.0
                logits[row, :horizon] = -2.0 - offset
            else:
                logits[row, :horizon] = -5.0 - offset
        reference_stage.extend([stage_id] * 4)
        reference_time.extend([2.0, 3.0, 5.0, 7.0])
        reference_event.extend([True, True, False, False])
        reference_horizon.extend([horizon] * 4)

    metrics = compute_piecewise_survival_metrics(
        SimpleNamespace(predictions=logits, label_ids=labels),
        train_reference={
            "stage_id": np.asarray(reference_stage),
            "time": np.asarray(reference_time),
            "event": np.asarray(reference_event),
            "stage_end_horizon": np.asarray(reference_horizon),
        },
        n_eval_grid=8,
    )

    for name in (
        "time_dependent_c",
        "harrell_c_index",
        "nd_calibration",
        "nam_dagostino",
        "ibs",
    ):
        assert math.isfinite(metrics[name])
    for stage_id, horizon in enumerate(horizons):
        assert metrics[f"stage_{stage_id}_nd_calibration_bins"] > 0
        assert metrics[f"stage_{stage_id}_nd_calibration_time_days"] == 3.0


def test_temporary_validation_split_is_patient_level_and_stage_balanced():
    samples = []
    for patient_index in range(20):
        patient_id = f"patient-{patient_index}"
        for stage_id in range(3):
            samples.append(
                {
                    "fname_key": patient_id,
                    "stage_id": stage_id,
                    "event_observed": (patient_index + stage_id) % 2 == 0,
                }
            )
    dataset = SimpleNamespace(samples=samples)

    train_subset, val_subset = split_by_patient(
        dataset,
        monitor_fraction=0.2,
        seed=42,
    )
    train_patients = {
        samples[index]["fname_key"] for index in train_subset.indices
    }
    val_patients = {
        samples[index]["fname_key"] for index in val_subset.indices
    }

    assert train_patients.isdisjoint(val_patients)
    assert len(val_patients) == 4
    for stage_id in range(3):
        stage_events = {
            samples[index]["event_observed"]
            for index in val_subset.indices
            if samples[index]["stage_id"] == stage_id
        }
        assert stage_events == {False, True}


def test_temporary_validation_split_preserves_rare_censoring_in_train():
    samples = [
        {
            "fname_key": f"patient-{patient_index}",
            "stage_id": 0,
            "event_observed": patient_index >= 10,
        }
        for patient_index in range(100)
    ]
    dataset = SimpleNamespace(samples=samples)

    train_subset, val_subset = split_by_patient(
        dataset,
        monitor_fraction=0.1,
        seed=42,
    )
    train_censored = sum(
        not samples[index]["event_observed"] for index in train_subset.indices
    )
    val_censored = sum(
        not samples[index]["event_observed"] for index in val_subset.indices
    )

    assert train_censored > 0
    assert val_censored > 0
