import json
import os
import sys
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import HfArgumentParser

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.eicu.eicu_dataset import EICUDataset
from dataset.mimic.mimic_dataset import MIMICIV
from dataset.mimic_iv_cdm.mimic_iv_cdm_dataset import MIMICIVCDM


EXPECTED_MEDS_COLUMNS = [
    "subject_id",
    "time",
    "code",
    "numeric_value",
    "text_value",
    "unit",
    "omop_table",
]


def rank0_print(*args, **kwargs):
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in (-1, 0):
        print(*args, **kwargs)


@dataclass
class ExportArguments:
    dataset_name: str = field(
        default="ehr_bench",
        metadata={"help": "One of: ehr_bench, eicu, ehrshot, mimic_iv_cdm."},
    )
    split: str = field(
        default="train",
        metadata={"help": "One of: train, val, validation, test."},
    )
    output_dir: str = field(
        default="/data/zikun_workspace/code/.cache/ethos_exports/ehr_bench/train",
        metadata={"help": "Directory to save exported raw MEDS parquet shards."},
    )
    overwrite_output: bool = field(
        default=False,
        metadata={"help": "Overwrite output_dir if it already exists."},
    )
    lazy_mode: bool = field(
        default=True,
        metadata={"help": "Whether to lazily read source datasets."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap for exported samples across all loaded tasks."},
    )
    task_names: Optional[str] = field(
        default=None,
        metadata={"help": "Optional comma-separated task names to export."},
    )
    samples_per_shard: int = field(
        default=2000,
        metadata={"help": "Maximum samples written into one parquet shard."},
    )
    empty_context_token: str = field(
        default="NO_EVENT_CONTEXT",
        metadata={"help": "Fallback token when a sample has no MEDS rows."},
    )


@dataclass
class DataArguments:
    # EHR-Bench / MIMIC
    ehr_bench_data_dir: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular")
    ehr_bench_train_sample_info_path: Optional[str] = field(default=None)
    ehr_bench_val_sample_info_path: Optional[str] = field(default=None)
    ehr_bench_test_sample_info_path: Optional[str] = field(default=None)
    ehr_bench_itemid_representation: str = field(default="code")
    ehr_bench_concept_map_dir: Optional[str] = field(default=None)

    # eICU
    eicu_root_dir: str = field(default="/data/EHR_data_public/eicu-crd/2.0")
    eicu_processed_dir: str = field(default="/data/zikun_workspace/eicu-crd/processed")
    eicu_train_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/sample_info_train.json")
    eicu_val_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/sample_info_val.json")
    eicu_test_info_path: Optional[str] = field(default=None)

    # EHRShot
    ehrshot_root_dir: str = field(default="/data/EHR_data_public/EHRSHOT")
    ehrshot_train_info_path: str = field(default="/data/EHR_data_public/EHRSHOT/index/ehrshot_train.csv")
    ehrshot_val_info_path: str = field(default="/data/EHR_data_public/EHRSHOT/index/ehrshot_val.csv")
    ehrshot_test_info_path: Optional[str] = field(default=None)

    # MIMIC-IV-CDM
    mimic_iv_cdm_root_dir: str = field(default="/data/EHR_data_public/mimic-iv-cdm")
    mimic_iv_cdm_concept_map_dir: Optional[str] = field(default=None)


def _normalize_split_name(split: str) -> str:
    split = str(split).strip().lower()
    if split == "validation":
        return "val"
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    return split


def _parse_task_names(task_names: Optional[str]) -> Optional[List[str]]:
    if task_names is None:
        return None
    parsed = [item.strip() for item in str(task_names).split(",") if item and item.strip()]
    return parsed or None


def _ensure_output_dir(output_dir: str, overwrite: bool):
    path = Path(output_dir)
    if path.exists():
        existing_files = list(path.glob("*"))
        if existing_files and not overwrite:
            raise FileExistsError(
                f"Output directory already contains files: {path}. "
                "Pass --overwrite_output True to overwrite."
            )
    path.mkdir(parents=True, exist_ok=True)


def _resolve_ehr_bench_paths(split: str, data_args: DataArguments, task_names: Optional[List[str]]) -> List[str]:
    if split == "train":
        if data_args.ehr_bench_train_sample_info_path:
            return [data_args.ehr_bench_train_sample_info_path]
        pattern = os.path.join(data_args.ehr_bench_data_dir, "task_index", "train", "*.csv")
    elif split == "val":
        if data_args.ehr_bench_val_sample_info_path:
            return [data_args.ehr_bench_val_sample_info_path]
        pattern = os.path.join(data_args.ehr_bench_data_dir, "task_index", "val", "*.csv")
    else:
        if data_args.ehr_bench_test_sample_info_path:
            return [data_args.ehr_bench_test_sample_info_path]
        pattern = os.path.join(data_args.ehr_bench_data_dir, "task_index", "test", "*.csv")

    paths = sorted(glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No EHR-Bench task index files found: {pattern}")

    if task_names is None:
        return paths

    keep = set(task_names)
    filtered = [path for path in paths if os.path.splitext(os.path.basename(path))[0] in keep]
    missing = sorted(keep.difference({os.path.splitext(os.path.basename(path))[0] for path in filtered}))
    if missing:
        raise FileNotFoundError(f"EHR-Bench task index not found for tasks: {missing}")
    return filtered


def _load_datasets(
    export_args: ExportArguments,
    data_args: DataArguments,
) -> List[Tuple[str, object]]:
    dataset_name = export_args.dataset_name.strip().lower()
    split = _normalize_split_name(export_args.split)
    task_names = _parse_task_names(export_args.task_names)
    datasets: List[Tuple[str, object]] = []

    if dataset_name == "ehr_bench":
        sample_info_paths = _resolve_ehr_bench_paths(split, data_args, task_names)
        for sample_info_path in sample_info_paths:
            task_name = os.path.splitext(os.path.basename(sample_info_path))[0]
            dataset = MIMICIV(
                root_dir=data_args.ehr_bench_data_dir,
                sample_info_path=sample_info_path,
                lazy_mode=export_args.lazy_mode,
                shuffle=False,
                table_mode="table_only",
                max_samples=export_args.max_samples,
                itemid_representation=data_args.ehr_bench_itemid_representation,
                concept_map_dir=data_args.ehr_bench_concept_map_dir,
                return_meds=True,
            )
            datasets.append((task_name, dataset))
        return datasets

    if dataset_name == "eicu":
        if split == "train":
            sample_info_path = data_args.eicu_train_info_path
        elif split == "val":
            sample_info_path = data_args.eicu_val_info_path
        else:
            sample_info_path = data_args.eicu_test_info_path
        if not sample_info_path:
            raise FileNotFoundError(f"Missing eICU sample info path for split={split}")

        if task_names is None:
            dataset = EICUDataset(
                root_dir=data_args.eicu_root_dir,
                processed_dir=data_args.eicu_processed_dir,
                sample_info_path=sample_info_path,
                task_name=None,
                lazy_mode=export_args.lazy_mode,
                shuffle=False,
                table_mode="table_only",
                max_samples=export_args.max_samples,
                return_meds=True,
            )
            datasets.append(("all", dataset))
        else:
            for task_name in task_names:
                dataset = EICUDataset(
                    root_dir=data_args.eicu_root_dir,
                    processed_dir=data_args.eicu_processed_dir,
                    sample_info_path=sample_info_path,
                    task_name=task_name,
                    lazy_mode=export_args.lazy_mode,
                    shuffle=False,
                    table_mode="table_only",
                    max_samples=export_args.max_samples,
                    return_meds=True,
                )
                datasets.append((task_name, dataset))
        return datasets

    if dataset_name == "ehrshot":
        if split == "train":
            sample_info_path = data_args.ehrshot_train_info_path
        elif split == "val":
            sample_info_path = data_args.ehrshot_val_info_path
        else:
            sample_info_path = data_args.ehrshot_test_info_path
        if not sample_info_path:
            raise FileNotFoundError(f"Missing EHRShot sample info path for split={split}")

        if task_names is None:
            dataset = EHRSHOTDataset(
                root_dir=data_args.ehrshot_root_dir,
                sample_info_path=sample_info_path,
                lazy_mode=export_args.lazy_mode,
                table_mode="table_only",
                max_samples=export_args.max_samples,
                task_name=None,
                return_meds=True,
            )
            datasets.append(("all", dataset))
        else:
            for task_name in task_names:
                dataset = EHRSHOTDataset(
                    root_dir=data_args.ehrshot_root_dir,
                    sample_info_path=sample_info_path,
                    lazy_mode=export_args.lazy_mode,
                    table_mode="table_only",
                    max_samples=export_args.max_samples,
                    task_name=task_name,
                    return_meds=True,
                )
                datasets.append((task_name, dataset))
        return datasets

    if dataset_name == "mimic_iv_cdm":
        if task_names is None:
            dataset = MIMICIVCDM(
                root_dir=data_args.mimic_iv_cdm_root_dir,
                split=split,
                lazy_mode=export_args.lazy_mode,
                shuffle=False,
                table_mode="table_only",
                task_name=None,
                max_samples=export_args.max_samples,
                return_meds=True,
                concept_map_dir=data_args.mimic_iv_cdm_concept_map_dir,
            )
            datasets.append(("all", dataset))
        else:
            for task_name in task_names:
                dataset = MIMICIVCDM(
                    root_dir=data_args.mimic_iv_cdm_root_dir,
                    split=split,
                    lazy_mode=export_args.lazy_mode,
                    shuffle=False,
                    table_mode="table_only",
                    task_name=task_name,
                    max_samples=export_args.max_samples,
                    return_meds=True,
                    concept_map_dir=data_args.mimic_iv_cdm_concept_map_dir,
                )
                datasets.append((task_name, dataset))
        return datasets

    raise ValueError(f"Unsupported dataset_name: {export_args.dataset_name}")


def _as_dataframe(sample: dict, empty_context_token: str) -> pd.DataFrame:
    meds_df = sample.get("meds_table")
    if isinstance(meds_df, pd.DataFrame):
        df = meds_df.copy()
    elif meds_df is None:
        df = pd.DataFrame(columns=EXPECTED_MEDS_COLUMNS)
    else:
        df = pd.DataFrame(meds_df)

    for col in EXPECTED_MEDS_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[EXPECTED_MEDS_COLUMNS].copy()

    if len(df) == 0:
        df = pd.DataFrame(
            [
                {
                    "subject_id": None,
                    "time": "1970-01-01 00:00:00",
                    "code": empty_context_token,
                    "numeric_value": None,
                    "text_value": "",
                    "unit": "",
                    "omop_table": "observation",
                }
            ]
        )
    return df


def _build_source_subject_id(sample: dict, df: pd.DataFrame) -> str:
    for key in ("subject_id", "patient_id", "patienthealthsystemstayid", "icustay_id", "hadm_id"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    if "subject_id" in df.columns:
        non_null = df["subject_id"].dropna()
        if len(non_null) > 0:
            return str(non_null.iloc[0]).strip()
    return ""


def _iter_export_rows(
    export_args: ExportArguments,
    datasets: Iterable[Tuple[str, object]],
):
    exported = 0
    for task_name, dataset in datasets:
        rank0_print(f"Exporting task={task_name} with {len(dataset)} samples")
        iterator = tqdm(
            range(len(dataset)),
            desc=f"Export [{task_name}]",
            disable=int(os.environ.get("LOCAL_RANK", "-1")) not in (-1, 0),
        )
        for idx in iterator:
            if export_args.max_samples is not None and exported >= int(export_args.max_samples):
                return
            sample = dataset[idx]
            exported += 1
            yield task_name, idx, sample


def _write_shard(output_dir: Path, shard_idx: int, shard_frames: List[pd.DataFrame]):
    if not shard_frames:
        return
    shard_path = output_dir / f"{shard_idx:05d}.parquet"
    shard_df = pd.concat(shard_frames, axis=0, ignore_index=True)
    shard_df.to_parquet(shard_path, index=False)


def main():
    parser = HfArgumentParser((ExportArguments, DataArguments))
    export_args, data_args = parser.parse_args_into_dataclasses()

    split = _normalize_split_name(export_args.split)
    export_args.split = split
    _ensure_output_dir(export_args.output_dir, overwrite=export_args.overwrite_output)

    datasets = _load_datasets(export_args, data_args)
    if not datasets:
        raise ValueError("No datasets were loaded for export.")

    output_dir = Path(export_args.output_dir)
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists() and export_args.overwrite_output:
        manifest_path.unlink()

    rank0_print("=" * 88)
    rank0_print("Export Benchmark To ETHOS Raw MEDS")
    rank0_print("=" * 88)
    rank0_print(f"Dataset: {export_args.dataset_name}")
    rank0_print(f"Split: {export_args.split}")
    rank0_print(f"Output dir: {output_dir}")
    rank0_print(f"Task filter: {export_args.task_names or '(all)'}")
    rank0_print(f"Samples/shard: {export_args.samples_per_shard}")

    shard_frames: List[pd.DataFrame] = []
    shard_sample_count = 0
    shard_idx = 0
    sample_timeline_id = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        for task_name, idx, sample in _iter_export_rows(export_args, datasets):
            sample_timeline_id += 1
            df = _as_dataframe(sample, empty_context_token=export_args.empty_context_token)
            df["subject_id"] = int(sample_timeline_id)
            df["time"] = df["time"].fillna("").astype(str)
            df["code"] = df["code"].fillna("").astype(str)
            df["text_value"] = df["text_value"].fillna("").astype(str)
            df["unit"] = df["unit"].fillna("").astype(str)
            df["omop_table"] = df["omop_table"].fillna("").astype(str)

            source_subject_id = _build_source_subject_id(sample, df)
            label_text = str(sample.get("output", ""))

            shard_frames.append(df)
            shard_sample_count += 1

            manifest = {
                "subject_id": int(sample_timeline_id),
                "dataset_name": str(export_args.dataset_name),
                "split": str(export_args.split),
                "task_name": str(task_name),
                "sample_index": int(idx),
                "source_subject_id": source_subject_id,
                "label": label_text,
                "num_rows": int(len(df)),
            }
            manifest_f.write(json.dumps(manifest, ensure_ascii=False) + "\n")

            if shard_sample_count >= int(export_args.samples_per_shard):
                _write_shard(output_dir, shard_idx, shard_frames)
                shard_idx += 1
                shard_frames = []
                shard_sample_count = 0

    if shard_frames:
        _write_shard(output_dir, shard_idx, shard_frames)
        shard_idx += 1

    rank0_print(f"Finished export. Timeline samples: {sample_timeline_id}")
    rank0_print(f"Parquet shards written: {shard_idx}")
    rank0_print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
