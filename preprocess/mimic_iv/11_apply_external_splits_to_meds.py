from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Set

import pyarrow.parquet as pq


DEFAULT_ETL = "/opt/conda/envs/MEDS/lib/python3.11/site-packages/MIMIC_IV_MEDS/configs/ETL.yaml"
DEFAULT_EVENT_CFG = "/opt/conda/envs/MEDS/lib/python3.11/site-packages/MIMIC_IV_MEDS/configs/event_configs.yaml"
DEFAULT_STAGE_BIN = "/opt/conda/envs/MEDS/bin/MEDS_transform-stage"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply external splits to an existing MEDS output.")
    p.add_argument(
        "--external_splits_json",
        type=Path,
        default=Path("/data/zikun_workspace/code/preprocess/mimic_iv/external_splits_train_val_test.json"),
        help="Input JSON with keys train/val/test or train/tuning/held_out.",
    )
    p.add_argument(
        "--mapped_splits_json",
        type=Path,
        default=Path("/data/zikun_workspace/code/preprocess/mimic_iv/external_splits_train_tuning_heldout.json"),
        help="Output JSON with keys train/tuning/held_out.",
    )
    p.add_argument(
        "--meds_output_dir",
        type=Path,
        default=Path("/data/EHR_data_public/mimic-iv-3.1-meds/MEDS_output"),
        help="Existing MEDS_output directory.",
    )
    p.add_argument(
        "--pre_meds_dir",
        type=Path,
        default=Path("/data/EHR_data_public/mimic-iv-3.1-meds/pre_MEDS"),
        help="pre_MEDS directory used by ETL.",
    )
    p.add_argument("--etl_config", type=Path, default=Path(DEFAULT_ETL))
    p.add_argument("--event_conversion_config", type=Path, default=Path(DEFAULT_EVENT_CFG))
    p.add_argument("--stage_bin", type=Path, default=Path(DEFAULT_STAGE_BIN))
    p.add_argument("--dataset_name", type=str, default="MIMIC-IV")
    p.add_argument("--dataset_version", type=str, default="3.1:external_split_fix")
    p.add_argument("--n_workers", type=int, default=8)
    p.add_argument(
        "--downstream_mode",
        type=str,
        choices=["tmp_then_sync", "in_place"],
        default="tmp_then_sync",
        help="How to run downstream stages after split.",
    )
    p.add_argument(
        "--tmp_output_dir",
        type=Path,
        default=Path("/tmp/mimic-iv-3.1-meds-external-splits/MEDS_output"),
        help="Temporary MEDS_output path used when downstream_mode=tmp_then_sync.",
    )
    p.add_argument(
        "--cleanup_tmp_output",
        action="store_true",
        help="Delete tmp_output_dir after syncing back.",
    )
    p.add_argument(
        "--skip_downstream",
        action="store_true",
        help="Only run split_and_shard_subjects; do not rebuild downstream stages.",
    )
    return p.parse_args()


def load_external_splits(fp: Path) -> Dict[str, Set[int]]:
    if not fp.exists():
        raise FileNotFoundError(f"external_splits_json not found: {fp}")

    obj = json.loads(fp.read_text())
    keys = set(obj.keys())
    src_val_test = {"train", "val", "test"}
    src_tuning = {"train", "tuning", "held_out"}

    if src_val_test.issubset(keys):
        mapped = {
            "train": obj["train"],
            "tuning": obj["val"],
            "held_out": obj["test"],
        }
    elif src_tuning.issubset(keys):
        mapped = {
            "train": obj["train"],
            "tuning": obj["tuning"],
            "held_out": obj["held_out"],
        }
    else:
        raise ValueError(
            "external_splits_json must contain either keys "
            f"{sorted(src_val_test)} or {sorted(src_tuning)}; got {sorted(keys)}"
        )

    out: Dict[str, Set[int]] = {}
    for k in ["train", "tuning", "held_out"]:
        vals = mapped[k]
        if not isinstance(vals, list):
            raise ValueError(f"split '{k}' must be a list, got {type(vals)}")
        out[k] = {int(v) for v in vals}

    return out


def validate_disjoint(splits: Dict[str, Set[int]]) -> None:
    keys = list(splits.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            inter = splits[a] & splits[b]
            if inter:
                raise ValueError(f"Splits overlap: {a} & {b} share {len(inter)} subject_ids")


def write_mapped_splits(src: Dict[str, Set[int]], out_fp: Path) -> Dict[str, Set[int]]:
    validate_disjoint(src)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(json.dumps({k: sorted(v) for k, v in src.items()}))
    return src


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def build_env(args: argparse.Namespace, meds_output_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PRE_MEDS_DIR"] = str(args.pre_meds_dir)
    env["MEDS_OUTPUT_DIR"] = str(meds_output_dir)
    env["EVENT_CONVERSION_CONFIG_FP"] = str(args.event_conversion_config)
    env["DATASET_NAME"] = args.dataset_name
    env["DATASET_VERSION"] = args.dataset_version
    env["N_WORKERS"] = str(args.n_workers)
    return env


def run_split_stage(args: argparse.Namespace, env: dict[str, str]) -> None:
    cmd = [
        str(args.stage_bin),
        str(args.etl_config),
        "split_and_shard_subjects",
        "--multirun",
        "stage=split_and_shard_subjects",
        f"output_dir={args.meds_output_dir}",
        "do_overwrite=True",
        f"worker=range(0,{args.n_workers})",
        "hydra/launcher=joblib",
        f"stage_cfg.external_splits_json_fp={args.mapped_splits_json}",
    ]
    run_cmd(cmd, env)


def run_downstream(args: argparse.Namespace, env: dict[str, str], output_dir: Path) -> None:
    stages = [
        "convert_to_subject_sharded",
        "convert_to_MEDS_events",
        "merge_to_MEDS_cohort",
        "extract_code_metadata",
        "finalize_MEDS_metadata",
        "finalize_MEDS_data",
    ]
    for stage in stages:
        cmd = [
            str(args.stage_bin),
            str(args.etl_config),
            stage,
            "--multirun",
            f"stage={stage}",
            f"output_dir={output_dir}",
            "do_overwrite=True",
            f"worker=range(0,{args.n_workers})",
            "hydra/launcher=joblib",
        ]
        run_cmd(cmd, env)


def prepare_tmp_output_for_downstream(src_output_dir: Path, tmp_output_dir: Path) -> None:
    if tmp_output_dir.exists():
        print(f"[INFO] Removing existing temporary output dir: {tmp_output_dir}")
        shutil.rmtree(tmp_output_dir)

    tmp_output_dir.mkdir(parents=True, exist_ok=True)
    (tmp_output_dir / "metadata").mkdir(parents=True, exist_ok=True)

    src_shard_events = src_output_dir / "shard_events"
    if not src_shard_events.exists():
        raise FileNotFoundError(f"Missing shard_events directory: {src_shard_events}")

    src_shards_map = src_output_dir / "metadata" / ".shards.json"
    if not src_shards_map.exists():
        raise FileNotFoundError(f"Missing shards map: {src_shards_map}")

    dst_shard_events = tmp_output_dir / "shard_events"
    dst_shard_events.symlink_to(src_shard_events, target_is_directory=True)
    shutil.copy2(src_shards_map, tmp_output_dir / "metadata" / ".shards.json")


def sync_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source directory does not exist for sync: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    rsync_cmd = ["rsync", "-a", "--delete", f"{src}/", f"{dst}/"]
    try:
        run_cmd(rsync_cmd, os.environ.copy())
        return
    except FileNotFoundError:
        print("[WARN] rsync not found, falling back to shutil copy.")

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def sync_back_from_tmp(tmp_output_dir: Path, final_output_dir: Path) -> None:
    print(f"[INFO] Syncing data back from {tmp_output_dir} to {final_output_dir}")
    sync_dir(tmp_output_dir / "data", final_output_dir / "data")
    sync_dir(tmp_output_dir / "metadata", final_output_dir / "metadata")


def read_subject_splits(subject_splits_fp: Path) -> Dict[str, Set[int]]:
    if not subject_splits_fp.exists():
        raise FileNotFoundError(f"subject_splits.parquet not found: {subject_splits_fp}")
    tbl = pq.read_table(str(subject_splits_fp), columns=["subject_id", "split"])
    sid = tbl.column("subject_id").to_pylist()
    sp = tbl.column("split").to_pylist()

    out: Dict[str, Set[int]] = {}
    for s, p in zip(sid, sp):
        out.setdefault(str(p), set()).add(int(s))
    return out


def verify(expected: Dict[str, Set[int]], got: Dict[str, Set[int]]) -> None:
    print("\n[VERIFY] split counts in subject_splits.parquet:", {k: len(v) for k, v in got.items()})

    for k in ["train", "tuning", "held_out"]:
        if k not in got:
            raise AssertionError(f"Missing split in subject_splits.parquet: {k}")

        a, b = got[k], expected[k]
        if a != b:
            only_a = len(a - b)
            only_b = len(b - a)
            raise AssertionError(
                f"Split '{k}' mismatch: only_in_output={only_a}, only_in_expected={only_b}, "
                f"intersection={len(a & b)}"
            )

    extra = set(got.keys()) - {"train", "tuning", "held_out"}
    if extra:
        print(f"[WARN] Extra split names in subject_splits.parquet: {sorted(extra)}")

    print("[OK] subject_id mapping matches expected train/tuning/held_out exactly.")


def main() -> None:
    args = parse_args()

    if args.n_workers < 1:
        raise ValueError("n_workers must be >= 1")

    src_splits = load_external_splits(args.external_splits_json)
    validate_disjoint(src_splits)
    mapped_expected = write_mapped_splits(src_splits, args.mapped_splits_json)

    print("[INFO] mapped split sizes:", {k: len(v) for k, v in mapped_expected.items()})
    print(f"[INFO] wrote mapped splits to: {args.mapped_splits_json}")

    split_env = build_env(args, args.meds_output_dir)
    run_split_stage(args, split_env)
    if not args.skip_downstream:
        if args.downstream_mode == "in_place":
            downstream_env = build_env(args, args.meds_output_dir)
            run_downstream(args, downstream_env, args.meds_output_dir)
        else:
            tmp_output_dir = args.tmp_output_dir
            prepare_tmp_output_for_downstream(args.meds_output_dir, tmp_output_dir)
            downstream_env = build_env(args, tmp_output_dir)
            run_downstream(args, downstream_env, tmp_output_dir)
            sync_back_from_tmp(tmp_output_dir, args.meds_output_dir)
            if args.cleanup_tmp_output:
                print(f"[INFO] Cleaning up temporary output dir: {tmp_output_dir}")
                shutil.rmtree(tmp_output_dir, ignore_errors=True)

    subject_splits_fp = args.meds_output_dir / "metadata" / "subject_splits.parquet"
    got = read_subject_splits(subject_splits_fp)
    verify(mapped_expected, got)


if __name__ == "__main__":
    main()
