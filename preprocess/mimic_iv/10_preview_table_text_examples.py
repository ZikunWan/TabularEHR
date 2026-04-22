import argparse
import os
import random
from typing import Dict, List

import pandas as pd

# Add project root for local imports.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.mimic.mimic_dataset import MIMICIV
from dataset.mimic.input_format import MIMICIVStringConvertor
from dataset.mimic.task_info import get_task_info


DEFAULT_ROOT = "/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular"
DEFAULT_SAMPLE_CSVS = [
    "/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/task_index/train/contrastive_learning.csv",
    "/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/task_index/val/contrastive_learning.csv",
]
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "contrastive_preview_examples.md")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview table/text pairs from contrastive sample_info CSV(s)."
    )
    parser.add_argument("--root_dir", type=str, default=DEFAULT_ROOT)
    parser.add_argument(
        "--sample_csv",
        action="append",
        default=None,
        help="Path to a sample_info csv. Can be repeated.",
    )
    parser.add_argument("--samples_per_csv", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_table_rows", type=int, default=20)
    parser.add_argument("--max_text_chars", type=int, default=2000)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_lightweight_dataset(root_dir: str):
    dataset = MIMICIV.__new__(MIMICIV)
    dataset.convertor = MIMICIVStringConvertor(
        origin_data_dir=os.path.join(root_dir, "index_mapping"),
        cache_dir=os.path.join(root_dir, "cache"),
    )
    dataset.task_schema = get_task_info()
    dataset.ehr_dir = os.path.join(root_dir, "patients_ehr")
    dataset.similar_item_dir = os.path.join(root_dir, "cache", "similar_item")
    dataset.similar_item = {}
    dataset.only_structed_ehr = True
    dataset.sample_info = []
    return dataset


def normalize_record(record: Dict) -> Dict:
    out = dict(record)
    int_fields = ["context_begin", "context_end", "admissions_id", "last_discharge_id"]
    for key in int_fields:
        if key in out and pd.notna(out[key]) and str(out[key]).strip() != "":
            out[key] = int(float(out[key]))
    return out


def preview_table_text(dataset, sample_info: Dict, max_rows: int, max_chars: int) -> Dict:
    dataset.sample_info = [sample_info]
    sample = dataset._process_item(0)

    table = sample.get("measurement_table")
    if isinstance(table, pd.DataFrame) and len(table) > 0:
        cols = [c for c in ["Time", "Item", "Value", "Unit", "Category"] if c in table.columns]
        if not cols:
            cols = list(table.columns)
        table_text = table[cols].head(max_rows).to_string(index=False)
    else:
        table_text = "(empty table)"

    input_text = str(sample.get("input", ""))
    text_preview = input_text[:max_chars]
    if len(input_text) > max_chars:
        text_preview += "\n... [truncated]"

    return {
        "table_text": table_text,
        "text_preview": text_preview,
        "table_rows": int(len(table)) if isinstance(table, pd.DataFrame) else 0,
        "text_chars": len(input_text),
    }


def main():
    args = parse_args()
    csv_paths = args.sample_csv if args.sample_csv else DEFAULT_SAMPLE_CSVS

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"sample_info csv not found: {csv_path}")

    dataset = build_lightweight_dataset(args.root_dir)
    rng = random.Random(args.seed)

    lines: List[str] = []
    lines.append("# Contrastive Table/Text Preview")
    lines.append("")
    lines.append(f"- root_dir: `{args.root_dir}`")
    lines.append(f"- samples_per_csv: {args.samples_per_csv}")
    lines.append("")

    ex_id = 1
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, low_memory=False)
        if "task" in df.columns:
            df = df[df["task"].astype(str) == "contrastive_learning"]

        if len(df) == 0:
            lines.append(f"## Source: `{csv_path}`")
            lines.append("")
            lines.append("No contrastive_learning rows found.")
            lines.append("")
            continue

        take = min(args.samples_per_csv, len(df))
        picked = sorted(rng.sample(range(len(df)), take))

        for idx in picked:
            raw = df.iloc[idx].to_dict()
            sample_info = normalize_record(raw)

            lines.append(f"## Example {ex_id}")
            lines.append(f"- source_csv: `{csv_path}`")
            lines.append(f"- row_index: {idx}")
            lines.append(
                f"- sample_key: `{sample_info.get('subject_id','')}|{sample_info.get('task','')}|"
                f"{sample_info.get('context_begin','')}|{sample_info.get('context_end','')}`"
            )

            try:
                preview = preview_table_text(
                    dataset=dataset,
                    sample_info=sample_info,
                    max_rows=args.max_table_rows,
                    max_chars=args.max_text_chars,
                )
                lines.append(f"- table_rows: {preview['table_rows']}")
                lines.append(f"- text_chars: {preview['text_chars']}")
                lines.append("")
                lines.append("### Table")
                lines.append("```text")
                lines.append(preview["table_text"])
                lines.append("```")
                lines.append("")
                lines.append("### Text")
                lines.append("```text")
                lines.append(preview["text_preview"])
                lines.append("```")
                lines.append("")
            except Exception as e:
                lines.append(f"- ERROR: {type(e).__name__}: {e}")
                lines.append("")

            ex_id += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved preview examples to: {args.output_path}")


if __name__ == "__main__":
    main()
