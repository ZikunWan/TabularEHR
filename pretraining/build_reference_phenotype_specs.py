import csv
import json
import os
import re
from dataclasses import dataclass, field
from typing import List

from transformers import HfArgumentParser


@dataclass
class ReferencePhenotypeSpecArguments:
    reference_path: str = field(
        default="data/phenotype_triplet_reference_scales.csv"
    )
    output_path: str = field(
        default="/data/zikun_workspace/.cache/phenotype_triplet_learning/phenotype_query_specs.json"
    )
    statistic: str = field(default="latest")
    window_name: str = field(default="full")


def sanitize_key(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return text or "phenotype"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def alias_candidates(item: str) -> List[str]:
    item = clean_text(item)
    aliases = {item}
    no_parentheses = re.sub(r"\s*\([^)]*\)", "", item).strip()
    if no_parentheses:
        aliases.add(no_parentheses)
    aliases.add(item.replace(",", ""))

    lower = item.lower()
    manual_aliases = {
        "alanine aminotransferase": ["ALT", "Alanine Aminotransferase (ALT)"],
        "aspartate aminotransferase": [
            "AST",
            "Asparate Aminotransferase (AST)",
            "Aspartate Aminotransferase (AST)",
        ],
        "bilirubin, total": ["Bilirubin Total", "Total Bilirubin"],
        "bilirubin, direct": ["Bilirubin Direct", "Direct Bilirubin"],
        "platelet count": ["Platelets"],
        "white blood cells": ["WBC", "Leukocytes"],
        "red blood cells": ["RBC", "Erythrocytes"],
        "urea nitrogen": ["BUN", "Blood Urea Nitrogen"],
        "carbon dioxide": ["Carbon Dioxide, Total", "Total CO2", "Bicarbonate"],
        "mcv": ["MCV"],
        "mch": ["MCH"],
        "mchc": ["MCHC"],
        "rdw": ["RDW", "Erythrocyte Distribution Width"],
    }
    for needle, values in manual_aliases.items():
        if needle in lower:
            aliases.update(values)
    return sorted(alias for alias in aliases if alias)


def build_query_text(item: str, unit: str, low: str, high: str, statistic: str, window: str) -> str:
    return (
        f"Continuous clinical measurement: {item}. "
        f"Unit: {unit}. "
        f"Normal range: {low}-{high}. "
        f"Target: {statistic} value during {window}."
    )


def main():
    parser = HfArgumentParser(ReferencePhenotypeSpecArguments)
    (args,) = parser.parse_args_into_dataclasses()

    specs = []
    seen = set()
    with open(args.reference_path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            item = clean_text(row["item"])
            unit = clean_text(row.get("unit", ""))
            low = clean_text(row["ref_low"])
            high = clean_text(row["ref_high"])
            key = sanitize_key(f"{item}_{unit}_{args.window_name}_{args.statistic}")
            dedupe_key = (item.lower(), unit.lower(), args.window_name, args.statistic)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            specs.append(
                {
                    "key": key,
                    "item": item,
                    "query_text": build_query_text(
                        item=item,
                        unit=unit,
                        low=low,
                        high=high,
                        statistic=args.statistic,
                        window=args.window_name,
                    ),
                    "aliases": alias_candidates(item),
                    "statistic": args.statistic,
                    "unit": unit,
                    "description": "",
                    "normal_range": f"{low}-{high}",
                    "window_name": args.window_name,
                    "window_start_hours": None,
                    "window_end_hours": None,
                    "category_regex": "^measurement$",
                    "item_regex": None,
                    "transform": "none",
                    "mean": None,
                    "scale": None,
                }
            )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(specs, file, indent=2, ensure_ascii=True)
        file.write("\n")
    print(f"Wrote {len(specs)} reference phenotype specs to {args.output_path}")


if __name__ == "__main__":
    main()
