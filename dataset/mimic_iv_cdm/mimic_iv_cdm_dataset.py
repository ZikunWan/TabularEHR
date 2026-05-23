import os
import sys
import pickle
import random
import json
from collections import defaultdict

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.mimic_iv_cdm.task_info import get_task_info


class MIMICIVCDM(Dataset):
    def __init__(
        self,
        root_dir=None,
        split="train",
        lazy_mode=False,
        shuffle=True,
        table_mode="text_only",
        task_name=None,
        max_samples=None,
        return_meds=False,
        concept_map_dir=None,
    ):
        super().__init__()
        random.seed(42)

        self.root_dir = root_dir
        self.split = split
        self.lazy_mode = lazy_mode
        if table_mode not in {"text_only", "table_only", "table_plus_rest_text"}:
            raise ValueError(f"Unsupported table_mode: {table_mode}")
        self.table_mode = table_mode
        self.task_schema = get_task_info()
        self.categories = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]   
        self.task_name = task_name
        self.return_meds = return_meds
        self.concept_map_dir = concept_map_dir

        self._load_mappings()
        self._load_standard_code_mappings()
        self._load_raw_data()
        self._load_index()

        if shuffle:
            random.shuffle(self.list_data)
        if max_samples is not None:
            self.list_data = self._balance_samples(self.list_data, max_samples)

        self.data = []
        if not self.lazy_mode:
            for idx in tqdm(range(len(self.list_data)), desc=f"Preprocessing {self.split}"):
                self.data.append(self._process_item(idx))

    def _load_mappings(self):
        lab_test_mapping_path = os.path.join(self.root_dir, "lab_test_mapping.pkl")
        with open(lab_test_mapping_path, "rb") as f:
            self.lab_test_mapping = pickle.load(f)

        microbiology_test_mapping_path = os.path.join(self.root_dir, "microbiology_test_mapping.pkl")
        with open(microbiology_test_mapping_path, "rb") as f:
            self.microbiology_test_mapping = pickle.load(f)

        icd_mapping_path = os.path.join(self.root_dir, "icd_desc_to_code_mapping.pkl")
        with open(icd_mapping_path, "rb") as f:
            self.icd_code_mapping = pickle.load(f)

    def _load_raw_data(self):
        self.raw_data = {}
        for category in self.categories:
            pkl_path = os.path.join(self.root_dir, f"{category}_hadm_info_first_diag.pkl")
            with open(pkl_path, "rb") as f:
                self.raw_data[category] = pickle.load(f)

    def _normalize_id_variants(self, value):
        variants = set()
        sval = str(value).strip()
        if not sval or sval.lower() in {"nan", "none"}:
            return variants

        variants.add(sval)
        if sval.endswith(".0") and sval[:-2].isdigit():
            variants.add(sval[:-2])
        if sval.isdigit():
            variants.add(str(int(sval)))
        try:
            variants.add(str(int(value)))
        except Exception:
            pass
        return variants

    def _parse_standard_code_csv(self, csv_path):
        mapping = {}
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return mapping

        source_col = None
        for c in ("itemid (omop_source_code)", "itemid", "test_itemid"):
            if c in df.columns:
                source_col = c
                break
        if source_col is None:
            return mapping

        vocab_col = "omop_vocabulary_id" if "omop_vocabulary_id" in df.columns else None
        code_col = None
        for c in ("omop_concept_code", "concept_code", "loinc_code", "LOINC_CODE", "loinc"):
            if c in df.columns:
                code_col = c
                break
        if code_col is None:
            return mapping

        for _, row in df.iterrows():
            source_id = row.get(source_col)
            code_raw = row.get(code_col)
            if pd.isna(source_id) or pd.isna(code_raw):
                continue

            code = str(code_raw).strip()
            if not code or code.lower() in {"nan", "none"}:
                continue

            vocab = ""
            if vocab_col is not None and not pd.isna(row.get(vocab_col)):
                vocab = str(row.get(vocab_col)).strip()
            if vocab.lower() in {"nan", "none"}:
                vocab = ""

            if vocab:
                standard_code = f"{vocab.upper()}/{code}"
            else:
                # For MIMIC concept-map csvs without explicit vocab column in some variants.
                standard_code = f"LOINC/{code}"

            for key in self._normalize_id_variants(source_id):
                if key not in mapping:
                    mapping[key] = standard_code

        return mapping

    def _load_standard_code_mappings(self):
        self.lab_standard_code_mapping = {}
        self.micro_standard_code_mapping = {}

        search_dirs = []
        if self.concept_map_dir:
            search_dirs.append(self.concept_map_dir)
        search_dirs.extend(
            [
                self.root_dir,
                os.path.join(self.root_dir, "concept_map"),
                os.path.join(self.root_dir, "index_mapping"),
            ]
        )

        env_concept_dir = os.environ.get("MIMIC_CONCEPT_MAP_DIR")
        if env_concept_dir:
            search_dirs.append(env_concept_dir)

        # De-duplicate while preserving order.
        uniq_dirs = []
        seen = set()
        for d in search_dirs:
            if not d:
                continue
            dd = os.path.abspath(d)
            if dd not in seen:
                seen.add(dd)
                uniq_dirs.append(dd)

        lab_candidates = ("d_labitems_to_loinc.csv", "lab_itemid_to_loinc.csv")
        micro_candidates = (
            "microbiology_test_to_loinc.csv",
            "microbiology_to_loinc.csv",
            "microbiologyevents_to_loinc.csv",
        )

        for d in uniq_dirs:
            for fn in lab_candidates:
                fp = os.path.join(d, fn)
                if os.path.exists(fp):
                    parsed = self._parse_standard_code_csv(fp)
                    if parsed:
                        self.lab_standard_code_mapping.update(parsed)

            for fn in micro_candidates:
                fp = os.path.join(d, fn)
                if os.path.exists(fp):
                    parsed = self._parse_standard_code_csv(fp)
                    if parsed:
                        self.micro_standard_code_mapping.update(parsed)

    def _lookup_standard_code(self, item_id, table="lab"):
        mapping = self.lab_standard_code_mapping if table == "lab" else self.micro_standard_code_mapping
        for key in self._normalize_id_variants(item_id):
            if key in mapping:
                return mapping[key]
        return ""

    def _load_index(self):
        index_path = os.path.join(self.root_dir, "index", f"mimiciv_cdm_{self.split}.csv")
        self.list_data = pd.read_csv(index_path).to_dict(orient="records")

    def _balance_samples(self, all_samples, max_samples):
        if max_samples is None or max_samples >= len(all_samples):
            return all_samples

        label_groups = defaultdict(list)
        for sample in all_samples:
            label = str(sample['category']).lower()
            label_groups[label].append(sample)

        sorted_labels = sorted(label_groups.keys(), key=lambda k: len(label_groups[k]))
        balanced = []
        remaining_quota = max_samples
        remaining_classes = len(sorted_labels)

        for label in sorted_labels:
            group = label_groups[label]
            random.shuffle(group)
            fair_share = remaining_quota // max(remaining_classes, 1)
            take_count = min(len(group), fair_share)
            balanced.extend(group[:take_count])
            remaining_quota -= take_count
            remaining_classes -= 1

        random.shuffle(balanced)
        return balanced

    def _lookup_lab_name(self, item_id):
        item_id_str = str(item_id).strip()
        item_id_int = None
        if isinstance(item_id, int):
            item_id_int = item_id
        elif isinstance(item_id, float) and item_id.is_integer():
            item_id_int = int(item_id)
        elif item_id_str.isdigit():
            item_id_int = int(item_id_str)
        elif item_id_str.endswith(".0") and item_id_str[:-2].isdigit():
            item_id_int = int(item_id_str[:-2])

        if isinstance(self.lab_test_mapping, pd.DataFrame):
            row = self.lab_test_mapping.loc[self.lab_test_mapping["itemid"] == item_id, "label"]
            if len(row) > 0:
                return row.iloc[0]
            row = self.lab_test_mapping.loc[self.lab_test_mapping["itemid"] == item_id_str, "label"]
            if len(row) > 0:
                return row.iloc[0]
            if item_id_int is not None:
                row = self.lab_test_mapping.loc[self.lab_test_mapping["itemid"] == item_id_int, "label"]
                if len(row) > 0:
                    return row.iloc[0]
            return str(item_id)

        if isinstance(self.lab_test_mapping, dict):
            if item_id in self.lab_test_mapping:
                return self.lab_test_mapping[item_id]
            if item_id_str in self.lab_test_mapping:
                return self.lab_test_mapping[item_id_str]
            if item_id_int is not None and item_id_int in self.lab_test_mapping:
                return self.lab_test_mapping[item_id_int]
            return str(item_id)

        return str(item_id)

    def _lookup_micro_name(self, item_id):
        if isinstance(self.microbiology_test_mapping, dict):
            if item_id in self.microbiology_test_mapping:
                return self.microbiology_test_mapping[item_id]
            if str(item_id) in self.microbiology_test_mapping:
                return self.microbiology_test_mapping[str(item_id)]

            int_id = int(item_id)
            if int_id in self.microbiology_test_mapping:
                return self.microbiology_test_mapping[int_id]

        return str(item_id)

    def _parse_lab_value_unit(self, value):
        if isinstance(value, str):
            parts = value.strip().split()
            if len(parts) >= 2:
                return parts[0], " ".join(parts[1:])
            return value, ""
        return value, ""

    def _dict_get(self, data, key, default=""):
        if key in data:
            return data[key]
        skey = str(key)
        if skey in data:
            return data[skey]
        try:
            ikey = int(key)
            if ikey in data:
                return data[ikey]
        except Exception:
            pass
        return default

    def free_text_input_process(self, cur_item, include_radiology=True, include_demographics=True, include_table_sections=True):
        lines = []

        if include_demographics:
            lines.append("## Patient Demographics")
            lines.append(f"- Patient History: {cur_item.get('Patient History', '')}")
            lines.append(f"- Physical Examination: {cur_item.get('Physical Examination', '')}")
            lines.append("")

        if include_table_sections:
            lines.append("## Laboratory Test")
            lab_keys = ["Item_name", "Valuenum", "Valueuom", "Ref_range_lower", "Ref_range_upper"]
            lines.append(f"| {' | '.join(lab_keys)} |")
            lines.append(f"| {' | '.join(['------'] * len(lab_keys))} |")

            laboratory_tests = cur_item.get("Laboratory Tests", {})
            ref_low = cur_item.get("Reference Range Lower", {})
            ref_high = cur_item.get("Reference Range Upper", {})
            for item_id, value in laboratory_tests.items():
                name = self._lookup_lab_name(item_id)
                result, unit = self._parse_lab_value_unit(value)
                low = self._dict_get(ref_low, item_id, "")
                high = self._dict_get(ref_high, item_id, "")
                lines.append(f"| {name} | {result} | {unit} | {low} | {high} |")

            microbiology_tests = cur_item.get("Microbiology", {})
            if microbiology_tests:
                lines.append("")
                lines.append("## Microbiology Test")
                micro_keys = ["Item_name", "Valuestr"]
                lines.append(f"| {' | '.join(micro_keys)} |")
                lines.append(f"| {' | '.join(['------'] * len(micro_keys))} |")
                for item_id, value in microbiology_tests.items():
                    name = self._lookup_micro_name(item_id)
                    lines.append(f"| {name} | {value} |")
            lines.append("")

        if include_radiology:
            lines.append("## Radiology Examinations")
            rad_keys = ["Exam_name", "Text"]
            lines.append(f"| {' | '.join(rad_keys)} |")
            lines.append(f"| {' | '.join(['------'] * len(rad_keys))} |")
            radiology_tests = cur_item.get("Radiology", [])
            if not isinstance(radiology_tests, list):
                radiology_tests = []
            for item in radiology_tests:
                exam_name = item.get("Exam Name", "")
                report = str(item.get("Report", "")).strip()
                lines.append(f"| {exam_name} | {report} |")

        return "\n".join(lines)

    def structed_EHR_input_process(self, cur_item):
        rows = []
        default_time = pd.to_datetime("2000-01-01 00:00:00")

        laboratory_tests = cur_item.get("Laboratory Tests", {})
        for item_id, value in laboratory_tests.items():
            name = self._lookup_lab_name(item_id)
            result, unit = self._parse_lab_value_unit(value)
            rows.append(
                {
                    "Time": default_time,
                    "Item": str(name),
                    "Value": result,
                    "Unit": unit,
                    "Category": "measurement",
                }
            )

        microbiology_tests = cur_item.get("Microbiology", {})
        for item_id, value in microbiology_tests.items():
            name = self._lookup_micro_name(item_id)
            rows.append(
                {
                    "Time": default_time,
                    "Item": str(name),
                    "Value": value,
                    "Unit": "",
                    "Category": "measurement",
                }
            )

        measurement_df = pd.DataFrame(rows, columns=["Time", "Item", "Value", "Unit", "Category"])
        if not measurement_df.empty:
            measurement_df["Time"] = pd.to_datetime(measurement_df["Time"], errors="coerce")
            measurement_df = measurement_df.sort_values(by=["Time"]).reset_index(drop=True)
        return measurement_df

    def meds_input_process(self, cur_item, return_hf_ehr_events=True):
        default_time = "2000-01-01 00:00:00"
        rows = []

        def _build_itemid_code(prefix, item_id):
            item_id_str = str(item_id).strip()
            if not item_id_str or item_id_str.lower() in {"nan", "none"}:
                return ""
            return f"{prefix}/{item_id_str}"

        laboratory_tests = cur_item.get("Laboratory Tests", {})
        for item_id, value in laboratory_tests.items():
            code = self._lookup_standard_code(item_id, table="lab")
            if not code:
                code = _build_itemid_code("LAB", item_id)
            if not code:
                continue

            result, unit = self._parse_lab_value_unit(value)
            numeric_value = pd.to_numeric(result, errors="coerce")
            if pd.notna(numeric_value):
                numeric_out = float(numeric_value)
                text_out = ""
            else:
                numeric_out = None
                text_out = "" if pd.isna(result) else str(result).strip()

            rows.append(
                {
                    "code": code,
                    "start": default_time,
                    "end": default_time,
                    "numeric_value": numeric_out,
                    "text_value": text_out,
                    "unit": "" if pd.isna(unit) else str(unit).strip(),
                    "omop_table": "measurement",
                }
            )

        microbiology_tests = cur_item.get("Microbiology", {})
        for item_id, value in microbiology_tests.items():
            code = self._lookup_standard_code(item_id, table="microbiology")
            if not code:
                code = _build_itemid_code("MICROBIOLOGY", item_id)
            if not code:
                continue

            numeric_value = pd.to_numeric(value, errors="coerce")
            if pd.notna(numeric_value):
                numeric_out = float(numeric_value)
                text_out = ""
            else:
                numeric_out = None
                text_out = "" if pd.isna(value) else str(value).strip()

            rows.append(
                {
                    "code": code,
                    "start": default_time,
                    "end": default_time,
                    "numeric_value": numeric_out,
                    "text_value": text_out,
                    "unit": "",
                    "omop_table": "measurement",
                }
            )

        meds_df = pd.DataFrame(
            rows,
            columns=["code", "start", "end", "numeric_value", "text_value", "unit", "omop_table"],
        )

        meds_events = []
        for row in meds_df.to_dict(orient="records"):
            code = str(row.get("code", "")).strip()
            if not code:
                continue

            event = {"code": code}

            start = str(row.get("start", "")).strip()
            end = str(row.get("end", "")).strip()
            if start:
                event["start"] = start
            if end:
                event["end"] = end

            numeric_value = row.get("numeric_value")
            text_value = str(row.get("text_value", "")).strip()
            if pd.notna(numeric_value):
                event["value"] = float(numeric_value)
            elif text_value:
                event["value"] = text_value

            unit = str(row.get("unit", "")).strip()
            if unit:
                event["unit"] = unit

            omop_table = str(row.get("omop_table", "")).strip()
            if omop_table:
                event["omop_table"] = omop_table

            meds_events.append(event)

        if not return_hf_ehr_events:
            return meds_df, meds_events

        try:
            from hf_ehr.config import Event
        except Exception:
            return meds_df, meds_events, None

        hf_ehr_events = []
        for event in meds_events:
            kwargs = {"code": event["code"]}
            for key in ("value", "unit", "start", "end", "omop_table"):
                if key in event and event[key] not in (None, ""):
                    kwargs[key] = event[key]
            hf_ehr_events.append(Event(**kwargs))

        return meds_df, meds_events, hf_ehr_events

    def _build_label(self, index_item, category):
        if self.task_name != "MIMIC-IV-CDM ICD Code Diagnoses":
            return category

        icd_desc_list = index_item['icd']
        
        labels = []
        for desc in icd_desc_list:
            clean_desc = str(desc).strip()
            labels.append(self.icd_code_mapping[clean_desc])
        return labels

    def _process_item(self, index):
        index_item = self.list_data[index]
        category = index_item["category"]
        hadm_id = index_item["hadm_id"]
        cur_item = self.raw_data[category][hadm_id]

        if self.table_mode in {"table_only", "table_plus_rest_text"}:
            measurement_table = self.structed_EHR_input_process(cur_item)
            if self.table_mode == "table_only":
                context = ""
            else:
                context = self.free_text_input_process(
                    cur_item,
                    include_radiology=False,
                    include_demographics=False,
                    include_table_sections=True,
                )
        else:
            context = self.free_text_input_process(cur_item)

        label = self._build_label(index_item, category)
        task_info = self.task_schema[self.task_name]
        sample = {
            "idx": index,
            "input": context,
            "output": "\n".join([str(x) for x in label]) if isinstance(label, list) else str(label),
            "task_info": task_info,
            "instruction": task_info["instruction"],
        }
        if self.return_meds:
            meds_df, meds_events, hf_ehr_events = self.meds_input_process(
                cur_item,
                return_hf_ehr_events=True,
            )
            sample["meds_table"] = meds_df
            sample["meds_events"] = meds_events
            if hf_ehr_events is not None:
                sample["hf_ehr_events"] = hf_ehr_events

        if self.table_mode in {"table_only", "table_plus_rest_text"}:
            sample["measurement_table"] = measurement_table
            if self.table_mode == "table_plus_rest_text":
                sample["remaining_text"] = self.free_text_input_process(
                    cur_item,
                    include_radiology=True,
                    include_demographics=True,
                    include_table_sections=False,
                )
        return sample

    def __getitem__(self, index):
        if self.lazy_mode:
            sample = self._process_item(index)
        else:
            sample = self.data[index]

        if self.return_meds and "meds_events" not in sample:
            index_item = self.list_data[index]
            category = index_item["category"]
            hadm_id = index_item["hadm_id"]
            cur_item = self.raw_data[category][hadm_id]

            meds_df, meds_events, hf_ehr_events = self.meds_input_process(
                cur_item,
                return_hf_ehr_events=True,
            )
            sample["meds_table"] = meds_df
            sample["meds_events"] = meds_events
            if hf_ehr_events is not None:
                sample["hf_ehr_events"] = hf_ehr_events

        return sample

    def __len__(self):
        return len(self.list_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MIMIC-IV-CDM dataset template usage example.")
    parser.add_argument("--root_dir", type=str, default="/data/EHR_data_public/mimic-iv-cdm")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task_name", type=str, default="MIMIC-IV-CDM Main Disease Diagnoses")
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--lazy_mode", action="store_true")
    parser.add_argument(
        "--concept_map_dir",
        type=str,
        default="/data/EHR_data_public/mimic-iv-3.1-meds/pre_MEDS",
        help="Optional directory containing MIMIC concept-map CSVs (e.g., d_labitems_to_loinc.csv).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data",
            "mimic_iv_cdm",
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dataset_text = MIMICIVCDM(
        root_dir=args.root_dir,
        split=args.split,
        lazy_mode=args.lazy_mode,
        table_mode="text_only",
        task_name=args.task_name,
        max_samples=args.max_samples,
        concept_map_dir=args.concept_map_dir,
        shuffle=False,
    )
    dataset_struct = MIMICIVCDM(
        root_dir=args.root_dir,
        split=args.split,
        lazy_mode=args.lazy_mode,
        table_mode="table_only",
        task_name=args.task_name,
        max_samples=args.max_samples,
        return_meds=True,
        concept_map_dir=args.concept_map_dir,
        shuffle=False,
    )
    dataset_mixed = MIMICIVCDM(
        root_dir=args.root_dir,
        split=args.split,
        lazy_mode=args.lazy_mode,
        table_mode="table_plus_rest_text",
        task_name=args.task_name,
        max_samples=args.max_samples,
        concept_map_dir=args.concept_map_dir,
        shuffle=False,
    )

    print(f"dataset_text size: {len(dataset_text)}")
    print(f"dataset_struct size: {len(dataset_struct)}")
    print(f"dataset_mixed size: {len(dataset_mixed)}")
    if len(dataset_text) == 0:
        raise SystemExit("No samples found.")

    idx = max(0, min(args.sample_index, len(dataset_text) - 1))
    sample_text = dataset_text[idx]
    sample_struct = dataset_struct[idx]
    sample_mixed = dataset_mixed[idx]

    print(f"\nSample index: {idx}")
    print(f"text keys: {list(sample_text.keys())}")
    print(f"struct keys: {list(sample_struct.keys())}")
    print(f"mixed keys: {list(sample_mixed.keys())}")

    text_out = os.path.join(args.out_dir, "mimic_iv_cdm_text_only_sample.txt")
    with open(text_out, "w", encoding="utf-8") as f:
        f.write(str(sample_text.get("input", "")) + "\n")
    print(f"Saved text sample: {text_out}")

    struct_text_out = os.path.join(args.out_dir, "mimic_iv_cdm_table_only_text_sample.txt")
    with open(struct_text_out, "w", encoding="utf-8") as f:
        f.write(str(sample_struct.get("input", "")) + "\n")
    print(f"Saved structured-text sample: {struct_text_out}")

    mixed_text_out = os.path.join(args.out_dir, "mimic_iv_cdm_table_plus_rest_text_sample.txt")
    with open(mixed_text_out, "w", encoding="utf-8") as f:
        f.write(str(sample_mixed.get("remaining_text", "")) + "\n")
    print(f"Saved mixed-text sample: {mixed_text_out}")

    table = sample_struct.get("measurement_table")
    if isinstance(table, pd.DataFrame) and not table.empty:
        table_out = os.path.join(args.out_dir, "mimic_iv_cdm_table_only_sample.csv")
        table.to_csv(table_out, index=False, encoding="utf-8-sig")
        print(f"Saved structured table sample: {table_out} (shape={table.shape})")
    else:
        print("No measurement_table found in structured sample.")

    meds_events = sample_struct.get("meds_events", [])
    if meds_events:
        meds_json_out = os.path.join(args.out_dir, "mimic_iv_cdm_meds_sample.json")
        meds_payload = {
            "sample_index": idx,
            "task_name": args.task_name,
            "events": meds_events,
        }
        with open(meds_json_out, "w", encoding="utf-8") as f:
            json.dump(meds_payload, f, ensure_ascii=False, indent=2)
        print(f"Saved MEDS events sample: {meds_json_out} (events={len(meds_events)})")
    else:
        print("No meds_events found in structured sample.")
