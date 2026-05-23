import os 
import sys
import json
import time
from torch.utils.data import Dataset
from functools import *
import pandas as pd
import random
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore", message=".*DataFrameGroupBy.apply operated on the grouping columns.*")
from tqdm import tqdm
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.ehrshot.task_info import get_task_info

ADDITIONAL_INFO = {
    "Body weight": {
        "unit": "oz",
        "ref_low": "350",
        "ref_high": "1000"
    },
    "Body height": {
        "unit": "inch",
        "ref_low": "5",
        "ref_high": "100"
    },
    "Body mass index / BMI": {
        "unit": "kg/m2",
        "ref_low": "18.5",
        "ref_high": "24.9"
    },
    "Body surface area": {
        "unit": "m2",
        "ref_low": "0.1",
        "ref_high": "10"
    },
    "Heart rate": {
        "unit": "bpm",
        "ref_low": "60",
        "ref_high": "100"
    },
    "Systolic blood pressure": {
        "unit": "mmHg",
        "ref_low": "90",
        "ref_high": "140"
    },
    "Diastolic blood pressure": {
        "unit": "mmHg",
        "ref_low": "60",
        "ref_high": "90"
    },
    "Body temperature": {
        "unit": "F",
        "ref_low": "95",
        "ref_high": "100.4"
    },
    "Respiratory rate": {
        "unit": "breaths/min",
        "ref_low": "12",
        "ref_high": "18"
    },
    "Oxygen saturation": {
        "unit": "%",
        "ref_low": "95",
        "ref_high": "100"
    },
    "Hemoglobin": {
        "unit": "g/dL",
        "ref_low": "12",
        "ref_high": "17"
    },
    "Hematocrit": {
        "unit": "%",
        "ref_low": "36",
        "ref_high": "51"
    },
    "Erythrocytes": {
        "unit": "10^6/uL",
        "ref_low": "4.2",
        "ref_high": "5.9"
    },
    "Leukocytes": {
        "unit": "10^6/uL",
        "ref_low": "4",
        "ref_high": "10"
    },
    "Platelets": {
        "unit": "10^3/uL",
        "ref_low": "150",
        "ref_high": "350"
    },
    "Sodium": {
        "unit": "mmol/L",
        "ref_low": "136",
        "ref_high": "145"
    },
    "Potassium": {
        "unit": "mmol/L",
        "ref_low": "3.5",
        "ref_high": "5.0"
    },
    "Chloride": {
        "unit": "mmol/L",
        "ref_low": "98",
        "ref_high": "106"
    },
    "Carbon dioxide, total": {
        "unit": "mmol/L",
        "ref_low": "23",
        "ref_high": "28"
    },
    "Calcium": {
        "unit": "mmol/L",
        "ref_low": "9",
        "ref_high": "10.5"
    },
    "Glucose": {
        "unit": "mmol/L",
        "ref_low": "70",
        "ref_high": "100"
    },
    "Urea nitrogen": {
        "unit": "mg/dL",
        "ref_low": "8",
        "ref_high": "20"
    },
    "Creatinine": {
        "unit": "mg/dL",
        "ref_low": "0.7",
        "ref_high": "1.3"
    },
    "Anion gap": {
        "unit": "mmol/L",
        "ref_low": "3",
        "ref_high": "11",
    }
}

AGGREGATED_MAPPING = {
    "LOINC/29463-7": "Body weight",
    "LOINC/8302-2": "Body height",
    " LOINC/39156-5": "Body mass index / BMI",
    "LOINC/8277-6": "Body surface area",
    "SNOMED/301898006": "Body surface area",

    "LOINC/8867-4": "Heart rate",
    "SNOMED/364075005": "Heart rate",
    "SNOMED/78564009": "Heart rate",
    "LOINC/8480-6": "Systolic blood pressure",
    "SNOMED/271649006": "Systolic blood pressure",
    "LOINC/8462-4": "Diastolic blood pressure",
    "SNOMED/271650006": "Diastolic blood pressure",
    "LOINC/8310-5": "Body temperature",
    "LOINC/9279-1": "Respiratory rate",
    "LOINC/LP21258-6": "Oxygen saturation",

    "LOINC/718-7": "Hemoglobin",
    "SNOMED/271026005": "Hemoglobin",
    "SNOMED/441689006": "Hemoglobin",
    "LOINC/4544-3": "Hematocrit",
    "LOINC/20570-8": "Hematocrit",
    "LOINC/48703-3": "Hematocrit",
    "SNOMED/28317006": "Hematocrit",

    "LOINC/789-8": "Erythrocytes",
    "LOINC/26453-1": "Erythrocytes",
    "LOINC/20584-9": "Leukocytes",
    "LOINC/6690-2": "Leukocytes",
    "LOINC/777-3": "Platelets",
    "SNOMED/61928009": "Platelets",
    "LOINC/2951-2": "Sodium",
    "LOINC/2947-0": "Sodium",
    "SNOMED/25197003": "Sodium",
    "LOINC/2823-3": "Potassium",
    "SNOMED/312468003": "Potassium",
    "LOINC/6298-4": "Potassium",
    "LOINC/2075-0": "Chloride",
    "SNOMED/104589004": "Chloride",
    "LOINC/2028-9": "Carbon dioxide, total",
    "LOINC/17861-6": "Calcium",
    "SNOMED/271240001": "Calcium",
    "LOINC/2345-7": "Glucose",
    "SNOMED/166900001": "Glucose",
    "LOINC/2339-0": "Glucose",
    "SNOMED/33747003": "Glucose",
    "LOINC/14749-6": "Glucose",
    "LOINC/3094-0": "Urea nitrogen",
    "SNOMED/105011006": "Urea nitrogen",
    "LOINC/2160-0": "Creatinine",
    "SNOMED/113075003": "Creatinine",
    "LOINC/33037-3": "Anion gap",
    "LOINC/41276-7": "Anion gap",
    "SNOMED/25469001": "Anion gap"
}


_TABLE_LENGTH_WORKER_DATASET = None


def _ehrshot_table_length_cache_key_from_sample(sample_info):
    return (
        f"{sample_info.get('patient_id', '')}|"
        f"{sample_info.get('period_begin', '')}|"
        f"{sample_info.get('period_end', '')}|"
        f"{sample_info.get('prediction_time', '')}"
    )


def _init_ehrshot_table_length_worker(root_dir):
    global _TABLE_LENGTH_WORKER_DATASET
    worker_dataset = EHRSHOTDataset.__new__(EHRSHOTDataset)
    worker_dataset.root_dir = root_dir
    worker_dataset.ehr_dir = os.path.join(root_dir, "patient_ehr")
    code_2_description_path = os.path.join(root_dir, "utils/code_2_description.json")
    with open(code_2_description_path, 'r', encoding='utf-8') as f:
        worker_dataset.code_2_description = json.load(f)
    worker_dataset.code_2_description.update(AGGREGATED_MAPPING)
    _TABLE_LENGTH_WORKER_DATASET = worker_dataset


def _compute_ehrshot_table_length_worker(payload):
    idx, sample_info = payload
    dataset = _TABLE_LENGTH_WORKER_DATASET
    table_length = int(len(dataset.structed_EHR_input_process(sample_info)))
    cache_key = _ehrshot_table_length_cache_key_from_sample(sample_info)
    return idx, cache_key, table_length

class EHRSHOTDataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        sample_info_path=None,
        sample_info=None,
        lazy_mode=True,
        table_mode="text_only",
        max_samples=None,
        task_name=None,
        return_meds=False,
        use_table_length_cache=False,
    ):  
        random.seed(42)
        
        self.task_schema = get_task_info()
        self.root_dir = root_dir
        if table_mode not in {"text_only", "table_only", "table_plus_rest_text"}:
            raise ValueError(f"Unsupported table_mode: {table_mode}")
        self.table_mode = table_mode
        self.return_meds = return_meds
        self.ehr_dir = os.path.join(root_dir, "patient_ehr")
        self.sample_info_path = sample_info_path
        self.lazy_mode = lazy_mode # load data on the fly when set to `True`, otherwise load all data to memory (require lots of memories).
        self.use_table_length_cache = use_table_length_cache
        self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        self.table_length_cache_dir = os.path.join(self.root_dir, "cache", "table_length")
        os.makedirs(self.table_length_cache_dir, exist_ok=True)
        sample_info_name = os.path.basename(self.sample_info_path) if self.sample_info_path else "sample_info_memory.csv"
        self.table_length_cache_file = os.path.join(self.table_length_cache_dir, f"{sample_info_name}.table_length.json")

        if sample_info is None:
            df = pd.read_csv(self.sample_info_path, low_memory=False)
            self.sample_info = df.to_dict(orient='records')
        else:
            self.sample_info = list(sample_info)
            
        # Filter by specific task if provided
        if task_name is not None:
            task_name = [task_name]
            self.sample_info = [sample for sample in self.sample_info if sample.get("task_name") in task_name]

        if self.use_table_length_cache and self.sample_info:
            self._ensure_table_lengths_cached()
        
        # 1. Sort all available samples by table length (estimated by period_end - period_begin)
        self.sample_info = sorted(self.sample_info, key=lambda x: x['period_end'] - x['period_begin'])

        # 2. Build a balanced subset while prioritizing shortest samples
        if max_samples is not None:
            self.sample_info = self._balance_samples(self.sample_info, max_samples)

        self.get_code_mapping()

        self.data = []
        if not self.lazy_mode:
            for idx in tqdm(range(len(self.sample_info)), desc="Pre-processing"):
                self.data.append(self._process_item(idx))

    def _table_length_cache_key(self, sample_info):
        return _ehrshot_table_length_cache_key_from_sample(sample_info)

    def _load_table_length_cache(self):
        if not os.path.exists(self.table_length_cache_file):
            return {}
        with open(self.table_length_cache_file, "r", encoding="utf-8") as f:
            return {str(k): int(v) for k, v in json.load(f).items()}

    def _save_table_length_cache(self, cache_data):
        with open(self.table_length_cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

    def _ensure_table_lengths_cached(self):
        cache_data = self._load_table_length_cache()
        missing_indices = []

        for idx, sample_info in enumerate(self.sample_info):
            cache_key = self._table_length_cache_key(sample_info)
            if cache_key in cache_data:
                sample_info["table_length"] = int(cache_data[cache_key])
            else:
                missing_indices.append(idx)

        if not missing_indices:
            return

        if self.local_rank not in (-1, 0):
            wait_seconds = int(os.environ.get("EHRSHOT_TABLE_LENGTH_WAIT_SECONDS", "7200"))
            deadline = time.time() + max(1, wait_seconds)
            while time.time() < deadline:
                cache_data = self._load_table_length_cache()
                unresolved = 0
                for idx in missing_indices:
                    cache_key = self._table_length_cache_key(self.sample_info[idx])
                    if cache_key in cache_data:
                        self.sample_info[idx]["table_length"] = int(cache_data[cache_key])
                    else:
                        unresolved += 1
                if unresolved == 0:
                    return
                time.sleep(2)
            return

        tasks = [(idx, self.sample_info[idx]) for idx in missing_indices]
        requested_workers = int(os.environ.get("EHRSHOT_TABLE_LENGTH_WORKERS", str(os.cpu_count() or 1)))
        num_workers = max(1, min(requested_workers, len(tasks)))

        if num_workers == 1:
            _init_ehrshot_table_length_worker(self.root_dir)
            iterator = (_compute_ehrshot_table_length_worker(task) for task in tasks)
            for idx, cache_key, table_length in tqdm(iterator, total=len(tasks), desc="Computing EHRSHOT table_length"):
                self.sample_info[idx]["table_length"] = table_length
                cache_data[cache_key] = table_length
        else:
            chunk_size = max(1, min(int(os.environ.get("EHRSHOT_TABLE_LENGTH_CHUNK_SIZE", "64")), len(tasks)))
            with mp.get_context("fork").Pool(
                processes=num_workers,
                initializer=_init_ehrshot_table_length_worker,
                initargs=(self.root_dir,),
            ) as pool:
                for idx, cache_key, table_length in tqdm(
                    pool.imap_unordered(_compute_ehrshot_table_length_worker, tasks, chunksize=chunk_size),
                    total=len(tasks),
                    desc=f"Computing EHRSHOT table_length ({num_workers} workers)",
                ):
                    self.sample_info[idx]["table_length"] = table_length
                    cache_data[cache_key] = table_length

        self._save_table_length_cache(cache_data)
    
    def _balance_samples(self, all_samples, max_samples):
        if max_samples is None or max_samples >= len(all_samples):
            return all_samples

        label_groups = defaultdict(list)
        for s in all_samples:
            lbl = str(s['label'])
            label_groups[lbl].append(s)

        # all_samples has already been sorted by length proxy.
        # Therefore, each group preserves shortest-first order.
        labels = sorted(label_groups.keys(), key=lambda k: len(label_groups[k]))
        capacities = {lbl: len(label_groups[lbl]) for lbl in labels}
        quotas = {lbl: 0 for lbl in labels}
        remaining = max_samples
        active = [lbl for lbl in labels if capacities[lbl] > 0]

        # Water-filling allocation: as balanced as possible under class capacities.
        while remaining > 0 and active:
            fair_share = max(1, remaining // len(active))
            progressed = False
            next_active = []
            for lbl in active:
                can_take = capacities[lbl] - quotas[lbl]
                if can_take <= 0:
                    continue
                take_count = min(can_take, fair_share, remaining)
                if take_count > 0:
                    quotas[lbl] += take_count
                    remaining -= take_count
                    progressed = True
                if quotas[lbl] < capacities[lbl]:
                    next_active.append(lbl)
                if remaining == 0:
                    break
            active = next_active
            if not progressed:
                break

        # If quota remains (due to integer division), fill one-by-one.
        if remaining > 0:
            for lbl in labels:
                if remaining == 0:
                    break
                can_take = capacities[lbl] - quotas[lbl]
                if can_take <= 0:
                    continue
                take_count = min(can_take, remaining)
                quotas[lbl] += take_count
                remaining -= take_count

        balanced_samples = []
        for lbl in labels:
            take_count = quotas[lbl]
            if take_count > 0:
                balanced_samples.extend(label_groups[lbl][:take_count])

        # Keep shortest samples first in the final subset.
        balanced_samples = sorted(
            balanced_samples,
            key=lambda x: x['period_end'] - x['period_begin']
        )
        return balanced_samples[:max_samples]

    def get_code_mapping(self):
        # Load unified code-to-description mapping from JSON
        code_2_description_path = os.path.join(self.root_dir, 'utils/code_2_description.json')
        with open(code_2_description_path, 'r', encoding='utf-8') as f:
            self.code_2_description = json.load(f)
        self.code_2_description.update(AGGREGATED_MAPPING)

    def _load_context_groups(self, sample):
        patient_id = sample['patient_id']
        patient_path = os.path.join(self.ehr_dir, str(patient_id) + '.csv')
        patient_info = pd.read_csv(patient_path, low_memory=False)
        patient_info['start'] = pd.to_datetime(patient_info['start'], format='mixed')
        patient_info['end'] = pd.to_datetime(patient_info['end'], format='mixed')
        person_info = patient_info[patient_info["omop_table"] == "person"]
        context_slice = patient_info.iloc[sample["period_begin"]:sample["period_end"]+1]
        if sample.get("visit_start") is not None and sample.get("visit_end") is not None:
            visit_start = pd.to_datetime(sample["visit_start"], errors="coerce")
            visit_end = pd.to_datetime(sample["visit_end"], errors="coerce")
            if not pd.isna(visit_start) and not pd.isna(visit_end):
                context_slice = context_slice[
                    (context_slice["start"] >= visit_start)
                    & (context_slice["start"] <= visit_end)
                ]
        context_info = pd.concat([person_info, context_slice])
        groups = []
        for name, group in context_info.groupby('omop_table'):
            group_records = group.to_dict(orient='records')
            for record in group_records:
                record['omop_table'] = name
            if group_records:
                groups.append(group_records)
        return sorted(groups, key=lambda x: x[0]['start'])
    
    def gather_visit_info(self, group_data, event_str_list):
        if len(group_data) == 1:
            item = group_data[0]
            code = item['code']
            if code is None:
                return event_str_list
            
            content = self.code_2_description.get(code, None)
            if content is None:
                return event_str_list

            event_str_list.append(
                f"- {content}"
            )
        else:
            for item in group_data:
                code = item['code']
                if code is None:
                    continue
                
                content = self.code_2_description.get(code, None)
                if content is None:
                    continue

                if f"- {content}" not in event_str_list:
                    event_str_list.append(
                        f"- {content}"
                    )

        return event_str_list
    
    def gather_meansurement_info(self, group_data, event_str_list):
        info_keys = ['Item_Name', 'Valuenum', 'Valueuom', "Ref_Range_Lower", "Ref_Range_Upper", "Flag"]
        event_str_list.append(
            f"""| {" | ".join([key.title() for key in info_keys])} |"""
        )
        event_str_list.append(f"""| {" | ".join(["------"] * len(info_keys))} |""")
        event_list = []
        for item in group_data:
            row = []

            code = item.get('code', None)
            item_name = AGGREGATED_MAPPING.get(code, None)
            if item_name is None:
                item_name = self.code_2_description.get(code, None)
                if item_name is None:
                    continue
                else:
                    value = item.get('value', None)
                    if not isinstance(value, str):
                        continue
                    
                    unit = item.get('unit', None)
                    row = [item_name, str(value), str(unit), "nan", "nan", "nan"]
                    event_str_list.append(
                        f"""| {" | ".join(row)} |"""
                    )
                    event_list.append(item_name)
                    continue

            value = item.get('value', None)
            if not isinstance(value, str):
                continue

            item_info = ADDITIONAL_INFO[item_name]
            try:
                item_flag = "normal" if float(value) >= float(item_info["ref_low"]) and float(value) <= float(item_info["ref_high"]) else "abnormal"
            except:
                continue

            row = [item_name, value, item_info["unit"], item_info["ref_low"], item_info["ref_high"], item_flag]
            event_str_list.append(
                f"""| {" | ".join(row)} |"""
            )
            event_list.append(item_name)
        
        latest_event_str_list = event_str_list[:3]
        for event_id, (event, event_str) in enumerate(zip(event_list, event_str_list[3:])):
            if event not in event_list[event_id+1:]:
                latest_event_str_list.append(event_str)

        event_str_list = latest_event_str_list
        return event_str_list
    
    def free_text_input_process(self, sample):
        context_info = self._load_context_groups(sample)
        '''
        Example: 

            context_info = [
                [  # group 1: person
                    {
                        "omop_table": "person",
                        "code": "Gender/F",
                        "description": "FEMALE",
                        "start": pd.Timestamp("1970-01-01 00:00:00"),
                        "end": pd.NaT,
                        "value": None,
                        "unit": None,
                    },
                    {
                        "omop_table": "person",
                        "code": "Race/5",
                        "description": "White",
                        "start": pd.Timestamp("1970-01-01 00:00:00"),
                        "end": pd.NaT,
                        "value": None,
                        "unit": None,
                    },
                ],
                [  # group 2: measurement
                    {
                        "omop_table": "measurement",
                        "code": "LOINC/8867-4",
                        "description": "Heart rate",
                        "start": pd.Timestamp("2020-06-01 08:00:00"),
                        "end": pd.NaT,
                        "value": "88.0",
                        "unit": "bpm",
                    },
                    {
                        "omop_table": "measurement",
                        "code": "LOINC/8480-6",
                        "description": "Systolic blood pressure",
                        "start": pd.Timestamp("2020-06-01 08:00:00"),
                        "end": pd.NaT,
                        "value": "120.0",
                        "unit": "mmHg",
                    },
                ],
                [  # group 3: note
                    {
                        "omop_table": "note",
                        "code": "LOINC/34117-2",
                        "description": "History and physical note",
                        "start": pd.Timestamp("2020-06-01 09:00:00"),
                        "end": pd.NaT,
                        "value": "Patient reports mild chest pain...",
                        "unit": None,
                    }
                ],
            ]
        '''
        item_str_mapping_list = []

        for group_data in context_info:
            event_str_list = []
            event_name = group_data[0]['omop_table']

            if event_name.lower() == 'note':
                continue

            event_time = group_data[0]['start'].strftime('%Y-%m-%d %H:%M:%S')
    
            title = f"## {event_name.title()} [{event_time}]"
            event_str_list.append(title)

            if event_name == "person":
                for item in group_data:
                    description = self.code_2_description[item['code']]
                    event_str_list.append(f"- {description}")

            elif "drug" in event_name or "condition" in event_name or "procedure" in event_name:
                event_str_list = self.gather_visit_info(group_data, event_str_list)
            
            else:
                event_str_list = self.gather_meansurement_info(group_data, event_str_list)
                if len(event_str_list) < 4:
                    continue
                
            item_str_mapping_list.append("\n".join(event_str_list))

        text = "\n\n".join(item_str_mapping_list)
        return text 

    def remaining_text_input_process(self, sample):
        context_info = self._load_context_groups(sample)
        remaining_blocks = []

        for group_data in context_info:
            event_name = group_data[0]['omop_table']

            if event_name.lower() == 'note':
                continue

            has_structured_row = False
            for item in group_data:
                code = item.get('code')
                item_name = self.code_2_description.get(code, None)
                if item_name is not None:
                    has_structured_row = True
                    break

            if has_structured_row:
                continue

            event_str_list = []
            event_time = group_data[0]['start'].strftime('%Y-%m-%d %H:%M:%S')
            title = f"## {event_name.title()} [{event_time}]"
            event_str_list.append(title)

            if event_name == "person":
                for item in group_data:
                    description = self.code_2_description.get(item['code'])
                    if description:
                        event_str_list.append(f"- {description}")
            elif "drug" in event_name or "condition" in event_name or "procedure" in event_name:
                event_str_list = self.gather_visit_info(group_data, event_str_list)
            else:
                event_str_list = self.gather_meansurement_info(group_data, event_str_list)
                if len(event_str_list) < 4:
                    continue

            if len(event_str_list) > 1:
                remaining_blocks.append("\n".join(event_str_list))

        return "\n\n".join(remaining_blocks)

    def structed_EHR_input_process(self, test_item):
        context_info = self._load_context_groups(test_item)
        
        rows = []

        for group_data in context_info:
            event_name = group_data[0]['omop_table']

            if event_name.lower() == 'note':
                continue

            for item in group_data:
                code = item['code']
                item_name = self.code_2_description.get(code, None)
                if item_name is None:
                    # Skip items whose code is not covered by code_2_description.
                    continue

                value = item['value']
                unit = item['unit']
                start_time = item['start']

                if pd.isna(value):
                    value = ""
                if pd.isna(unit):
                    unit = ""

                # Normalize person rows to fixed Item names.
                if str(event_name).lower() == "person":
                    code_text = str(code)
                    raw_description = item["description"]
                    if code_text == "SNOMED/3950001":
                        table_item = "Birth"
                        value = ""
                    elif code_text.startswith("Race/"):
                        table_item = "Race"
                        value = raw_description
                    elif code_text.startswith("Gender/"):
                        table_item = "Gender"
                        if code_text.upper().startswith("GENDER/F"):
                            value = "Female"
                        elif code_text.upper().startswith("GENDER/M"):
                            value = "Male"
                    elif code_text.startswith("Ethnicity/"):
                        table_item = "Ethnicity"
                        value = str(raw_description)

                else:
                    table_item = str(item_name)

                rows.append({
                    "Time": start_time,
                    "Item": table_item,
                    "Value": value,
                    "Unit": "" if unit is None else str(unit),
                    "Category": event_name if event_name else "unknown",
                })

        measurement_df = pd.DataFrame(rows, columns=["Time", "Item", "Value", "Unit", "Category"])
        if not measurement_df.empty:
            measurement_df["Time"] = pd.to_datetime(measurement_df["Time"], errors="coerce")
            measurement_df = measurement_df.sort_values(by=["Time"]).reset_index(drop=True)

        return measurement_df

    def MEDS_input_process(self, test_item, return_hf_ehr_events=True):
        context_info = self._load_context_groups(test_item)
        rows = []

        for group_data in context_info:
            event_name = group_data[0]["omop_table"]
            if str(event_name).lower() == "note":
                continue

            for item in group_data:
                code = item.get("code")
                if code is None or pd.isna(code):
                    continue

                value = item.get("value")
                unit = item.get("unit")
                start_time = item.get("start")

                rows.append(
                    {
                        "Time": start_time,
                        "Item": str(code),
                        "Value": "" if pd.isna(value) else str(value),
                        "Unit": "" if pd.isna(unit) else str(unit),
                        "Category": event_name if event_name else "unknown",
                    }
                )

        meds_df = pd.DataFrame(rows, columns=["Time", "Item", "Value", "Unit", "Category"])
        if not meds_df.empty:
            meds_df["Time"] = pd.to_datetime(meds_df["Time"], errors="coerce")
            meds_df = meds_df.sort_values(by=["Time"]).reset_index(drop=True)

        meds_events = []
        for _, row in meds_df.iterrows():
            code = str(row.get("Item", "")).strip()
            if not code:
                continue

            event = {"code": code}

            time_value = pd.to_datetime(row.get("Time"), errors="coerce")
            if not pd.isna(time_value):
                time_str = time_value.strftime("%Y-%m-%d %H:%M:%S")
                event["start"] = time_str
                event["end"] = time_str

            value = str(row.get("Value", "")).strip()
            if value:
                try:
                    event["value"] = float(value)
                except Exception:
                    event["value"] = value

            unit = str(row.get("Unit", "")).strip()
            if unit:
                event["unit"] = unit

            category = str(row.get("Category", "")).strip()
            if category:
                event["omop_table"] = category

            meds_events.append(event)

        if not return_hf_ehr_events:
            return meds_df, meds_events

        from hf_ehr.config import Event
        

        hf_ehr_events = []
        for event in meds_events:
            kwargs = {"code": event["code"]}
            for key in ("value", "unit", "start", "end", "omop_table"):
                if key in event and event[key] not in (None, ""):
                    kwargs[key] = event[key]
            hf_ehr_events.append(Event(**kwargs))


        return meds_df, meds_events, hf_ehr_events
            
    def __len__(self):
        return len(self.sample_info)
        
    def _process_item(self, index):
        sample = self.sample_info[index]
        if sample.get("task") == "pretraining_context":
            context = "" if self.table_mode == "table_only" else self.free_text_input_process(sample)
            output_sample = {
                "idx": index,
                "input": context,
                "output": "",
                "task_info": {"task": "pretraining_context"},
                "instruction": "",
            }
            if self.table_mode in {"table_only", "table_plus_rest_text"}:
                measurement_df = self.structed_EHR_input_process(sample)
                output_sample["measurement_table"] = measurement_df
                output_sample["table_length"] = len(measurement_df)
                if self.table_mode == "table_plus_rest_text":
                    output_sample["remaining_text"] = self.remaining_text_input_process(sample)
            return output_sample

        task_name = sample['task_name']
        context = "" if self.table_mode == "table_only" else self.free_text_input_process(sample)

        if self.table_mode in {"table_only", "table_plus_rest_text"}:
            measurement_df = self.structed_EHR_input_process(sample)

        task_info = {}
        task_info["task"] = task_name
        task_info["task_type"] = self.task_schema[task_name]["task_type"]
        task_info["metric"] = self.task_schema[task_name]["metric"]
        raw_label = sample['label']
        
        if "lab_" in task_name:
            task_info["label"] = int(raw_label)
        else: 
            str_label = str(raw_label).strip().lower()
            if str_label == 'true':
                task_info["label"] = "1"
            else:
                task_info["label"] = "0"
        
        output_sample = {
            "idx": index,
            "input": context,
            "output": str(task_info["label"]),
            "task_info": task_info,
            "instruction": self.task_schema[task_name]['instruction'],
        }

        if self.return_meds:
            meds_df, meds_events, hf_ehr_events = self.MEDS_input_process(sample, return_hf_ehr_events=True)
            output_sample["meds_table"] = meds_df
            output_sample["meds_events"] = meds_events
            if hf_ehr_events is not None:
                output_sample["hf_ehr_events"] = hf_ehr_events
        
        if self.table_mode in {"table_only", "table_plus_rest_text"}:
            output_sample["measurement_table"] = measurement_df
            output_sample["table_length"] = len(measurement_df)
            if self.table_mode == "table_plus_rest_text":
                output_sample["remaining_text"] = self.remaining_text_input_process(sample)
        return output_sample

    def __getitem__(self, index):
        # Eager mode: return cached preprocessed sample.
        # Lazy mode: process current sample on demand.
        if not self.lazy_mode:
            sample = self.data[index]
        else:
            sample = self._process_item(index)

        if self.return_meds and "meds_events" not in sample:
            meds_df, meds_events, hf_ehr_events = self.MEDS_input_process(self.sample_info[index], return_hf_ehr_events=True)
            sample["meds_table"] = meds_df
            sample["meds_events"] = meds_events
            if hf_ehr_events is not None:
                sample["hf_ehr_events"] = hf_ehr_events

        return sample


__all__ = ["EHRSHOTDataset", "ADDITIONAL_INFO", "AGGREGATED_MAPPING"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print one EHRSHOT sample as MEDS-style Event list.")
    parser.add_argument("--root_dir", type=str, default="/data/EHR_data_public/EHRSHOT")
    parser.add_argument("--sample_info_path", type=str, default="/data/EHR_data_public/EHRSHOT/index/ehrshot_test.csv")
    parser.add_argument("--task_name", type=str, default="guo_los")
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--lazy_mode", action="store_true")
    parser.add_argument("--max_events", type=int, default=32)
    args = parser.parse_args()

    dataset = EHRSHOTDataset(
        root_dir=args.root_dir,
        sample_info_path=args.sample_info_path,
        lazy_mode=args.lazy_mode,
        table_mode="text_only",
        max_samples=args.max_samples,
        task_name=args.task_name,
        return_meds=True,
    )

    if len(dataset) == 0:
        raise SystemExit("No samples found.")

    idx = max(0, min(args.sample_index, len(dataset) - 1))
    sample = dataset[idx]

    def event_repr_from_dict(event):
        return (
            "Event("
            f"code={event.get('code')!r}, "
            f"value={event.get('value', None)!r}, "
            f"unit={event.get('unit', None)!r}, "
            f"start={event.get('start', None)!r}, "
            f"end={event.get('end', None)!r}, "
            f"omop_table={event.get('omop_table', None)!r}"
            ")"
        )

    hf_events = sample.get("hf_ehr_events")
    if hf_events is not None:
        all_events = [repr(ev) for ev in hf_events]
    else:
        all_events = [event_repr_from_dict(ev) for ev in sample.get("meds_events", [])]

    shown_events = all_events[: max(1, args.max_events)]

    print("patient: List[Event] = [")
    for line in shown_events:
        print(f"    {line},")
    if len(all_events) > len(shown_events):
        print(f"    # ... {len(all_events) - len(shown_events)} more events")
    print("]")
