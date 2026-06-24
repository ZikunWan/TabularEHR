import json
import os

import pandas as pd
import tqdm


EVENT_NAME = {
    "patients": "Patient Demographics",
    "omr": "Online Medical Record",
    "pharmacy": "Pharmacy",
    "poe": "Provider Order Entry",
    "procedures_icd": "Procedures on International Classification of Diseases",
    "prescriptions": "Prescriptions",
    "services": "Services",
    "labevents": "Laboratory Test Events",
    "microbiologyevents": "Microbiology Test Events",
    "admissions": "Admissions",
    "transfers": "Transfers",
    "emar": "Electronic Medicine Administration Record",
    "hcpcsevents": "Hcpcs Events",
    "diagnoses_icd": "Diagnoses on International Classification of Diseases",
    "drgcodes": "Drgcodes",
    "radiology": "Radiology Examinations",
    "discharge": "Discharge",
    "edstays": "EDstays",
    "triage": "Triage",
    "medrecon": "Medrecon",
    "pyxis": "Pyxis",
    "vitalsign": "Vitalsign",
    "diagnosis": "ED Diagnoses on International Classification of Diseases",
    "icustays": "ICUstays",
    "chartevents": "Chart Events",
    "ingredientevents": "Ingredient Events",
    "datetimeevents": "Datetime Events",
    "procedureevents": "Procedure Events",
    "inputevents": "Input Events",
    "outputevents": "Output Events",
}


def safe_read(json_element):
    if isinstance(json_element, (float, int)):
        json_element = str(json_element)

    if isinstance(json_element, list):
        return json_element

    if isinstance(json_element, str) and json_element in {"NaN", "nan", "None"}:
        return ""

    if pd.isna(json_element):
        json_element = ""

    if json_element is None:
        return ""

    return json_element


def process_icd(index_csv):
    inflect_index = {"9": {}, "10": {}}
    with open(index_csv) as f:
        icd_index = pd.read_csv(f)
    for index in tqdm.tqdm(range(len(icd_index))):
        sample = icd_index.iloc[index]
        inflect_index[str(sample["icd_version"])][str(sample["icd_code"])] = sample["long_title"]
    return inflect_index


def process_prescriptions_atc(index_csv):
    df = pd.read_csv(index_csv)
    return df.set_index("ndc_code")["atc_name"].to_dict()


def process_hcpcs_item(index_csv):
    inflect_index = {}
    with open(index_csv) as f:
        index_pd = pd.read_csv(f)
    for index in tqdm.tqdm(range(len(index_pd))):
        sample = index_pd.iloc[index]
        inflect_index[str(sample["code"])] = sample["short_description"]
    return inflect_index


def process_lab_item(index_csv):
    inflect_index = {}
    with open(index_csv) as f:
        index_pd = pd.read_csv(f)
    for index in tqdm.tqdm(range(len(index_pd))):
        sample = index_pd.iloc[index]
        inflect_index[str(sample["itemid"])] = sample["label"]
    return inflect_index


def process_icu_item(index_csv):
    inflect_index = {}
    with open(index_csv) as f:
        index_pd = pd.read_csv(f)
    for index in tqdm.tqdm(range(len(index_pd))):
        sample = index_pd.iloc[index]
        inflect_index[str(sample["itemid"])] = sample["label"]
    return inflect_index


def process_concept_map_item_code(index_csv):
    inflect_index = {}
    with open(index_csv) as f:
        index_pd = pd.read_csv(f)

    source_col = "itemid (omop_source_code)"
    vocab_col = "omop_vocabulary_id"
    code_col = "omop_concept_code"
    if source_col not in index_pd.columns:
        return inflect_index

    for index in tqdm.tqdm(range(len(index_pd))):
        sample = index_pd.iloc[index]
        src = str(sample.get(source_col, "")).strip()
        if not src or src.lower() in {"nan", "none"}:
            continue

        vocab = str(sample.get(vocab_col, "")).strip()
        code = str(sample.get(code_col, "")).strip()
        if vocab.lower() in {"nan", "none"}:
            vocab = ""
        if code.lower() in {"nan", "none"}:
            code = ""

        if vocab and code:
            inflect_index[src] = f"{vocab}/{code}"
        elif code:
            inflect_index[src] = code

    return inflect_index


def process_radiology_item(index_csv):
    inflect_index = {}
    with open(index_csv) as f:
        index_pd = pd.read_csv(f)

    index_pd = index_pd[index_pd["field_name"] == "exam_name"]
    for index in tqdm.tqdm(range(len(index_pd))):
        sample = index_pd.iloc[index]
        note_id = sample["note_id"]
        inflect_index.setdefault(note_id, []).append(sample["field_value"])

    for note_id in inflect_index:
        inflect_index[note_id] = list(set(inflect_index[note_id]))

    return inflect_index


class MIMICIVStringConvertor:
    def __init__(
        self,
        origin_data_dir,
        cache_dir,
        itemid_representation="description",
        concept_map_dir=None,
    ):
        self.origin_data_dir = origin_data_dir
        self.cache_dir = cache_dir
        self.itemid_representation = itemid_representation
        self.concept_map_dir = concept_map_dir
        if self.itemid_representation not in {"description", "code"}:
            raise ValueError(
                f"Unsupported itemid_representation={self.itemid_representation}. "
                "Use one of: description, code."
            )
        self._concept_map_index_cache = {}
        self.event_info = self._build_event_info()
        self.get_item_index_for_event()

    @staticmethod
    def _event(event_type, item_mapping=None):
        return {"event_type": event_type, "item_mapping": item_mapping}

    def _build_event_info(self):
        ndc_mapping = [
            {
                "input_name": "ndc",
                "item_index_file": "d_atc_prescriptions",
                "item_index_preprocess_func": process_prescriptions_atc,
                "output_name": "ATC Type",
            }
        ]
        icd_diagnosis_mapping = [
            {
                "input_name": "icd_code",
                "item_index_preprocess_func": process_icd,
                "item_index_file": "d_icd_diagnoses",
                "output_name": "diagnoses",
            },
            {
                "input_name": "icd_code",
                "item_index_preprocess_func": process_icd,
                "item_index_file": "d_ccs_diagnoses",
                "output_name": "CCS Type",
            },
        ]
        itemid_mapping = lambda item_index_file: [
            {
                "input_name": "itemid",
                "item_index_preprocess_func": process_icu_item,
                "item_index_file": item_index_file,
                "output_name": "item_name",
            }
        ]

        return {
            "patients": self._event("hosp"),
            "omr": self._event("hosp"),
            "pharmacy": self._event("hosp"),
            "prescriptions": self._event("hosp", ndc_mapping),
            "poe": self._event("hosp"),
            "procedures_icd": self._event(
                "hosp",
                [
                    {
                        "input_name": "icd_code",
                        "item_index_preprocess_func": process_icd,
                        "item_index_file": "d_icd_procedures",
                        "output_name": "procedures",
                    },
                    {
                        "input_name": "icd_code",
                        "item_index_preprocess_func": process_icd,
                        "item_index_file": "d_ccs_procedures",
                        "output_name": "CCS Type",
                    },
                ],
            ),
            "services": self._event("hosp"),
            "labevents": self._event(
                "hosp",
                [
                    {
                        "input_name": "itemid",
                        "item_index_file": "d_labitems",
                        "item_index_preprocess_func": process_lab_item,
                        "output_name": "item_name",
                    }
                ],
            ),
            "microbiologyevents": self._event("hosp"),
            "admissions": self._event("hosp"),
            "transfers": self._event("hosp"),
            "emar": self._event("hosp"),
            "diagnoses_icd": self._event("hosp", icd_diagnosis_mapping),
            "radiology": self._event(
                "note",
                [
                    {
                        "input_name": "note_id",
                        "item_index_preprocess_func": process_radiology_item,
                        "item_index_file": "radiology_detail",
                        "output_name": "exam_name",
                    }
                ],
            ),
            "discharge": self._event("note"),
            "edstays": self._event("ed"),
            "triage": self._event("ed"),
            "medrecon": self._event("ed", ndc_mapping),
            "pyxis": self._event("ed"),
            "vitalsign": self._event("ed"),
            "diagnosis": self._event(
                "ed",
                [
                    {
                        "input_name": "icd_code",
                        "item_index_preprocess_func": process_icd,
                        "item_index_file": "d_ccs_diagnoses",
                        "output_name": "CCS Type",
                    }
                ],
            ),
            "icustays": self._event("icu"),
            "chartevents": self._event("icu", itemid_mapping("d_items")),
            "ingredientevents": self._event("icu", itemid_mapping("d_items")),
            "datetimeevents": self._event("icu", itemid_mapping("d_items")),
            "procedureevents": self._event("icu", itemid_mapping("d_items")),
            "inputevents": self._event("icu", itemid_mapping("d_items")),
            "outputevents": self._event("icu", itemid_mapping("d_items")),
        }

    def _concept_map_files_for_event(self, event_name):
        files = {
            "labevents": ["d_labitems_to_loinc"],
            "chartevents": ["meas_chartevents_main"],
            "inputevents": ["inputevents_to_rxnorm"],
            "ingredientevents": ["inputevents_to_rxnorm"],
            "outputevents": ["outputevents_to_loinc"],
            "datetimeevents": ["proc_datetimeevents"],
            "procedureevents": ["proc_itemid", "proc_datetimeevents"],
        }
        return files.get(event_name, [])

    def _load_concept_map_index(self, map_file):
        if map_file in self._concept_map_index_cache:
            return self._concept_map_index_cache[map_file]

        if not self.concept_map_dir:
            self._concept_map_index_cache[map_file] = {}
            return {}

        fp = os.path.join(self.concept_map_dir, f"{map_file}.csv")
        if not os.path.exists(fp):
            self._concept_map_index_cache[map_file] = {}
            return {}

        idx = process_concept_map_item_code(fp)
        self._concept_map_index_cache[map_file] = idx
        return idx

    def _resolve_itemid_mapping_index(self, event_name, mapping, default_index_datas):
        if not (
            self.itemid_representation == "code"
            and mapping.get("input_name") == "itemid"
            and mapping.get("output_name") == "item_name"
        ):
            return default_index_datas

        merged = {}
        for map_file in self._concept_map_files_for_event(event_name):
            merged.update(self._load_concept_map_index(map_file))

        final_index = {}
        for src_id in default_index_datas.keys():
            src_key = str(src_id)
            final_index[src_key] = merged.get(src_key, f"ITEMID/{src_key}")
        return final_index

    def get_item_index_for_event(self):
        index_cache_dir = os.path.join(self.cache_dir, "index")
        os.makedirs(index_cache_dir, exist_ok=True)

        for event_name, event_config in self.event_info.items():
            if not event_config["item_mapping"]:
                continue
            event_type = event_config["event_type"]

            for mapping in event_config["item_mapping"]:
                event_mapping_file = mapping["item_index_file"]
                cache_tag = self.itemid_representation
                item_index_cache_file = os.path.join(
                    index_cache_dir,
                    event_type,
                    f"{event_mapping_file}.{cache_tag}.json",
                )

                if os.path.exists(item_index_cache_file):
                    with open(item_index_cache_file, "r") as f:
                        index_datas = json.load(f)
                else:
                    base_index = mapping["item_index_preprocess_func"](
                        os.path.join(self.origin_data_dir, event_type, f"{event_mapping_file}.csv")
                    )
                    index_datas = self._resolve_itemid_mapping_index(
                        event_name=event_name,
                        mapping=mapping,
                        default_index_datas=base_index,
                    )
                    os.makedirs(os.path.dirname(item_index_cache_file), exist_ok=True)
                    with open(item_index_cache_file, "w") as f:
                        json.dump(index_datas, f, indent=4, ensure_ascii=False)

                mapping["item_index"] = index_datas

    @staticmethod
    def mapping_item(item_lists, item_mappings):
        if not item_mappings:
            return item_lists

        for item in item_lists:
            for item_mapping in item_mappings:
                input_name = item_mapping["input_name"]
                if input_name == "icd_code":
                    item[item_mapping["output_name"]] = item_mapping["item_index"][item["icd_version"]].get(
                        item[input_name],
                        None,
                    )
                elif input_name == "ndc":
                    ndc_code = item[input_name].split(".", 1)[0]
                    ndc_code = ("0" * (11 - len(ndc_code))) + ndc_code
                    item[item_mapping["output_name"]] = item_mapping["item_index"].get(ndc_code, None)
                else:
                    item[item_mapping["output_name"]] = item_mapping["item_index"].get(item[input_name], None)

        return item_lists

    def item_info_process(self, event_item, key_info):
        event_name = event_item["file_name"]
        if event_name not in self.event_info:
            return {}

        item_info = {}
        for task_name, task_key_info in key_info.items():
            item_info[task_name] = self.output_process(task_name, event_item, task_key_info)

        return item_info

    def output_process(self, task_name, item, target_key):
        event_name = item["file_name"]
        if event_name not in self.event_info:
            return False

        item_lists = self.mapping_item(item["items"], self.event_info[event_name]["item_mapping"])

        if task_name == "next_event":
            if safe_read(item["starttime"]):
                return item[target_key]
            return False

        output = [safe_read(_[target_key]) for _ in item_lists]
        output = [o for o in output if o]

        if task_name == "discharge":
            if len(output) > 1:
                return False
            return output[0]

        if task_name == "transfers":
            output = []
            for output_item in item_lists:
                eventtype = output_item["eventtype"]
                careunit = output_item["careunit"]
                output.append(careunit if careunit != "nan" else eventtype)
        elif task_name == "radiology":
            output = [o for output_list in output for o in output_list]

        output = [o.strip() for o in output if o.strip()]
        output = list(set(output))
        if output:
            return output
        return False
