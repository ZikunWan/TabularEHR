import os
import sys
import json
from copy import deepcopy
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import re
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.renji.task_info import ALL_METRICS as TASK_ALL_METRICS, ALL_POINTS as TASK_ALL_POINTS, PREDICTION_POINTS as TASK_PREDICTION_POINTS, get_task_info


class RenjiDataset(Dataset):
    """
    Renji Pediatric Liver Transplant Dataset.
    
    Predicts outcomes at 6 fixed time points per patient:
    - Day 14: Predict 2w-1m labels
    - Day 30: Predict 2m-6m labels
    - Day 180: Predict 7m-12m labels
    - Day 365: Predict 13m-14m labels
    - Day 395: Predict 15m-24m labels
    - Day 730: Predict 2y+ labels
    """
    
    # Prediction points: (cutoff_day, label_prefix, readable_name)
    # Note: the 0-2w window was removed due to extreme label imbalance.
    PREDICTION_POINTS = {
        'day14': (14, '2w-1m', 'Day 14'),
        'day30': (30, '2m-6m', 'Day 30'),
        'day180': (180, '7m-12m', 'Day 180'),
        'day365': (365, '13m-14m', 'Day 365'),
        'day395': (395, '15m-24m', 'Day 395'),
        'day730': (730, '2y+', 'Day 730')
    }
    
    # Static features from patient info: {column_name: readable_name}
    STATIC_FEATURES = {
        'recipient_gender': 'Recipient Gender',
        'recipient_weight': 'Recipient Weight (kg)',
        'donor_liver_weight': 'Donor Liver Weight (g)',
        'GRWR cl': 'GRWR',
        'operation_type_code': 'Operation Type',
        'recipient_blood_type': 'Recipient Blood Type',
        'recipient_cyp3a5_genotype': 'Recipient CYP3A5 Genotype',
        'donor_gender': 'Donor Gender',
        'donor_age': 'Donor Age',
        'donor_blood_type': 'Donor Blood Type',
        'donor_cyp3a5_genotype': 'Donor CYP3A5 Genotype',
        'ABO_compatibility': 'ABO Compatibility',
    }
    
    # Combined list of all metrics for multi-task learning (sorted for consistency)
    ALL_METRICS = sorted([
        'ALT', 'AST', 'ALP', 'GGT', 'TB', 'DB', 'Bile_Acid', 'TP', 'ALB', 'PT', 'INR', # Graft Injury
        'Tacrolimus_Conc', 'CsA_Trough', 'CsA_Peak', #'Rapa_Conc',                     # Drug Conc
        'WBC', 'N_Percent', 'Lymphocyte_Abs', 'HB', 'PLT', #'Eosinophil_Percent',      # Immune/Infection
        'CR', 'Glucose', 'Uric_Acid', 'Triglyceride', 'Cholesterol', #'Blood_Ammonia', # Metabolic/Renal
        'CMV_DNA', 'EBV_DNA', 'HBV_DNA', # 'HBsAg', 'HBsAb', 'HBeAg', 'HBeAb', 'HBcAb' # Virus
    ])
    
    # Combined list of all prediction points (sorted by day)
    ALL_POINTS = ['day14', 'day30', 'day180', 'day365', 'day395', 'day730']
    TASK_INFO = get_task_info()
    TASK_ALL_METRICS = TASK_ALL_METRICS
    TASK_ALL_POINTS = TASK_ALL_POINTS
    TASK_PREDICTION_POINTS = TASK_PREDICTION_POINTS

    def __init__(self, 
                 root_dir, 
                 split='train', 
                 max_samples=None,
                 table_mode='table_only',
                 target_metrics=None,
                 target_prediction_points=None,
                 shuffle=False,
                 task_mode='single' # 'single' or 'multi_task'
                 ):
        """
        Initialize RenjiDataset.
        
        Args:
            root_dir: Root directory containing the data
            split: Data split ('train', 'val', 'test')
            max_samples: Maximum number of samples to use
            table_mode: One of 'text_only', 'table_only', or 'table_plus_rest_text'
            target_metrics: List of metric names to filter (e.g., ['ALT', 'AST', 'TB'])
                          If None, use all metrics
            target_prediction_points: List of prediction point keys to filter
                          (e.g., ['day14', 'day365', 'day730']). If None, use all.
            shuffle: Whether to shuffle the dataset
            task_mode: 'single' for one sample per (patient, point, metric),
                       'multi_task' for one sample per (patient, point) with matrix label.
        """
        self.root_dir = root_dir
        self.split = split
        self.max_samples = max_samples
        if table_mode not in {'text_only', 'table_only', 'table_plus_rest_text'}:
            raise ValueError(f"Invalid table_mode: {table_mode}")
        self.table_mode = table_mode
        self.target_metrics = target_metrics
        self.target_prediction_points = target_prediction_points
        self.shuffle = shuffle
        self.task_mode = task_mode
        self.task_schema = get_task_info()
        
        if self.table_mode in {'table_only', 'table_plus_rest_text'}:
            self.task_mode = 'multi_label'
            
        if self.task_mode not in ['single', 'multi_task', 'multi_label']:
            raise ValueError(f"Invalid task_mode: {self.task_mode}. Must be 'single', 'multi_task', or 'multi_label'.")
        
        self.followup_dir = os.path.join(self.root_dir, 'follow_ups')
        self.index_dir = os.path.join(self.root_dir, 'index')
        
        self._init_configs()
        self._load_auxiliary_data()
        
        # Load split file list
        split_path = os.path.join(self.index_dir, f'{split}_renji.json')
        with open(split_path, 'r', encoding='utf-8') as f:
            self.filenames = json.load(f)
        
        self.samples = self._build_index()

    def _init_configs(self):
        """Initialize feature configurations."""
        
        # Columns to exclude from follow-up data (prediction targets, not inputs)
        self.EXCLUDED_COLS = {
            '免疫抑制剂浓度',
            'HBsAg', 'HBsAb', 'HBeAg', 'HBeAb', 'HBcAb',
            '细菌真菌感染', '排斥', 'CMV感染', 'EBV感染', 'HBV感染'
        }
    
        # Medication columns
        self.MED_COLS_ZH = {
            '他克莫司缓释胶囊', '他克莫司(赛福开)', '他克莫司(普乐可复)', 
            '环孢素', '新山地明', '吗替麦考酚酯', '骁悉', '赛可平', 
            '米芙', '雷帕鸣', '醋酸泼尼松', '甲泼尼龙片', '美卓乐'
        }

        # Chinese to English mapping
        self.ZH_TO_EN = {
            # Medications
            '他克莫司缓释胶囊': 'Tacrolimus_ER',
            '他克莫司(赛福开)': 'Tacrolimus_Saifukai',
            '他克莫司(普乐可复)': 'Tacrolimus_Prograf',
            '环孢素': 'Cyclosporine',
            '新山地明': 'Sandimmun',
            '吗替麦考酚酯': 'MMF',
            '骁悉': 'CellCept',
            '赛可平': 'Saikeping',
            '米芙': 'Mifu',
            '雷帕鸣': 'Rapamune',
            '醋酸泼尼松': 'Prednisone',
            '甲泼尼龙片': 'Methylprednisolone',
            '美卓乐': 'Medrol',

            # Labs & Vitals
            '身高': 'Height',
            '体重': 'Weight',
            'WBC': 'WBC',
            'N(%)': 'N_Percent',
            '淋巴细胞绝对值': 'Lymphocyte_Abs',
            '嗜酸性粒细胞百分比': 'Eosinophil_Percent',
            'HB': 'HB',
            'PLT': 'PLT',
            'TP': 'TP',
            'ALB': 'ALB',
            'ALT': 'ALT',
            'AST': 'AST',
            'ALP': 'ALP',
            'γ-GT': 'GGT',
            'DB': 'DB',
            'TB': 'TB',
            '胆汁酸': 'Bile_Acid',
            'CR': 'CR',
            '血糖': 'Glucose',
            '甘油三脂': 'Triglyceride', 
            '总胆固醇': 'Cholesterol',
            '尿酸': 'Uric_Acid',
            'PT': 'PT',
            'INR': 'INR',
            '雷帕浓度': 'Rapa_Conc',
            '血氨': 'Blood_Ammonia',
            '他克莫司浓度': 'Tacrolimus_Conc',
            '环孢素谷浓度': 'CsA_Trough',
            '环孢素峰浓度': 'CsA_Peak',
            'CMV-DNA': 'CMV_DNA',
            'EBV-DNA': 'EBV_DNA',
            'HBV-DNA': 'HBV_DNA',
            'HBsAg': 'HBsAg',
            'HBsAb': 'HBsAb',
            'HBeAg': 'HBeAg',
            'HBeAb': 'HBeAb',
            'HBcAb': 'HBcAb'
        }
        
        self.TASK_GROUPS = {
            "graft_injury": {
                "name": "Graft Injury Assessment",
                "features": ['ALT', 'AST', 'ALP', 'GGT', 'TB', 'DB', 'Bile_Acid', 'TP', 'ALB', 'PT', 'INR']
            },
            "drug_conc": {
                "name": "Drug Concentration Monitoring",
                "features": ['Tacrolimus_Conc', 'CsA_Trough', 'CsA_Peak']
            },
            "immune_infection": {
                "name": "Immune & Infection Assessment",
                "features": ['WBC', 'N_Percent', 'Lymphocyte_Abs', 'HB', 'PLT']
            },
            "metabolic_renal": {
                "name": "Metabolic & Renal Function",
                "features": ['CR', 'Glucose', 'Uric_Acid', 'Triglyceride', 'Cholesterol']
            },
            "virus_activation": {
                "name": "Virus Activation Monitoring",
                "features": ['CMV_DNA', 'EBV_DNA', 'HBV_DNA']
            }
        }

        self.med_cols_en = {self.ZH_TO_EN.get(c, c) for c in self.MED_COLS_ZH}

        # Medication metadata
        self.MED_META = {
            'Tacrolimus_ER':      {'str': '24 HR tacrolimus 0.5 MG Extended Release Oral Capsule', 'unit': 'mg', 'MEDS_CODE': 'RxNorm/1431971'},
            'Tacrolimus_Saifukai':{'str': 'tacrolimus 0.5 MG Oral Capsule', 'unit': 'mg', 'MEDS_CODE': 'RxNorm/313190'},
            'Tacrolimus_Prograf': {'str': 'tacrolimus 0.5 MG Oral Capsule', 'unit': 'mg', 'MEDS_CODE': 'RxNorm/313190'},
            'Tacrolimus':         {'str': 'tacrolimus 0.5 MG Oral Capsule', 'unit': 'mg', 'MEDS_CODE': 'RxNorm/313190'},
            'Cyclosporine':       {'str': 'cycloSPORINE 25 MG Oral Capsule', 'unit': 'mg', 'MEDS_CODE': 'RxNorm/197553'},
            'Sandimmun':          {'str': 'cycloSPORINE 25 MG Oral Capsule', 'unit': 'mg', 'MEDS_CODE': 'RxNorm/197553'},
            'Rapamune':           {'str': 'sirolimus 1 MG Oral Tablet', 'unit': 'mg', 'MEDS_CODE': 'RxNorm/349208'},
            'Prednisone':         {'str': 'predniSONE 5 MG Oral Tablet', 'unit': 'mg', 'MEDS_CODE': 'RxNorm/312617'},
            'Methylprednisolone': {'str': 'methylPREDNISolone 4 MG Oral Tablet', 'unit': 'mg', 'MEDS_CODE': 'RxNorm/259966'},
            'Medrol':             {'str': 'methylPREDNISolone 4 MG Oral Tablet', 'unit': 'mg', 'MEDS_CODE': 'RxNorm/259966'},
            'Mifu':               {'str': 'mycophenolic acid 180 MG Delayed Release Oral Tablet', 'unit': 'mg', 'MEDS_CODE': 'RxNorm/485020'},
            'MMF':                {'str': 'mycophenolate mofetil 250 MG Oral Capsule', 'unit': 'g', 'MEDS_CODE': 'RxNorm/199058'},
            'CellCept':           {'str': 'mycophenolate mofetil 250 MG Oral Capsule', 'unit': 'g', 'MEDS_CODE': 'RxNorm/199058'},
            'Saikeping':          {'str': 'mycophenolate mofetil 250 MG Oral Capsule', 'unit': 'g', 'MEDS_CODE': 'RxNorm/199058'},
        }

        # Value mapping for static features
        self.ZH_VALUE_MAP = {
            # Gender
            '男': 'Male',
            '女': 'Female',
            # Operation Type
            '活体': 'Living Donor', 
            '劈离': 'Split Liver',
            '全肝': 'Whole Liver',
            # ABO Compatibility
            '不相容': 'Incompatible',
            '相容': 'Compatible',
            '相同': 'Identical',
        }
        
        # Lab reference ranges (based on pediatric liver transplant standards)
        # Age in years, ranges list: [(age_min, age_max, gender, low, high), ...]
        # gender: 'M'=male, 'F'=female, None=both
        # Age ranges: 0-0.5 (28天-6月), 0.5-1, 1-3, 3-6, 6-12, 12-18, 18+
        self.LAB_META = {            
            'WBC': {
                'unit': '10^9/L',
                'ranges': [
                    (0, 0.5, None, 4.3, 14.2), (0.5, 1, None, 4.8, 14.6),
                    (1, 2, None, 5.1, 14.1), (2, 6, None, 4.4, 11.9),
                    (6, 13, None, 4.3, 11.3), (13, 99, None, 4.1, 11.0)
                ],
                "LONIC": "Leukocytes [#/volume] in Blood by Automated count",
                "MEDS_CODE": "LOINC/6690-2"
            },
            'N_Percent': {
                'unit': '%',
                'ranges': [
                    (0, 0.5, None, 7, 56), (0.5, 1, None, 9, 57),
                    (1, 2, None, 13, 55), (2, 6, None, 22, 65),
                    (6, 13, None, 31, 70), (13, 99, None, 37, 77)
                ],
                "LONIC": "Neutrophils/Leukocytes in Blood by Automated count",
                "MEDS_CODE": "LOINC/770-8"
            },
            'Lymphocyte_Abs': {
                'unit': '10^9/L',
                'ranges': [
                    (0, 0.5, None, 2.4, 9.5), (0.5, 1, None, 2.5, 9.0),
                    (1, 2, None, 2.4, 8.7), (2, 6, None, 1.8, 6.3),
                    (6, 13, None, 1.5, 4.6), (13, 99, None, 1.2, 3.8)
                ],
                "LONIC": "Lymphocytes [#/volume] in Blood by Automated count",
                "MEDS_CODE": "LOINC/731-0"
            },
            'Eosinophil_Percent': {'unit': '%',
                                   'ranges': [(0, 2, None, 0.0, 0.1), (2, 99, None, 0.0, 0.07)],
                                   "LONIC": "Eosinophils/Leukocytes in Blood by Automated count",
                                   "MEDS_CODE": "LOINC/713-8"},
            'HB': {
                'unit': 'g/L',
                'ranges': [
                    (0, 0.5, None, 97, 183), (0.5, 1, None, 97, 141),
                    (1, 2, None, 107, 141), (2, 6, None, 112, 149),
                    (6, 13, None, 118, 156),
                    (13, 99, 'M', 129, 172), (13, 99, 'F', 114, 154)
                ],
                "LONIC": "Hemoglobin [Mass/volume] in Central venous blood by calculation",
                "MEDS_CODE": "LOINC/97550-8"
            },
            'PLT': {
                'unit': '10^9/L',
                'ranges': [
                    (0, 0.5, None, 183, 614), (0.5, 1, None, 190, 445),
                    (1, 2, None, 190, 472), (2, 6, None, 188, 472),
                    (6, 13, None, 167, 453), (13, 18, None, 150, 407),
                ],
                "LONIC": "Platelets [#/volume] in Blood by Automated count",
                "MEDS_CODE": "LOINC/777-3"
            },
            'TP': {
                'unit': 'g/L',
                'ranges': [
                    (0, 0.5, None, 49, 71), (0.5, 1, None, 55, 75),
                    (1, 2, None, 58, 76), (2, 6, None, 61, 79),
                    (6, 13, None, 65, 84), (13, 99, None, 68, 88)
                ],
                "LONIC": "Protein [Mass/volume] in Serum or Plasma",
                "MEDS_CODE": "LOINC/2885-2"
            },
            'ALB': {
                'unit': 'g/L',
                'ranges': [
                    (0, 0.5, None, 35, 50), (0.5, 13, None, 39, 54),
                    (13, 99, None, 42, 56),
                ],
                "LONIC": "Albumin in serum - albumin in pleural fluid [Mass concentration difference]",
                "MEDS_CODE": "LOINC/72647-1"
            },
            'ALT': {
                'unit': 'U/L',
                'ranges': [
                    (0, 1, None, 8, 71), (1, 2, None, 8, 42),
                    (2, 13, None, 7, 30),
                    (13, 99, 'M', 7, 43), (13, 99, 'F', 6, 29)
                ],
                "LONIC": "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
                "MEDS_CODE": "LOINC/1742-6"
            },
            'AST': {
                'unit': 'U/L',
                'ranges': [
                    (0, 1, None, 21, 80), (1, 2, None, 22, 59),
                    (2, 13, None, 14, 44),
                    (13, 99, 'M', 12, 37), (13, 99, 'F', 10, 31)
                ],
                "LONIC": "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
                "MEDS_CODE": "LOINC/1920-8"
            },
            'ALP': {
                'unit': 'U/L',
                'ranges': [
                    (0, 0.5, None, 98, 532), (0.5, 1, None, 106, 420),
                    (1, 2, None, 128, 432), (2, 9, None, 143, 406),
                    (9, 12, None, 146, 500),
                    (12, 14, 'M', 160, 610), (12, 14, 'F', 81, 454),
                    (14, 15, 'M', 82, 603), (14, 15, 'F', 63, 327),
                    (15, 17, 'M', 64, 443), (15, 17, 'F', 52, 215),
                    (17, 99, 'M', 51, 202), (17, 99, 'F', 43, 130),
                ],
                "LONIC": "Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma",
                "MEDS_CODE": "LOINC/6768-6"
            },
            'GGT': {
                'unit': 'U/L',
                'ranges': [
                    (0, 0.5, None, 9, 150), (0.5, 1, None, 6, 31),
                    (1, 13, None, 5, 19), (13, 99, 'M', 8, 40), (13, 99, 'F', 6, 26)
                ],
                "LONIC": "Gamma glutamyl transferase [Enzymatic activity/volume] in Serum or Plasma",
                "MEDS_CODE": "LOINC/2324-2"
            },
            'DB': {'unit': 'μmol/L',
                   'ranges': [(0, 99, None, 0, 6.84)],
                   "LONIC": "Bilirubin.direct [Moles/volume] in Serum or Plasma",
                   "MEDS_CODE": "LOINC/14629-0"},
            'TB': {
                'unit': 'μmol/L',
                'ranges': [(0, 99, None, 0, 23)],
                "LONIC": "Bilirubin.total [Moles/volume] in Serum or Plasma",
                "MEDS_CODE": "LOINC/14631-6"
            },
            'Bile_Acid': {'unit': 'μmol/L',
                          'ranges': [(0, 99, None, 0.01, 10)],
                          "LONIC": "Bile acid [Moles/volume] in Serum or Plasma",
                          "MEDS_CODE": "LOINC/14628-2"},
            'CR': {
                'unit': 'μmol/L',
                'ranges': [
                    (0, 2, None, 13, 33), (2, 6, None, 19, 44),
                    (6, 13, None, 27, 66),
                    (13, 16, 'M', 37, 93), (13, 16, 'F', 33, 75),
                    (16, 99, 'M', 52, 101), (16, 99, 'F', 39, 76),
                ],
                "LONIC": "Creatinine [Moles/volume] in Blood",
                "MEDS_CODE": "LOINC/59826-8"
            },
            'Glucose': {
                'unit': 'mmol/L',
                'ranges': [(0, 99, None, 3.9, 6.1)],
                "LONIC": "Glucose [Moles/volume] in Capillary blood by Glucometer",
                "MEDS_CODE": "LOINC/14743-9"
            },
            'Triglyceride': {'unit': 'mmol/L',
                             'ranges': [(0, 99, None, 0, 1.7)],
                             'severity_bands': [
                                {'label': 'Normal', 'range': (0, 99, None, 0, 1.7)},
                                {'label': 'Elevated', 'range': (0, 99, None, 1.7, 2.3)},
                                {'label': 'Very High', 'range': (0, 99, None, 2.3, float('inf'))},
                            ],
                             "LONIC": "Triglyceride [Moles/volume] in Serum or Plasma",
                             "MEDS_CODE": "LOINC/14927-8"},
            'Cholesterol': {'unit': 'mmol/L',
                            'ranges': [(0, 99, None, 0, 5.2)],
                            'severity_bands': [
                                {'label': 'Normal', 'range': (0, 99, None, 0, 5.2)},
                                {'label': 'Elevated', 'range': (0, 99, None, 5.2, 6.2)},
                                {'label': 'Very High', 'range': (0, 99, None, 6.2, float('inf'))},
                            ],
                            "LONIC": "Cholesterol [Moles/volume] in Serum or Plasma",
                            "MEDS_CODE": "LOINC/14647-2"},
            'Uric_Acid': {
                'unit': 'μmol/L',
                'ranges': [
                    (0, 99, None, 155, 428),
                ],
                "LONIC": "Urate [Mass/volume] in Serum or Plasma",
                "MEDS_CODE": "LOINC/3084-1"
            },
            'PT': {'unit': 's',
                   'ranges': [(0, 99, None, 9.4, 12.5)],
                   "LONIC": "Prothrombin time (PT) in Blood by Coagulation assay",
                    "MEDS_CODE": "LOINC/5964-2"},
            'INR': {'unit': '',
                    'ranges': [(0, 99, None, 0.8, 1.15)],
                    "LONIC": "INR in Blood by Coagulation assay",
                    "MEDS_CODE": "LOINC/34714-6"},
            'Blood_Ammonia': {'unit': 'μmol/L',
                              'ranges': [(0, 99, None, 9, 30)],
                              "LONIC": "Ammonia [Moles/volume] in Plasma",
                              "MEDS_CODE": "LOINC/16362-6"},
            'Tacrolimus_Conc': {'unit': 'ng/mL',
                                'ranges': [(0, 99, None, 0, 30)],
                                "LONIC": "Tacrolimus [Mass/volume] in Serum or Plasma",
                                "MEDS_CODE": "LOINC/32721-3"},
            'CsA_Trough': {'unit': 'ng/mL',
                           'ranges': [(0, 99, None, 100, 400)],
                           "LONIC": "cycloSPORINE [Mass/volume] in Blood",
                           "MEDS_CODE": "LOINC/3520-4"},
            'CsA_Peak': {'unit': 'ng/mL',
                         'ranges': [(0, 99, None, 400, 1600)],
                         "LONIC": "cycloSPORINE [Mass/volume] in Blood --2 hours post dose",
                         "MEDS_CODE": "LOINC/32997-9"},
            'Rapa_Conc': {'unit': 'ng/mL',
                          'ranges': [(0, 99, None, 5, 20)],
                          "LONIC": "Sirolimus [Mass/volume] in Blood",
                          "MEDS_CODE": "LOINC/29247-4"},
            'CMV_DNA': {'unit': 'copies/mL',
                        'ranges': [(0, 99, None, 0, 400)],
                        "LONIC": "Cytomegalovirus DNA [#/volume] (viral load) in Blood by NAA with probe detection",
                        "MEDS_CODE": "LOINC/29604-6"},
            'EBV_DNA': {'unit': 'copies/mL',
                        'ranges': [(0, 99, None, 0, 400)],
                        "LONIC": "Epstein Barr virus DNA [#/volume] (viral load) in Specimen by NAA with probe detection",
                        "MEDS_CODE": "LOINC/32585-2"},
            'HBV_DNA': {'unit': 'IU/mL',
                        'ranges': [(0, 99, None, 0, 20)],
                        "LONIC": "Hepatitis B virus DNA [Units/volume] (viral load) in Serum or Plasma by NAA with probe detection",
                        "MEDS_CODE": "LOINC/42595-9"},
            'HBsAg': {'unit': 'COI',
                      'ranges': [(0, 99, None, 0, 1)],
                      "LONIC": "Hepatitis B virus surface Ag [Presence] in Serum or Plasma by Immunoassay",
                      "MEDS_CODE": "LOINC/5196-1"},
            'HBsAb': {'unit': 'mIU/mL',
                      'ranges': [(0, 99, None, 0, 10)],
                      "LONIC": "Hepatitis B virus surface Ab [Units/volume] in Serum or Plasma by Immunoassay",
                      "MEDS_CODE": "LOINC/5193-8"},
            'HBeAg': {'unit': 'COI',
                      'ranges': [(0, 99, None, 0, 1)],
                      "LONIC": "Hepatitis B virus e Ag [Presence] in Serum or Plasma by Immunoassay",
                      "MEDS_CODE": "LOINC/13954-3"},
            'HBeAb': {'unit': 'COI', 'ranges': [(0, 99, None, 1, float('inf'))],
                      "LONIC": "Hepatitis B virus e Ab [Presence] in Serum or Plasma by Immunoassay",
                      "MEDS_CODE": "LOINC/13953-5"},
            'HBcAb': {'unit': 'COI', 'ranges': [(0, 99, None, 1, float('inf'))],
                      "LONIC": "Hepatitis B virus core Ab [Presence] in Serum or Plasma by Immunoassay",
                      "MEDS_CODE": "LOINC/13952-7"},
        }

    def _get_reference_range(self, lab_item, age_years, gender=None):
        """
        Get reference range for a lab item based on age and gender.
        
        Args:
            lab_item: Lab item name (English)
            age_years: Patient age in years (float)
            gender: 'M' for male, 'F' for female, or None
        
        Returns:
            (low, high, unit, range_str) or (None, None, unit, '-') if not found
        """
        if lab_item not in self.LAB_META:
            return None, None, '-', '-'
        
        meta = self.LAB_META[lab_item]
        unit = meta['unit']
        ranges = meta['ranges']
        
        # Find matching range
        for age_min, age_max, range_gender, low, high in ranges:
            if age_min <= age_years < age_max:
                # Check gender match
                if range_gender is None or range_gender == gender:
                    range_str = f"{low} - {high}" if high != float('inf') else f"> {low}"
                    return low, high, unit, range_str
        
        # No exact match, try to find any range without gender restriction
        for age_min, age_max, range_gender, low, high in ranges:
            if age_min <= age_years < age_max and range_gender is None:
                range_str = f"{low} - {high}" if high != float('inf') else f"> {low}"
                return low, high, unit, range_str
        
        # Fallback: use last range as default
        if ranges:
            _, _, _, low, high = ranges[-1]
            range_str = f"{low} - {high}" if high != float('inf') else f"> {low}"
            return low, high, unit, range_str
        
        return None, None, unit, '-'

    def _get_severity_flag(self, lab_item, value, age_years, gender=None):
        """
        Get a finer-grained severity flag for lab items that define severity_bands.

        Returns:
            A severity label such as "Normal", "Elevated", or "Very High",
            or None if no matching severity band is defined.
        """
        if lab_item not in self.LAB_META:
            return None

        meta = self.LAB_META[lab_item]
        severity_bands = meta.get('severity_bands')
        if not severity_bands:
            return None

        try:
            numeric_value = float(value)
        except Exception:
            return None

        for band in severity_bands:
            band_range = band.get('range')
            label = band.get('label')
            if not band_range or not label or len(band_range) != 5:
                continue

            age_min, age_max, range_gender, low, high = band_range
            if not (age_min <= age_years < age_max):
                continue
            if range_gender is not None and range_gender != gender:
                continue

            # Severity bands use left-inclusive / right-exclusive intervals so
            # shared boundaries like 1.7 or 5.2 fall into the higher category.
            if high == float('inf'):
                in_range = numeric_value >= low
            else:
                in_range = low <= numeric_value < high

            if in_range:
                return label

        return None

    def _split_multivalue_parts(self, value):
        """Split a value like '100-200' or '阴性-<400' into meaningful parts."""
        value_str = str(value).strip()
        if not value_str or '-' not in value_str:
            return [value_str]

        raw_parts = [part.strip() for part in value_str.split('-') if part.strip()]
        if len(raw_parts) < 2:
            return [value_str]

        def _is_meaningful_part(part):
            part_lower = part.lower()
            if part in {"阴性", "阳性"} or part_lower in {"negative", "positive", "normal", "abnormal"}:
                return True
            if "<" in part or ">" in part:
                return True
            try:
                float(part)
                return True
            except Exception:
                return False

        if all(_is_meaningful_part(part) for part in raw_parts):
            return raw_parts
        return [value_str]

    def _describe_lab_value(self, lab_item, value, age_years, gender=None):
        """
        Build display-facing lab value information for text/table rendering.

        Returns:
            dict with keys: value, unit, flag, low_str, high_str
        """
        low_limit, high_limit, unit, _ = self._get_reference_range(
            lab_item, age_years if age_years else 5.0, gender
        )
        low_str = str(low_limit) if low_limit is not None else "-"
        high_str = str(high_limit) if high_limit is not None and high_limit != float('inf') else "-"

        raw_value = str(value).strip()
        raw_lower = raw_value.lower()
        flag = "Unknown"
        display_value = raw_value
        display_unit = unit

        if raw_value in {"阴性"} or raw_lower in {"negative"}:
            return {
                "value": "Normal",
                "unit": "",
                "flag": "Normal",
                "low_str": "-",
                "high_str": "-",
            }

        if raw_value in {"阳性"} or raw_lower in {"positive"}:
            return {
                "value": "Abnormal",
                "unit": "",
                "flag": "Abnormal",
                "low_str": "-",
                "high_str": "-",
            }

        if "<" in raw_value:
            return {
                "value": display_value,
                "unit": display_unit,
                "flag": "Normal",
                "low_str": low_str,
                "high_str": high_str,
            }

        if ">" in raw_value:
            return {
                "value": display_value,
                "unit": display_unit,
                "flag": "Abnormal",
                "low_str": low_str,
                "high_str": high_str,
            }

        try:
            numeric_value = float(raw_value)
            severity_flag = self._get_severity_flag(
                lab_item,
                numeric_value,
                age_years if age_years else 5.0,
                gender,
            )
            if severity_flag is not None:
                flag = severity_flag
            elif low_limit is not None and high_limit is not None:
                flag = "Abnormal" if (numeric_value < low_limit or numeric_value > high_limit) else "Normal"
        except Exception:
            pass

        return {
            "value": display_value,
            "unit": display_unit,
            "flag": flag,
            "low_str": low_str,
            "high_str": high_str,
        }

    def _load_auxiliary_data(self):
        """Load labels, patient info, and pathology data."""
        
        # Load labels.csv
        labels_path = os.path.join(self.root_dir, 'labels.csv')
        if os.path.exists(labels_path):
            self.labels_df = pd.read_csv(labels_path, encoding='utf-8-sig')
            
            # Translate Chinese column names to English
            # Column format: "{window}_{metric}" e.g. "2w-1m_胆汁酸" -> "2w-1m_Bile_Acid"
            new_columns = {}
            for col in self.labels_df.columns:
                if '_' in col and col != 'filename':
                    parts = col.split('_', 1)
                    if len(parts) == 2:
                        window, metric = parts
                        metric_en = self.ZH_TO_EN.get(metric, metric)
                        new_columns[col] = f"{window}_{metric_en}"
            if new_columns:
                self.labels_df.rename(columns=new_columns, inplace=True)
            
            if 'filename' in self.labels_df.columns:
                self.labels_df.set_index('filename', inplace=True)
        else:
            self.labels_df = None
            print(f"Warning: labels.csv not found at {labels_path}")
        
        # Load patient info
        patient_info_path = os.path.join(self.root_dir, '患儿基本信息总表251023_含免疫事件.xlsx')
        if os.path.exists(patient_info_path):
            self.patient_info_df = pd.read_excel(patient_info_path)
            # Build lookup map: transplant_id without "_1" suffix -> row
            self.patient_info_map = {}
            for _, row in self.patient_info_df.iterrows():
                tid = str(row.get('transplant_id', ''))
                # Remove trailing "_1", "_2" etc.
                key = re.sub(r'_\d+$', '', tid)
                if key:
                    self.patient_info_map[key] = row
            print(f"Loaded patient info: {len(self.patient_info_map)} patients")
        else:
            self.patient_info_df = None
            self.patient_info_map = {}
            print(f"Warning: patient info not found at {patient_info_path}")
        
        # Load liver pathology (placeholder - matching not yet implemented)
        pathology_path = os.path.join(self.root_dir, '肝脏病理.xlsx')
        if os.path.exists(pathology_path):
            self.pathology_df = pd.read_excel(pathology_path)
            # TODO: Implement matching strategy when available
            self.pathology_map = {}  # Placeholder
            print(f"Loaded pathology: {self.pathology_df.shape} (matching not implemented)")
        else:
            self.pathology_df = None
            self.pathology_map = {}

    def _build_index(self):
        """
        Build sample index.
        - 'single': One sample per (patient, prediction_point, metric)
        - 'multi_task': One sample per (patient, prediction_point)
        """
        if self.labels_df is None:
            print("[WARNING] No labels loaded, returning empty samples")
            return []
        
        print(f"[{self.split}] Building sample index (mode={self.task_mode})...")
        
        # Determine which prediction points to use
        active_points = self.target_prediction_points or list(self.PREDICTION_POINTS.keys())
        
        samples = []
        
        # Pre-calculate metrics for single task mode optimization
        # Get all label columns grouped by window prefix
        all_label_cols = [c for c in self.labels_df.columns if c != 'filename']
        metrics_by_window = {}
        for col in all_label_cols:
            parts = col.split('_', 1)
            if len(parts) == 2:
                window, metric = parts
                if window not in metrics_by_window:
                    metrics_by_window[window] = set()
                metrics_by_window[window].add(metric)
        
        for fname in tqdm(self.filenames, desc=f"[{self.split}] Indexing"):
            # Get filename without extension
            fname_key = os.path.splitext(fname)[0] if fname.endswith(('.xlsx')) else fname
            
            # Check if labels exist for this patient
            if fname_key not in self.labels_df.index:
                continue
            
            patient_labels = self.labels_df.loc[fname_key]
            
            # For each prediction point
            for point_key in active_points:
                if point_key not in self.PREDICTION_POINTS:
                    continue
                    
                cutoff_day, label_prefix, readable_point = self.PREDICTION_POINTS[point_key]
                
                if self.task_mode == 'multi_label' or self.task_mode == 'multi_task':
                    sample = {
                        'fname': fname,
                        'fname_key': fname_key,
                        'prediction_point': point_key,
                        'cutoff_day': cutoff_day,
                        'label_prefix': label_prefix,
                        'metric': 'all',
                        'label_col': 'all',
                        'label_val': -100,
                    }
                    samples.append(sample)
                else:
                    # SINGLE TASK MODE: One sample per metric
                    if label_prefix not in metrics_by_window:
                        continue
                    
                window_metrics = metrics_by_window[label_prefix]
                
                # Filter by target_metrics if specified
                if self.target_metrics:
                    window_metrics = [m for m in window_metrics if m in self.target_metrics]
                
                for metric in window_metrics:
                    label_col = f"{label_prefix}_{metric}"
                    label_val = patient_labels.get(label_col)
                    
                    # Skip if label is NaN
                    if pd.isna(label_val):
                        continue
                    sample = {
                        'fname': fname,
                        'fname_key': fname_key,
                        'prediction_point': point_key,
                        'cutoff_day': cutoff_day,
                        'label_prefix': label_prefix,
                        'metric': metric,
                        'label_col': label_col,
                        'label_val': int(label_val),
                    }
                    samples.append(sample)
                        
        # Apply max_samples
        if self.max_samples and len(samples) > self.max_samples:
            samples = self._balanced_sample(samples, self.max_samples)
            np.random.shuffle(samples)
            samples = samples[:self.max_samples]
        
        # Shuffle if requested
        if self.shuffle:
            np.random.shuffle(samples)
        
        # Print statistics
        self._print_stats(samples)
        
        return samples

    def _balanced_sample(self, samples, target_count):
        """Sample with task and label balancing."""
        from collections import defaultdict
        
        # Group by (prediction_point, metric)
        task_buckets = defaultdict(list)
        for s in samples:
            key = (s['prediction_point'], s['metric'])
            task_buckets[key].append(s)
        
        all_tasks = list(task_buckets.keys())
        sorted_tasks = sorted(all_tasks, key=lambda t: len(task_buckets[t]))
        
        final_samples = []
        remaining_quota = target_count
        remaining_tasks_count = len(all_tasks)
        
        for task in sorted_tasks:
            bucket = task_buckets[task]
            fair_share = remaining_quota // remaining_tasks_count
            take_count = min(len(bucket), fair_share)
            
            # Label-balanced sampling within task
            label_0 = [s for s in bucket if s['label_val'] == 0]
            label_1 = [s for s in bucket if s['label_val'] == 1]
            
            if label_0 and label_1:
                per_label = take_count // 2
                n0 = min(len(label_0), per_label)
                n1 = min(len(label_1), per_label)
                
                idx0 = np.random.choice(len(label_0), n0, replace=False)
                idx1 = np.random.choice(len(label_1), n1, replace=False)
                
                selected = [label_0[i] for i in idx0] + [label_1[i] for i in idx1]
                
                # Fill remaining from larger group
                total = len(selected)
                if total < take_count:
                    remaining = take_count - total
                    used_0 = set(idx0)
                    used_1 = set(idx1)
                    unused_0 = [i for i in range(len(label_0)) if i not in used_0]
                    unused_1 = [i for i in range(len(label_1)) if i not in used_1]
                    
                    if len(unused_0) > len(unused_1):
                        extra = min(remaining, len(unused_0))
                        extra_idx = np.random.choice(unused_0, extra, replace=False)
                        selected.extend([label_0[i] for i in extra_idx])
                    else:
                        extra = min(remaining, len(unused_1))
                        extra_idx = np.random.choice(unused_1, extra, replace=False)
                        selected.extend([label_1[i] for i in extra_idx])
            else:
                # Only one label class
                indices = np.random.choice(len(bucket), min(take_count, len(bucket)), replace=False)
                selected = [bucket[i] for i in indices]
            
            final_samples.extend(selected)
            remaining_quota -= len(selected)
            remaining_tasks_count -= 1
        
        return final_samples

    def _print_stats(self, samples):
        """Print dataset statistics."""
        from collections import Counter
        
        print(f"\n[{self.split}] === Dataset Statistics ===")
        print(f"Total samples: {len(samples)}")
        
        # By prediction point
        point_dist = Counter(s['prediction_point'] for s in samples)
        print(f"\nBy prediction point:")
        for point, count in sorted(point_dist.items()):
            print(f"  {point}: {count}")
        
        # By metric (top 10) - checks if 'metric' exists
        if samples and 'metric' in samples[0]:
            metric_dist = Counter(s['metric'] for s in samples)
            print(f"\nTop 10 metrics:")
            for metric, count in metric_dist.most_common(10):
                print(f"  {metric}: {count}")
        
        # Label distribution - checks if 'label_val' exists
        if samples and 'label_val' in samples[0]:
            label_0 = sum(1 for s in samples if s['label_val'] == 0)
            label_1 = sum(1 for s in samples if s['label_val'] == 1)
            if label_0 > 0:
                print(f"\nLabel distribution: 0={label_0}, 1={label_1}, ratio={label_1/(label_0):.2f}")
            else:
                print(f"\nLabel distribution: 0={label_0}, 1={label_1}")

    def _load_followup_data(self, sample):
        """Load and filter follow-up data for a sample."""
        fname = sample['fname']
        cutoff_day = sample['cutoff_day']
        
        # Determine file path and extension
        fpath = os.path.join(self.followup_dir, fname)
        ext = os.path.splitext(fname)[1]
        
        if not os.path.exists(fpath):
            # Try finding with extensions if exact match fails
            # Prioritize CSV
            for e in ['.csv', '.xlsx', '.xls']:
                alt_path = os.path.join(self.followup_dir, sample['fname_key'] + e)
                if os.path.exists(alt_path):
                    fpath = alt_path
                    ext = e
                    break
        
        # Load based on extension
        if ext.lower() == '.csv':
            df = pd.read_csv(fpath)
        else:
            df = pd.read_excel(fpath)
        
        # Sort by report date
        if '报告日期' in df.columns:
            df['报告日期'] = pd.to_datetime(df['报告日期'], errors='coerce')
            df = df.sort_values('报告日期').reset_index(drop=True)
        
        # Filter by 术后天数
        if '术后天数' in df.columns:
            if cutoff_day == 0:
                # Day 0: Use first record or pre-surgery records (术后天数 <= 0)
                mask = df['术后天数'] <= 0
                if mask.sum() == 0:
                    # No pre-surgery records, use first row
                    df_filtered = df.iloc[:1].copy()
                else:
                    df_filtered = df[mask].copy()
            else:
                # Filter records up to cutoff_day
                df_filtered = df[df['术后天数'] <= cutoff_day].copy()
        else:
            # No 术后天数 column, use all data
            df_filtered = df.copy()
        
        # Remove excluded and label columns
        cols_to_keep = []
        for c in df_filtered.columns:
            if c in self.EXCLUDED_COLS:
                continue
            if c.endswith('_label'):
                continue
            cols_to_keep.append(c)
        
        df_clean = df_filtered[cols_to_keep].copy()
        
        # Rename columns to English
        new_columns = [self.ZH_TO_EN.get(c, c) for c in df_clean.columns]
        df_clean.columns = new_columns
        
        return df_clean

    def _get_static_features(self, fname_key):
        """Get static patient features as a dict with readable names."""
        if fname_key not in self.patient_info_map:
            return {}
        
        row = self.patient_info_map[fname_key]
        features = {}  # {readable_name: value}
        for col_name, readable_name in self.STATIC_FEATURES.items():
            val = row.get(col_name)
            if pd.notna(val):
                features[readable_name] = val
        return features

    def _get_pathology_data(self, fname_key):
        """Get pathology data (placeholder - not implemented)."""
        # TODO: Implement when matching strategy is available
        return None

    def _build_static_table(self, features, surgery_date=None):
        """Convert static features to DataFrame table."""
        if not features:
            return None
        
        # Use provided surgery_date or NaT (not empty string)
        # Ensure it's treated as a timestamp compatible with other tables
        time_val = surgery_date if surgery_date is not None else pd.NaT
        
        data = []
        for k, v in features.items():
            # Map value if it's Chinese
            v_str = str(v).strip()
            if v_str in self.ZH_VALUE_MAP:
                v_str = self.ZH_VALUE_MAP[v_str]
            
            item_name = self.STATIC_FEATURES.get(k, k)
            unit = "" # Default to empty string instead of "-"
            
            # Handle special static features with units
            if 'Weight (kg)' in item_name:
                item_name = item_name.replace(' (kg)', '')
                unit = 'kg'
            elif 'Weight (g)' in item_name:
                item_name = item_name.replace(' (g)', '')
                unit = 'g'
            
            data.append({
                'Time': time_val,
                'Item': item_name,
                'Value': v_str,
                'Unit': unit,
                'Category': 'person'
            })
        return pd.DataFrame(data)

    def _build_static_text(self, features):
        """Convert static features to markdown text."""
        lines = ["## Patient Information\n| Feature | Value | Unit |\n| --- | --- | --- |"]
        for k, v in features.items():
            # Map value if it's Chinese
            v_str = str(v).strip()
            if v_str in self.ZH_VALUE_MAP:
                v_str = self.ZH_VALUE_MAP[v_str]
            
            item_name = self.STATIC_FEATURES.get(k, k)
            unit = "-"
            
            # Handle special static features with units
            if 'Weight (kg)' in item_name:
                item_name = item_name.replace(' (kg)', '')
                unit = 'kg'
            elif 'Weight (g)' in item_name:
                item_name = item_name.replace(' (g)', '')
                unit = 'g'
                
            lines.append(f"| {item_name} | {v_str} | {unit} |")
        return "\n".join(lines)

    def _process_followup_text(self, df_slice, age_years=None, gender=None):
        """Convert follow-up DataFrame to markdown text.
        
        Args:
            df_slice: DataFrame of follow-up records
            age_years: Patient age in years (for reference range lookup)
            gender: 'M' or 'F' (for reference range lookup)
        """
        if df_slice.empty or '报告日期' not in df_slice.columns:
            return ""

        med_rows = []
        lab_rows = []

        for _, row in df_slice.iterrows():
            base_time = row.get('报告日期')
            for col, val in row.items():
                if pd.isna(val) or val == '':
                    continue
                if col in {'报告日期', '术后天数'}:
                    continue

                value_parts = self._split_multivalue_parts(val)
                interval_hours = 24.0 / len(value_parts) if len(value_parts) > 1 else 0.0

                for part_idx, part in enumerate(value_parts):
                    event_time = pd.to_datetime(base_time, errors='coerce')
                    if pd.notna(event_time):
                        event_time = event_time + pd.Timedelta(hours=part_idx * interval_hours)

                    if col in self.med_cols_en:
                        col_meta = self.MED_META.get(col, {'str': 'Unknown', 'unit': 'Unknown', 'MEDS_CODE': ''})
                        med_rows.append(
                            {
                                "Time": event_time,
                                "Item": col_meta['str'],
                                "Value": str(part).strip(),
                                "Unit": col_meta['unit'],
                            }
                        )
                    else:
                        lab_meta = self.LAB_META.get(col, {})
                        item_name = lab_meta.get('LONIC', col)
                        desc = self._describe_lab_value(col, part, age_years, gender)
                        lab_rows.append(
                            {
                                "Time": event_time,
                                "Item": item_name,
                                "Value": desc["value"],
                                "Unit": desc["unit"],
                                "Ref_range_lower": desc["low_str"],
                                "Ref_range_upper": desc["high_str"],
                                "Flag": desc["flag"],
                            }
                        )

        texts = []
        all_times = sorted(
            {row["Time"] for row in med_rows + lab_rows if pd.notna(row["Time"])}
        )

        if not all_times and (med_rows or lab_rows):
            all_times = [pd.NaT]

        for current_time in all_times:
            if pd.isna(current_time):
                t_str = "Unknown"
                current_med_rows = [row for row in med_rows if pd.isna(row["Time"])]
                current_lab_rows = [row for row in lab_rows if pd.isna(row["Time"])]
            else:
                t_str = pd.to_datetime(current_time).strftime('%Y-%m-%d %H:%M:%S')
                current_med_rows = [row for row in med_rows if row["Time"] == current_time]
                current_lab_rows = [row for row in lab_rows if row["Time"] == current_time]

            day_text = []
            if current_med_rows:
                header = "| Drug | Dose Value rx | Dose Unit rx |\n| ------ | ------ | ------ |"
                med_lines = [
                    f"| {row['Item']} | {row['Value']} | {row['Unit']} |"
                    for row in current_med_rows
                ]
                day_text.append(f"## Medication [{t_str}]\n" + "\n".join([header] + med_lines))

            if current_lab_rows:
                header = "| Item Name | Value | Unit | Ref_range_lower | Ref_range_upper | Flag |\n| ------ | ------ | ------ | ------ | ------ | ------ |"
                lab_lines = [
                    f"| {row['Item']} | {row['Value']} | {row['Unit']} | {row['Ref_range_lower']} | {row['Ref_range_upper']} | {row['Flag']} |"
                    for row in current_lab_rows
                ]
                day_text.append(f"## Laboratory Test [{t_str}]\n" + "\n".join([header] + lab_lines))

            if day_text:
                texts.append("\n\n".join(day_text))

        return "\n\n".join(texts)

    def free_text_input_process(self, static_features, df_followup, age_years=None, gender=None):
        sections = []

        static_text = self._build_static_text(static_features)
        if static_text:
            sections.append(static_text)

        followup_text = self._process_followup_text(df_followup, age_years, gender)
        if followup_text:
            sections.append(followup_text)

        return "\n\n".join(sections)

    def structed_EHR_input_process(self, static_features, df_followup, surgery_date=None, age_years=None, gender=None):
        measurement_tables = {}

        static_table = self._build_static_table(static_features, surgery_date)
        if static_table is not None:
            measurement_tables['static'] = static_table

        if '报告日期' in df_followup.columns:
            med_cols = [c for c in df_followup.columns if c in self.med_cols_en or c == '报告日期']
            if len(med_cols) > 1:
                med_df = df_followup[med_cols].melt(id_vars=['报告日期'], var_name='Item', value_name='Value')
                med_df = med_df.dropna(subset=['Value'])
                if len(med_df) > 0:
                    med_df = med_df.rename(columns={'报告日期': 'Time'})
                    med_df['Category'] = 'drug_exposure'
                    med_df = self._expand_multivalue_rows(med_df)
                    med_df['Unit'] = med_df['Item'].apply(lambda x: self.MED_META.get(x, {}).get('unit', ''))
                    med_df['Item'] = med_df['Item'].apply(lambda x: self.MED_META.get(x, {}).get('str', x))
                    med_df = med_df[['Time', 'Item', 'Value', 'Unit', 'Category']].sort_values('Time')
                    measurement_tables['medication'] = med_df

            lab_cols = [c for c in df_followup.columns if c not in self.med_cols_en and c != '术后天数']
            if len(lab_cols) > 1:
                lab_df = df_followup[lab_cols].melt(id_vars=['报告日期'], var_name='Item', value_name='Value')
                lab_df = lab_df.dropna(subset=['Value'])
                if len(lab_df) > 0:
                    lab_df = lab_df.rename(columns={'报告日期': 'Time'})
                    lab_df['Category'] = 'measurement'
                    lab_df = self._expand_multivalue_rows(lab_df)
                    desc_df = lab_df.apply(
                        lambda row: pd.Series(
                            self._describe_lab_value(
                                row['Item'],
                                row['Value'],
                                age_years if age_years else 5.0,
                                gender,
                            )
                        ),
                        axis=1,
                    )
                    lab_df['Value'] = desc_df['value']
                    lab_df['Unit'] = desc_df['unit']
                    lab_df['Item'] = lab_df['Item'].apply(lambda x: self.LAB_META.get(x, {}).get('LONIC', x))
                    lab_df = lab_df[['Time', 'Item', 'Value', 'Unit', 'Category']].sort_values('Time')
                    measurement_tables['laboratory'] = lab_df

        dynamic_dfs = []
        if 'medication' in measurement_tables:
            dynamic_dfs.append(measurement_tables['medication'])
        if 'laboratory' in measurement_tables:
            dynamic_dfs.append(measurement_tables['laboratory'])

        if dynamic_dfs:
            final_dynamic = pd.concat(dynamic_dfs, ignore_index=True)
            final_dynamic['Time'] = pd.to_datetime(final_dynamic['Time'], errors='coerce')
            final_dynamic = final_dynamic.sort_values('Time')
        else:
            final_dynamic = pd.DataFrame()

        all_dfs = []
        if 'static' in measurement_tables:
            all_dfs.append(measurement_tables['static'])
        if not final_dynamic.empty:
            all_dfs.append(final_dynamic)

        final_table = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

        if not final_table.empty and 'Time' in final_table.columns:
            sort_keys = pd.to_datetime(final_table['Time'], errors='coerce')
            final_table = final_table.loc[sort_keys.sort_values(na_position='first').index].reset_index(drop=True)

        return final_table

    def structured_text_input_process(self, measurement_table):
        if measurement_table is None or measurement_table.empty:
            return ""

        section_order = [
            ("person", "Patient Static Information"),
            ("drug_exposure", "Medication"),
            ("measurement", "Laboratory Examination"),
        ]
        sections = []

        table = measurement_table.copy()
        if 'Time' in table.columns:
            table['Time'] = pd.to_datetime(table['Time'], errors='coerce')
            table['Time'] = table['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            table['Time'] = table['Time'].fillna('')

        for category, title in section_order:
            if 'Category' not in table.columns:
                continue
            sub_df = table[table['Category'] == category].copy()
            if sub_df.empty:
                continue

            if category == "person":
                lines = [
                    f"## {title}",
                    "| Item | Value | Unit |",
                    "| --- | --- | --- |",
                ]
                for _, row in sub_df.iterrows():
                    lines.append(
                        f"| {str(row.get('Item', '')).strip()} | {str(row.get('Value', '')).strip()} | {str(row.get('Unit', '')).strip()} |"
                    )
            else:
                lines = [
                    f"## {title}",
                    "| Time | Item | Value | Unit |",
                    "| --- | --- | --- | --- |",
                ]
                for _, row in sub_df.iterrows():
                    lines.append(
                        f"| {str(row.get('Time', '')).strip()} | {str(row.get('Item', '')).strip()} | {str(row.get('Value', '')).strip()} | {str(row.get('Unit', '')).strip()} |"
                    )

            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def remaining_text_input_process(self, static_features, df_followup, age_years=None, gender=None):
        return self.free_text_input_process(
            static_features=static_features,
            df_followup=df_followup,
            age_years=age_years,
            gender=gender,
        )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load data
        fname = sample['fname']
        fname_key = sample['fname_key']
        prediction_point = sample['prediction_point']
        cutoff_day, label_prefix, readable_point = self.PREDICTION_POINTS[prediction_point]
        
        # Load follow-up data
        df_followup = self._load_followup_data(sample)
        
        # --- Determine Surgery Date (Day 0) ---
        surgery_date = None
        if '报告日期' in df_followup.columns and '术后天数' in df_followup.columns:
            # Try to find row with 术后天数 == 0
            day0_rows = df_followup[df_followup['术后天数'] == 0]
            if len(day0_rows) > 0:
                surgery_date = day0_rows.iloc[0]['报告日期']
            else:
                # If no Day 0, calculate from the first available record
                # surgery_date = report_date - postoperative_days
                if len(df_followup) > 0:
                    first_row = df_followup.iloc[0]
                    try:
                        report_date = pd.to_datetime(first_row['报告日期'])
                        post_days = float(first_row['术后天数'])
                        if pd.notna(report_date) and pd.notna(post_days):
                            surgery_date = report_date - pd.Timedelta(days=post_days)
                    except:
                        pass
                        
        if surgery_date is None and '报告日期' in df_followup.columns and len(df_followup) > 0:
             # Fallback: use first report date if no postoperative days info
             surgery_date = df_followup.iloc[0]['报告日期']
             
        # Format surgery_date as YYYY-MM-DD
        if surgery_date is not None:
            try:
                surgery_date = pd.to_datetime(surgery_date)
            except:
                surgery_date = None
        
        # Load static features
        static_features = self._get_static_features(fname_key)
        
        # Load pathology (placeholder)
        pathology_data = self._get_pathology_data(fname_key)
        
        # Get patient age and gender for reference range lookup
        age_years = 5.0  # Default age
        gender = None
        if fname_key in self.patient_info_map:
            patient_info = self.patient_info_map[fname_key]
            # Get gender
            recipient_gender = patient_info.get('recipient_gender')
            if pd.notna(recipient_gender):
                gender = 'M' if str(recipient_gender).upper() in ['M', 'MALE', '男'] else 'F'
            # Get age from date_of_birth
            dob = patient_info.get('date_of_birth')
            if pd.notna(dob):
                try:
                    dob = pd.to_datetime(dob)
                    # Use current or first report date to estimate age
                    if '报告日期' in df_followup.columns and len(df_followup) > 0:
                        report_date = df_followup['报告日期'].iloc[0]
                        if pd.notna(report_date):
                            age_years = (report_date - dob).days / 365.25
                except:
                    pass
        
        # PREPARE TARGETS
        if self.task_mode == 'multi_label' or self.task_mode == 'multi_task':
            # labels_matrix: shape [num_points, num_metrics]
            labels_matrix = torch.full((len(self.ALL_POINTS), len(self.ALL_METRICS)), -100, dtype=torch.float32)
            
            if sample['fname_key'] in self.labels_df.index:
                patient_labels = self.labels_df.loc[sample['fname_key']]
                for p_idx, p_key in enumerate(self.ALL_POINTS):
                    _, prefix, _ = self.PREDICTION_POINTS[p_key]
                    for m_idx, met in enumerate(self.ALL_METRICS):
                        col_name = f"{prefix}_{met}"
                        if col_name in patient_labels and pd.notna(patient_labels[col_name]):
                            labels_matrix[p_idx, m_idx] = float(patient_labels[col_name])
            
            output_label = labels_matrix
            task_info = deepcopy(self.task_schema["multi_label_prediction"])
            instruction = task_info["instruction_template"].format(
                prediction_point=f"{readable_point} post-transplant",
            )
            task_info.update(
                {
                    "task": "multi_label",
                    "prediction_point": readable_point,
                    "label_window": "all_windows",
                    "window": "all_windows",
                    "instruction": instruction,
                }
            )
        else:
            metric = sample['metric']
            label_val = sample['label_val']
            # Translate metric to English for instruction
            metric_en = self.ZH_TO_EN.get(metric, metric)

            task_info = deepcopy(self.task_schema["single_metric_prediction"])
            instruction = task_info["instruction_template"].format(
                prediction_point=f"{readable_point} post-transplant",
                metric=metric_en,
                label_window=label_prefix,
            )
            instruction += "\nAnswer with 0 for Normal, or 1 for Abnormal."

            output_label = str(label_val)
            task_info.update(
                {
                    "task": metric,
                    "label": "Abnormal" if label_val == 1 else "Normal",
                    "prediction_point": readable_point,
                    "label_window": label_prefix,
                    "window": label_prefix,
                    "instruction": instruction,
                }
            )
        if self.table_mode in {'table_only', 'table_plus_rest_text'}:
            final_table = self.structed_EHR_input_process(
                static_features=static_features,
                df_followup=df_followup,
                surgery_date=surgery_date,
                age_years=age_years,
                gender=gender,
            )
            input_text = self.structured_text_input_process(final_table)
        else:
            input_text = self.free_text_input_process(
                static_features=static_features,
                df_followup=df_followup,
                age_years=age_years,
                gender=gender,
            )
            final_table = None
        

        output_sample = {
            "idx": idx,
            "instruction": instruction,
            "input": input_text,
            "output": str(output_label),
            "task_info": task_info,
        }
        
        output_sample["candidates"] = ["0", "1"]
        
        if self.table_mode in {'table_only', 'table_plus_rest_text'}:
            output_sample["measurement_table"] = final_table
            if self.table_mode == 'table_plus_rest_text':
                output_sample["remaining_text"] = self.remaining_text_input_process(
                    static_features=static_features,
                    df_followup=df_followup,
                    age_years=age_years,
                    gender=gender,
                )
            
        return output_sample

    def _expand_multivalue_rows(self, df):
        """
        Split rows where Value contains multiple meaningful parts (e.g., '100-200' or '阴性-<400').
        Distribute them evenly across the day.
        """
        if df.empty or 'Value' not in df.columns:
            return df
            
        new_rows = []
        has_changes = False

        for idx in range(len(df)):
            row = df.iloc[idx]
            parts = self._split_multivalue_parts(row['Value'])

            if len(parts) >= 2:
                has_changes = True
                base_time = pd.to_datetime(row['Time'], errors='coerce')
                interval_hours = 24.0 / len(parts)

                for i, part in enumerate(parts):
                    new_row_dict = row.to_dict()
                    new_row_dict['Value'] = str(part).strip()

                    if pd.notna(base_time):
                        delta = pd.Timedelta(hours=i * interval_hours)
                        new_row_dict['Time'] = base_time + delta
                    else:
                        new_row_dict['Time'] = row['Time']

                    new_rows.append(new_row_dict)
            else:
                new_rows.append(row.to_dict())
                
        if has_changes:
            return pd.DataFrame(new_rows)
        return df

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def table_to_markdown(measurement_table):
        if measurement_table is None or measurement_table.empty:
            return ""

        sections = []
        table = measurement_table.copy()
        if 'Time' in table.columns:
            table['Time'] = pd.to_datetime(table['Time'], errors='coerce')
            table['Time'] = table['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            table['Time'] = table['Time'].fillna('')

        for category, title in [
            ("person", "Patient Static Information"),
            ("drug_exposure", "Medication"),
            ("measurement", "Laboratory Examination"),
        ]:
            if 'Category' not in table.columns:
                continue
            sub_df = table[table['Category'] == category]
            if sub_df.empty:
                continue

            if category == "person":
                lines = [
                    f"## {title}",
                    "| Item | Value | Unit |",
                    "| --- | --- | --- |",
                ]
                for _, row in sub_df.iterrows():
                    lines.append(
                        f"| {str(row.get('Item', '')).strip()} | {str(row.get('Value', '')).strip()} | {str(row.get('Unit', '')).strip()} |"
                    )
            else:
                lines = [
                    f"## {title}",
                    "| Time | Item | Value | Unit |",
                    "| --- | --- | --- | --- |",
                ]
                for _, row in sub_df.iterrows():
                    lines.append(
                        f"| {str(row.get('Time', '')).strip()} | {str(row.get('Item', '')).strip()} | {str(row.get('Value', '')).strip()} | {str(row.get('Unit', '')).strip()} |"
                    )

            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    @staticmethod
    def build_input_text(sample, include_unstructured_text=True, unstructured_key="input"):
        sections = []
        table_text = RenjiDataset.table_to_markdown(sample.get("measurement_table"))
        if table_text:
            sections.append(table_text)
        if include_unstructured_text:
            unstructured_text = str(sample.get(unstructured_key, "") or "").strip()
            if unstructured_text and unstructured_text != table_text:
                sections.append(unstructured_text)
        return "\n\n".join(sections)


__all__ = ["RenjiDataset"]


if __name__ == "__main__":
    # Test execution
    root_dir = "/home/ma-user/sfs_turbo/sai6/zkwan/Renji"
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "renji")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Testing Tabular Mode ===")
    dataset_tab = RenjiDataset(
        root_dir=root_dir, 
        split='test', 
        table_mode='table_only',
        shuffle=False
    )
    print(f"Tabular Dataset size: {len(dataset_tab)}")
    sample_tab = dataset_tab[0]
    
    if 'measurement_table' in sample_tab:
        df = sample_tab['measurement_table']
        if df is not None and not df.empty:
            output_csv = os.path.join(out_dir, "renji_sample_tabular.csv")
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"Successfully saved tabular sample to {output_csv}")
    
    print("\n=== Testing Text Mode ===")
    # For text mode we can use task_mode='single' since return_table=False
    dataset_txt = RenjiDataset(
        root_dir=root_dir, 
        split='test', 
        table_mode='text_only',
        shuffle=False,
        task_mode='single'
    )
    sample_txt = dataset_txt[0]
    
    output_txt = os.path.join(out_dir, "renji_sample_text.txt")
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("=== INPUT ===\n")
        f.write(str(sample_txt.get('input', '')) + "\n\n")
        f.write("=== INSTRUCTION ===\n")
        f.write(str(sample_txt.get('instruction', '')) + "\n\n")
        f.write("=== TASK INFO ===\n")
        f.write(str(sample_txt.get('task_info', '')) + "\n")
    print(f"Successfully saved text sample to {output_txt}")
