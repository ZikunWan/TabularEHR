"""
Prepare Renji dataset in MEDS schema format for CLMBR model.

- Loads follow-up Excel files directly with Chinese column names
- Uses mapping.json to convert column names to LOINC/RxNorm codes
- Handles multi-value cells (separated by '-')
- No try-except - errors should be visible
"""

import json
import os
import pickle
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm



# Paths
# Adjust these paths to your local environment
DATA_DIR = f"/home/ma-user/sfs_turbo/sai6/zkwan/Renji"
FOLLOWUP_DIR = Path(f"{DATA_DIR}/随访记录_labeled_v80")
LABELS_FILE = f"{DATA_DIR}/labels.csv"
BIRTHDATE_FILE = Path(f"{DATA_DIR}/患儿基本信息总表251023_含免疫事件.xlsx")
MAPPING_FILE = f"/home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/models/clmbr/mapping.json"
OUTPUT_DIR = Path(f"/home/ma-user/sfs_turbo/sai6/zkwan/Renji/meds_data")
INDEX_DIR = Path(f"/home/ma-user/sfs_turbo/sai6/zkwan/Renji/index")

TRAIN_INDEX = os.path.join(INDEX_DIR, "train_renji.json")
TEST_INDEX = os.path.join(INDEX_DIR, "test_renji.json")

EXCLUDED_COLS = {
    # Metadata columns (not measurements)
    '术后天数',
    # Excluded from training
    'CMV-DNA', 'CMV_DNA',
    'EBV-DNA', 'EBV_DNA', 
    'HBsAg', 'HBsAb', 'HBeAg', 'HBeAb', 'HBcAb',
}


def load_mapping():
    """Load column to LOINC/RxNorm mapping."""
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return mapping.get("mappable", {})


def load_patient_demographics():
    """
    Load birthdates and gender from Excel file.
    Returns: dict {transplant_id_stripped (str): {'birth_date': datetime, 'gender': str (code)}}
    """
    if not os.path.exists(BIRTHDATE_FILE):
        print(f"Warning: Demographics file not found at {BIRTHDATE_FILE}")
        return {}
        
    print(f"Loading demographics from {BIRTHDATE_FILE}...")
    try:
        df = pd.read_excel(BIRTHDATE_FILE, usecols=['transplant_id', 'date_of_birth', 'recipient_gender'])
        
        # Filter rows with valid dates
        df = df.dropna(subset=['date_of_birth'])
        
        # Create map
        demographics_map = {}
        for _, row in df.iterrows():
            try:
                # Use transplant_id as key, remove _1 suffix
                t_id = str(row['transplant_id']).strip()
                if t_id.endswith('_1'):
                    t_id = t_id[:-2]
                
                bdate = row['date_of_birth']
                gender_raw = row.get('recipient_gender')
                
                gender_code = None
                if gender_raw == '男':
                    gender_code = 'Gender/M'
                elif gender_raw == '女':
                    gender_code = 'Gender/F'
                
                if isinstance(bdate, (datetime, pd.Timestamp)):
                    demographics_map[t_id] = {
                        'birth_date': bdate,
                        'gender': gender_code
                    }
            except (ValueError, TypeError):
                continue
                
        return demographics_map
    except Exception as e:
        print(f"Error loading demographics: {e}")
        return {}


def parse_numeric_value(val_str):
    """
    Parse a value string to extract numeric value.
    
    Handles:
    - Simple numbers: "45.5" -> 45.5
    - Prefixed numbers: ">100", "↑45.5" -> 100, 45.5
    - Multi-value cells: "2.5-3.0" -> average to 2.75 (or take first value)
    """
    if pd.isna(val_str):
        return None
    
    val_str = str(val_str).strip()
    if val_str == '' or val_str == '-':
        return None
    
    # Check for multi-value cells separated by '-' that are NOT ranges
    # Pattern: "value1-value2" where both are numbers
    # BUT: "2.5-3.0" could be a range, while "阴性" is text
    
    # First, remove common prefixes/suffixes
    prefixes = ['>', '<', '≥', '≤']
    cleaned = val_str
    for prefix in prefixes:
        cleaned = cleaned.replace(prefix, '')
    cleaned = cleaned.strip()
    
    if cleaned == '':
        return None
    
    # Try to parse as a single number
    try:
        return float(cleaned)
    except ValueError:
        pass
    
    # Check if it contains '-' (could be multi-value like "1.5-2.0-3.5" or range)
    # Split and try to parse each part after removing prefixes
    if '-' in cleaned and not cleaned.startswith('-'):
        parts = cleaned.split('-')
        valid_numbers = []
        for part in parts:
            part = part.strip()
            # Remove any prefix from this part
            for prefix in prefixes:
                part = part.replace(prefix, '')
            part = part.strip()
            if part:
                try:
                    valid_numbers.append(float(part))
                except ValueError:
                    pass
        
        if valid_numbers:
            # Return average of all valid numbers
            return sum(valid_numbers) / len(valid_numbers)
    
    # Cannot parse as numeric
    return None


def process_patient_file(filepath, mapping, patient_id):
    """Process a single patient follow-up file into MEDS format."""
    # Load file based on extension
    ext = filepath.suffix.lower()
    if ext == '.csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath, engine='openpyxl')
    
    if df.empty:
        return None
    
    # Sort by report date
    if '报告日期' in df.columns:
        df['报告日期'] = pd.to_datetime(df['报告日期'], errors='coerce')
        df = df.sort_values('报告日期').reset_index(drop=True)
    
    events = []
    
    # Process each row as a time point
    for row_idx, row in df.iterrows():
        # Get time from 报告日期
        event_time = row['报告日期']
        if pd.isna(event_time):
            # Skip rows without valid date
            continue
        measurements = []
        
        # Process each column - use column name directly to lookup in mapping
        for col_name, val in row.items():
            # Skip excluded columns
            if col_name in EXCLUDED_COLS:
                continue
            if pd.isna(val):
                continue
            
            # Check if column is in mapping (mapping uses Chinese column names as keys)
            if col_name not in mapping:
                continue
            
            code_info = mapping[col_name]
            code = code_info.get("code") if isinstance(code_info, dict) else code_info
            code_type = code_info.get("type", "numeric") if isinstance(code_info, dict) else "numeric"
            
            if code_type == "numeric":
                numeric_val = parse_numeric_value(val)
                if numeric_val is not None:
                    measurements.append({
                        'code': code,
                        'numeric_value': numeric_val
                    })
            else:
                # Text type
                val_str = str(val).strip()
                if val_str and val_str != '-':
                    measurements.append({
                        'code': code,
                        'text_value': val_str
                    })
        
        if measurements:
            events.append({
                'time': event_time.to_pydatetime() if hasattr(event_time, 'to_pydatetime') else event_time,
                'measurements': measurements
            })
    
    if not events:
        return None
    
    return {
        'patient_id': patient_id,
        'events': events
    }


def load_labels():
    """Load labels and create patient index."""
    df = pd.read_csv(LABELS_FILE, encoding='utf-8-sig')
    
    # Create dict: filename -> row
    labels = {}
    if 'filename' in df.columns:
        for _, row in df.iterrows():
            labels[row['filename']] = row.to_dict()
    
    return labels


def load_index(path):
    """Load index file (list of filenames)."""
    if not os.path.exists(path):
        print(f"Warning: Index file not found at {path}")
        return set()
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Normalize filenames if needed (e.g., strip whitespace)
    return set(f.strip() for f in data)


def main():
    print("=== Preparing Renji data in MEDS format ===\n")
    
    # Load mapping
    print("Loading column mapping...")
    mapping = load_mapping()
    print(f"  {len(mapping)} columns mapped to LOINC/RxNorm codes\n")
    
    # Load labels
    print("Loading labels...")
    labels = load_labels()
    print(f"  {len(labels)} patient labels loaded\n")
    
    # Load Demographics
    demographics_map = load_patient_demographics()
    print(f"  {len(demographics_map)} demographics loaded\n")
    
    # Load Split Indices
    print("Loading indices...")
    train_indices = load_index(TRAIN_INDEX)
    test_indices = load_index(TEST_INDEX)
    print(f"  Train: {len(train_indices)}, Test: {len(test_indices)}")
    
    patient_files = list(FOLLOWUP_DIR.glob("*.xlsx")) + list(FOLLOWUP_DIR.glob("*.csv"))
    print(f"Found {len(patient_files)} patient files in {FOLLOWUP_DIR}\n")
    
    # Process each patient
    train_patients = []
    test_patients = []
    skipped = 0
    other_skipped = 0 # Files not in train or test index
    no_birthdate_skipped = 0
    
    for i, filepath in enumerate(tqdm(patient_files, desc="Processing patients")):
        patient_id = i + 1  # Simple numeric ID
        filename = filepath.stem # Filename without extension
        
        # Determine split
        is_train = filename in train_indices
        is_test = filename in test_indices
        
        if not is_train and not is_test:
            # Maybe check with other extensions or fuzzy match? 
            # For now strict match based on index content
            other_skipped += 1
            continue
            
        # Get Birthdate and Gender
        # Use filename as key (matches transplant_id without _1)
        birth_date = None
        gender_code = None
        
        demo_info = demographics_map.get(filename)
        if demo_info:
            birth_date = demo_info.get('birth_date')
            gender_code = demo_info.get('gender')
            
        if birth_date is None:
            # Skip if no birthdate found
            no_birthdate_skipped += 1
            continue
            
        patient = process_patient_file(filepath, mapping, patient_id)
        
        if patient is None:
            skipped += 1
            continue
        
        # Store filename for label matching
        patient['filename'] = filename
        
        # Add birth event
        # Note: The birth event is treated as the first event in the patient's timeline.
        birth_measurements = [{
            'code': 'SNOMED/184099003',
        }]
        
        if gender_code:
            birth_measurements.append({
                'code': gender_code
            })
            
        birth_event = {
            'time': birth_date,
            'measurements': birth_measurements
        }
        
        patient['events'].append(birth_event)
        
        # Sort associated events by time
        patient['events'].sort(key=lambda x: x['time'])
        
        # Add labels if available
        if filename in labels:
            patient['labels'] = labels[filename]
        
        if is_train:
            train_patients.append(patient)
        elif is_test:
            test_patients.append(patient)
    
    print(f"\nProcessed Patients:")
    print(f"  Train: {len(train_patients)}")
    print(f"  Test:  {len(test_patients)}")
    print(f"  Skipped (empty/error): {skipped}")
    print(f"  Skipped (not in index): {other_skipped}")
    print(f"  Skipped (no birthdate): {no_birthdate_skipped}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save Train
    train_output = OUTPUT_DIR / "renji_meds_train.pkl"
    print(f"\nSaving train set to {train_output}...")
    with open(train_output, 'wb') as f:
        pickle.dump(train_patients, f)
        
    # Save Test
    test_output = OUTPUT_DIR / "renji_meds_test.pkl"
    print(f"Saving test set to {test_output}...")
    with open(test_output, 'wb') as f:
        pickle.dump(test_patients, f)
    
    # Also save a small sample as JSON for inspection
    sample_file = OUTPUT_DIR / "sample_patients.json"
    sample = (train_patients[:3] if train_patients else []) + (test_patients[:2] if test_patients else [])
    
    # Convert datetime to string for JSON
    def convert_dates(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert_dates(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_dates(x) for x in obj]
        return obj
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(convert_dates(sample), f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(sample)} sample patients to {sample_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
