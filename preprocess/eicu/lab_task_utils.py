"""
eICU Clinical Lab Task Utilities
Implements SOFA-based severity labeling for clinical lab values
Adapted from GenHPF's clinical_task method
"""
import os
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


# SOFA (Sequential Organ Failure Assessment) Criteria
SOFA_THRESHOLDS = {
    'creatinine': {
        'unit': 'mg/dL',
        'thresholds': [1.2, 2.0, 3.5, 5.0],  # Boundaries for 0-4 scale
        'labels': {
            0: '<1.2 (Normal)',
            1: '1.2-2.0 (Mild)',
            2: '2.0-3.5 (Moderate)',
            3: '3.5-5.0 (Severe)',
            4: '≥5.0 (Very Severe)'
        }
    },
    'bilirubin': {
        'unit': 'mg/dL',
        'thresholds': [1.2, 2.0, 6.0, 12.0],
        'labels': {
            0: '<1.2 (Normal)',
            1: '1.2-2.0 (Mild)',
            2: '2.0-6.0 (Moderate)',
            3: '6.0-12.0 (Severe)',
            4: '≥12.0 (Very Severe)'
        }
    },
    'platelets': {
        'unit': '×10³/μL',
        'thresholds': [20, 50, 100, 150],  # Note: reversed (lower is worse)
        'reversed': True,  # Lower values = higher severity
        'labels': {
            0: '≥150 (Normal)',
            1: '100-150 (Mild)',
            2: '50-100 (Moderate)',
            3: '20-50 (Severe)',
            4: '<20 (Very Severe)'
        }
    },
    'wbc': {
        'unit': '×10³/μL',
        'thresholds': [4, 12],  # Only 3 categories
        'labels': {
            0: '<4 (Low)',
            1: '4-12 (Normal)',
            2: '>12 (High)'
        }
    }
}

# Lab name mappings (eICU labname -> standard name)
LAB_NAME_MAPPING = {
    'creatinine': ['creatinine', 'creat'],
    'total bilirubin': ['total bilirubin', 'bilirubin', 't bili', 'tbil'],
    'platelets x 1000': ['platelets x 1000', 'platelets', 'platelet count', 'plt'],
    'WBC x 1000': ['WBC x 1000', 'wbc', 'white blood cell', 'leukocytes']
}


def identify_dialysis_patients(data_dir, cohorts, obs_hours, pred_hours):
    """
    Identify patients who received dialysis before the prediction window
    (Should be excluded from creatinine task)
    
    Args:
        data_dir: Path to eICU raw data directory
        cohorts: DataFrame with cohort information
        obs_hours: Observation window in hours
        pred_hours: Prediction window in hours
    
    Returns:
        set: Patient IDs (uniquepid) who received dialysis
    """
    print("\n  Identifying dialysis patients...")
    
    # Load intakeOutput table
    io_path = os.path.join(data_dir, "intakeOutput.csv")
    if not os.path.exists(io_path):
        io_path = os.path.join(data_dir, "intakeOutput.csv.gz")
    
    if not os.path.exists(io_path):
        print("    Warning: intakeOutput.csv not found, skipping dialysis filtering")
        return set()
    
    print(f"    Reading {os.path.basename(io_path)}...")
    io_df = pd.read_csv(io_path, low_memory=False)
    
    # Filter for dialysis records (dialysistotal != 0)
    dialysis_df = io_df[io_df['dialysistotal'] != 0].copy()
    
    if len(dialysis_df) == 0:
        print("    No dialysis records found")
        return set()
    
    # Merge with cohorts to get patient IDs
    dialysis_df = dialysis_df.merge(
        cohorts[['patientunitstayid', 'uniquepid']],
        on='patientunitstayid',
        how='inner'
    )
    
    # Find patients with multiple hospital stays
    patient_stay_counts = cohorts.groupby('uniquepid')['patienthealthsystemstayid'].nunique()
    multi_stay_patients = patient_stay_counts[patient_stay_counts > 1].index
    
    # Filter dialysis records before prediction window ends
    max_time = (obs_hours + pred_hours) * 60  # Convert to minutes
    dialysis_df = dialysis_df[dialysis_df['intakeoutputoffset'] <= max_time]
    
    # Get patients who had dialysis and have multiple stays
    dialysis_patients = set(dialysis_df[dialysis_df['uniquepid'].isin(multi_stay_patients)]['uniquepid'])
    
    print(f"    Found {len(dialysis_patients)} multi-stay patients with dialysis")
    
    return dialysis_patients


def get_lab_values_in_window(data_dir, cohorts, lab_name, obs_hours, gap_hours, pred_hours):
    """
    Get lab values within the prediction window for each ICU stay
    
    Args:
        data_dir: Path to eICU raw data directory
        cohorts: DataFrame with cohort information
        lab_name: Name of the lab test (must be in LAB_NAME_MAPPING)
        obs_hours: Observation window in hours
        gap_hours: Gap between observation and prediction
        pred_hours: Prediction window in hours
    
    Returns:
        DataFrame with columns: patientunitstayid, avg_value
    """
    # Load lab table
    lab_path = os.path.join(data_dir, "lab.csv")
    if not os.path.exists(lab_path):
        lab_path = os.path.join(data_dir, "lab.csv.gz")
    
    if not os.path.exists(lab_path):
        raise FileNotFoundError(f"Lab data not found: {lab_path}")
    
    print(f"\n  Loading lab data for {lab_name}...")
    
    # Get possible lab names
    possible_names = LAB_NAME_MAPPING.get(lab_name, [lab_name])
    possible_names_lower = [n.lower() for n in possible_names]
    
    # Get cohort ICU stay IDs for faster filtering
    cohort_icustay_ids = set(cohorts['patientunitstayid'])
    
    # Match GenHPF:
    # prediction window is [obs + gap, obs + pred] in minutes
    start_time = (obs_hours + gap_hours) * 60
    end_time = (obs_hours + pred_hours) * 60
    
    # Read lab data with progress bar
    print(f"    Reading {os.path.basename(lab_path)}...")
    
    # Use chunked reading for large files
    chunk_size = 1000000  # 1M rows per chunk
    chunks = []
    
    with tqdm(desc="    Processing lab data", unit=" rows", unit_scale=True) as pbar:
        for chunk in pd.read_csv(lab_path, low_memory=False, chunksize=chunk_size):
            # Filter for relevant lab tests
            chunk['labname_lower'] = chunk['labname'].str.lower()
            mask = chunk['labname_lower'].isin(possible_names_lower)
            chunk = chunk[mask]
            
            if len(chunk) > 0:
                # Filter for ICU stays in cohorts
                chunk = chunk[chunk['patientunitstayid'].isin(cohort_icustay_ids)]
                
                if len(chunk) > 0:
                    # Filter for prediction window (left-inclusive, right-inclusive)
                    in_window = (chunk['labresultoffset'] >= start_time) & (chunk['labresultoffset'] <= end_time)
                    chunk = chunk[in_window]
                    
                    if len(chunk) > 0:
                        chunks.append(chunk[['patientunitstayid', 'labresult']])
            
            pbar.update(len(chunk))
    
    if not chunks:
        print(f"    No {lab_name} measurements found in prediction window")
        return pd.DataFrame(columns=['patientunitstayid', 'avg_value'])
    
    # Combine all chunks
    lab_df = pd.concat(chunks, ignore_index=True)
    print(f"    Found {len(lab_df)} {lab_name} measurements in prediction window")
    
    # Convert labresult to numeric
    print(f"    Converting values to numeric...")
    lab_df['labresult'] = pd.to_numeric(lab_df['labresult'], errors='coerce')
    lab_df = lab_df.dropna(subset=['labresult'])
    
    if len(lab_df) == 0:
        print(f"    No valid numeric values found")
        return pd.DataFrame(columns=['patientunitstayid', 'avg_value'])
    
    # Calculate average value per ICU stay
    print(f"    Computing averages per ICU stay...")
    avg_values = lab_df.groupby('patientunitstayid')['labresult'].mean().reset_index()
    avg_values.columns = ['patientunitstayid', 'avg_value']
    
    print(f"    ✓ Computed averages for {len(avg_values)} ICU stays")
    
    return avg_values


def apply_sofa_thresholds(avg_values, task_name):
    """
    Apply SOFA thresholds to convert lab values to severity categories
    
    Args:
        avg_values: DataFrame with avg_value column
        task_name: One of 'creatinine', 'bilirubin', 'platelets', 'wbc'
    
    Returns:
        Series with severity category (0-4 for most, 0-2 for wbc)
    """
    if task_name not in SOFA_THRESHOLDS:
        raise ValueError(f"Unknown task: {task_name}")
    
    values = avg_values['avg_value']

    # Use explicit inequalities to match GenHPF boundaries exactly.
    if task_name == 'creatinine':
        labels = np.select(
            [
                values < 1.2,
                (values >= 1.2) & (values < 2.0),
                (values >= 2.0) & (values < 3.5),
                (values >= 3.5) & (values < 5.0),
                values >= 5.0,
            ],
            [0, 1, 2, 3, 4],
            default=np.nan,
        )
    elif task_name == 'bilirubin':
        labels = np.select(
            [
                values < 1.2,
                (values >= 1.2) & (values < 2.0),
                (values >= 2.0) & (values < 6.0),
                (values >= 6.0) & (values < 12.0),
                values >= 12.0,
            ],
            [0, 1, 2, 3, 4],
            default=np.nan,
        )
    elif task_name == 'platelets':
        labels = np.select(
            [
                values >= 150.0,
                (values >= 100.0) & (values < 150.0),
                (values >= 50.0) & (values < 100.0),
                (values >= 20.0) & (values < 50.0),
                values < 20.0,
            ],
            [0, 1, 2, 3, 4],
            default=np.nan,
        )
    elif task_name == 'wbc':
        labels = np.select(
            [
                values < 4.0,
                (values >= 4.0) & (values <= 12.0),
                values > 12.0,
            ],
            [0, 1, 2],
            default=np.nan,
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")

    return pd.Series(labels, index=avg_values.index).astype('Int64')


def process_clinical_lab_task(data_dir, cohorts, task_name, obs_hours=12, gap_hours=12, pred_hours=24):
    """
    Process a clinical lab task and add labels to cohorts
    
    Args:
        data_dir: Path to eICU raw data directory
        cohorts: DataFrame with cohort information
        task_name: One of 'creatinine', 'bilirubin', 'platelets', 'wbc'
        obs_hours: Observation window in hours
        gap_hours: Gap window in hours
        pred_hours: Prediction window in hours
    
    Returns:
        DataFrame with added task column
    """
    print(f"\n{'='*60}")
    print(f"Processing {task_name.upper()} task")
    print(f"{'='*60}")
    
    # Get lab name
    lab_name = {
        'creatinine': 'creatinine',
        'bilirubin': 'total bilirubin',
        'platelets': 'platelets x 1000',
        'wbc': 'WBC x 1000'
    }[task_name]
    
    # Get lab values in prediction window
    avg_values = get_lab_values_in_window(
        data_dir, cohorts, lab_name, obs_hours, gap_hours, pred_hours
    )
    
    if len(avg_values) == 0:
        print(f"  Warning: No {task_name} values found in prediction window")
        cohorts[task_name] = np.nan
        return cohorts
    
    # Apply SOFA thresholds
    print(f"\n  Applying SOFA thresholds...")
    avg_values[task_name] = apply_sofa_thresholds(avg_values, task_name)
    
    # Special handling for creatinine: exclude dialysis patients
    if task_name == 'creatinine':
        dialysis_patients = identify_dialysis_patients(
            data_dir, cohorts, obs_hours, pred_hours
        )
        
        if len(dialysis_patients) > 0:
            # Merge patient IDs
            avg_values = avg_values.merge(
                cohorts[['patientunitstayid', 'uniquepid']],
                on='patientunitstayid',
                how='left'
            )
            
            # Exclude dialysis patients
            before_count = len(avg_values)
            avg_values = avg_values[~avg_values['uniquepid'].isin(dialysis_patients)]
            after_count = len(avg_values)
            
            print(f"  Excluded {before_count - after_count} ICU stays from dialysis patients")
            
            avg_values = avg_values[['patientunitstayid', task_name]]
    
    # Merge with cohorts
    cohorts = cohorts.merge(
        avg_values[['patientunitstayid', task_name]],
        on='patientunitstayid',
        how='left'
    )
    
    # Print distribution
    label_dist = cohorts[task_name].value_counts().sort_index()
    config = SOFA_THRESHOLDS[task_name]
    
    print(f"\n  Label distribution:")
    for label, count in label_dist.items():
        if not pd.isna(label):
            label_int = int(label)
            label_name = config['labels'].get(label_int, str(label_int))
            pct = count / len(cohorts) * 100
            print(f"    {label_int}: {label_name:30s} - {count:5d} ({pct:5.1f}%)")
    
    missing_count = cohorts[task_name].isna().sum()
    if missing_count > 0:
        pct = missing_count / len(cohorts) * 100
        print(f"    NaN: No data in prediction window - {missing_count:5d} ({pct:5.1f}%)")
    
    return cohorts


if __name__ == "__main__":
    # Test individual task processing
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python lab_task_utils.py <data_dir> <cohorts_csv> <task_name>")
        print("  task_name: creatinine, bilirubin, platelets, or wbc")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    cohorts_path = sys.argv[2]
    task_name = sys.argv[3]
    
    # Load cohorts
    print(f"Loading cohorts from {cohorts_path}...")
    cohorts = pd.read_csv(cohorts_path)
    print(f"  Loaded {len(cohorts)} cohorts")
    
    # Process task
    cohorts = process_clinical_lab_task(data_dir, cohorts, task_name)
    
    # Save result
    output_path = f"cohorts_with_{task_name}.csv"
    cohorts.to_csv(output_path, index=False)
    print(f"\n✓ Saved result to {output_path}")
