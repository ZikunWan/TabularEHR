"""
==============================================================================
STEP 1 of 5: Build eICU Patient Cohorts
==============================================================================

Purpose:
    Filter ICU stays from patient.csv based on age, LOS, and other criteria.
    Label readmissions and standardize fields for compatibility.

Input:
    - config.yaml: Configuration parameters
    - data/eicu-crd/2.0/patient.csv: Raw eICU patient data

Output:
    - data/eicu-crd/processed/cohorts.csv: Filtered ICU stays with standardized fields

Usage:
    python 1_build_cohorts.py --config config.yaml

Next Step:
    Run 2_prepare_tasks.py to generate task labels

==============================================================================
"""
import os
import argparse
import pandas as pd
import yaml
from pathlib import Path


def build_cohorts(config):
    """
    Build patient cohorts from eICU patient.csv
    
    Filters:
    - Age range
    - Minimum LOS (obs_size + gap_size)
    - First ICU admission only (optional)
    - Readmission labeling
    """
    print("=" * 80)
    print("eICU Cohort Building")
    print("=" * 80)
    
    # Load config
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    min_age = config.get('min_age', 18)
    max_age = config.get('max_age', 89)
    obs_size = config.get('obs_size', 12)  # hours
    gap_size = config.get('gap_size', 12)
    first_icu = config.get('first_icu_only', True)
    
    min_los_hours = obs_size + gap_size
    
    print(f"\nConfiguration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Age range: {min_age} - {max_age}")
    print(f"  Minimum LOS: {min_los_hours} hours")
    print(f"  First ICU only: {first_icu}")
    
    # Load patient.csv
    patient_path = os.path.join(data_dir, "patient.csv")
    if not os.path.exists(patient_path):
        patient_path = os.path.join(data_dir, "patient.csv.gz")
    
    print(f"\nLoading patient data from {patient_path}...")
    icustays = pd.read_csv(patient_path, low_memory=False)
    print(f"Loaded {len(icustays)} ICU stays")
    
    # Make compatible with GenHPF format
    print("\nProcessing patient data...")
    
    # Calculate LOS in days
    icustays['LOS'] = icustays['unitdischargeoffset'] / 60 / 24
    
    # Process age
    icustays = icustays.dropna(subset=['age'])
    icustays['AGE'] = icustays['age'].replace('> 89', 300).astype(str)
    icustays['AGE'] = pd.to_numeric(icustays['AGE'], errors='coerce')
    icustays = icustays.dropna(subset=['AGE'])
    
    print(f"After age processing: {len(icustays)} stays")
    
    # Standardize time fields
    icustays['INTIME'] = 0
    icustays['OUTTIME'] = icustays['unitdischargeoffset']
    icustays['DISCHTIME'] = icustays['hospitaldischargeoffset']
    
    # Standardize discharge status
    icustays['IN_ICU_MORTALITY'] = icustays['unitdischargestatus'] == 'Expired'
    
    # Map discharge location
    disch_map = {
        'Home': 'Home',
        'Death': 'Death',
        'Nursing Home': 'Other',
        'Other': 'Other',
        'Other External': 'Other',
        'Other Hospital': 'Other',
        'Rehabilitation': 'Rehabilitation',
        'Skilled Nursing Facility': 'Skilled Nursing Facility',
    }
    icustays['HOS_DISCHARGE_LOCATION'] = icustays['hospitaldischargelocation'].map(disch_map)
    icustays['HOS_DISCHARGE_LOCATION'] = icustays['HOS_DISCHARGE_LOCATION'].fillna('Other')
    
    # Apply filters
    print("\nApplying filters...")
    
    # Filter by LOS
    los_filter = icustays['LOS'] >= (min_los_hours / 24)
    print(f"  LOS >= {min_los_hours/24:.2f} days: {los_filter.sum()} / {len(icustays)}")
    icustays = icustays[los_filter]
    
    # Filter by age
    age_filter = (icustays['AGE'] >= min_age) & (icustays['AGE'] <= max_age)
    print(f"  Age {min_age}-{max_age}: {age_filter.sum()} / {len(icustays)}")
    icustays = icustays[age_filter]
    
    # Readmission labeling
    print("\nLabeling readmissions...")
    icustays = icustays.sort_values(['patienthealthsystemstayid', 'patientunitstayid']).reset_index(drop=True)
    icustays['readmission'] = 1
    
    # Match GenHPF: last ICU admission per hospital stay gets readmission=0
    # (all preceding ICU stays within the same hospital stay are labeled as readmission=1)
    last_icu_idx = icustays.groupby('patienthealthsystemstayid')['unitvisitnumber'].idxmax()
    icustays.loc[last_icu_idx, 'readmission'] = 0
    first_icu_idx = icustays.groupby('patienthealthsystemstayid')['unitvisitnumber'].idxmin()
    
    readmit_counts = icustays['readmission'].value_counts()
    print(f"  Last ICU (readmission=0): {readmit_counts.get(0, 0)}")
    print(f"  Preceding ICU stays (readmission=1): {readmit_counts.get(1, 0)}")
    
    # Filter for first ICU only
    if first_icu:
        print("\nFiltering for first ICU admission only...")
        icustays = icustays.loc[first_icu_idx].reset_index(drop=True)
        print(f"  Remaining: {len(icustays)} stays")
    
    # Select relevant columns
    cohort_cols = [
        'patientunitstayid',
        'patienthealthsystemstayid',
        'uniquepid',
        'age',
        'AGE',
        'gender',
        'ethnicity',
        'unittype',
        'unitadmitsource',
        'unitvisitnumber',
        'LOS',
        'INTIME',
        'OUTTIME',
        'DISCHTIME',
        'IN_ICU_MORTALITY',
        'HOS_DISCHARGE_LOCATION',
        'readmission'
    ]
    
    # Keep only available columns
    available_cols = [c for c in cohort_cols if c in icustays.columns]
    cohorts = icustays[available_cols].copy()
    
    # Save cohorts
    output_path = os.path.join(output_dir, "cohorts.csv")
    cohorts.to_csv(output_path, index=False)
    print(f"\nSaved {len(cohorts)} cohorts to {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("COHORT SUMMARY")
    print("=" * 80)
    print(f"Total cohorts: {len(cohorts)}")
    print(f"\nAge statistics:")
    print(f"  Mean: {cohorts['AGE'].mean():.1f}")
    print(f"  Median: {cohorts['AGE'].median():.1f}")
    print(f"  Range: {cohorts['AGE'].min():.0f} - {cohorts['AGE'].max():.0f}")
    
    print(f"\nGender distribution:")
    print(cohorts['gender'].value_counts())
    
    print(f"\nLOS statistics (days):")
    print(f"  Mean: {cohorts['LOS'].mean():.2f}")
    print(f"  Median: {cohorts['LOS'].median():.2f}")
    print(f"  Range: {cohorts['LOS'].min():.2f} - {cohorts['LOS'].max():.2f}")
    
    print(f"\nICU mortality:")
    print(cohorts['IN_ICU_MORTALITY'].value_counts())
    
    print(f"\nDischarge location:")
    print(cohorts['HOS_DISCHARGE_LOCATION'].value_counts())
    
    print("\n" + "=" * 80)
    
    return cohorts


def main():
    parser = argparse.ArgumentParser(description='Build eICU cohorts')
    parser.add_argument('--config', type=str, default='preprocess/eicu/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line args if provided
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Build cohorts
    cohorts = build_cohorts(config)
    
    print("\n✓ Cohort building completed successfully!")


if __name__ == "__main__":
    main()
