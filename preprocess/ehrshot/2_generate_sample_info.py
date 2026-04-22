import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

# Paths
ASSETS_DIR = "/home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/evaluation/ehrshot-benchmark/EHRSHOT_ASSETS"
SPLITS_PATH = os.path.join(ASSETS_DIR, "splits", "person_id_map.csv")
BENCHMARK_DIR = os.path.join(ASSETS_DIR, "benchmark")
PATIENT_EHR_DIR = "/home/ma-user/sfs_turbo/sai6/zkwan/EHRSHOT/patient_ehr"
OUTPUT_DIR = "/home/ma-user/sfs_turbo/sai6/zkwan/EHRSHOT/index"

# Task names from ehrshot_dataset.py
TASK_NAMES = [
    'guo_los',
    'guo_readmission',
    'guo_icu',
    'lab_anemia',
    'lab_hyperkalemia',
    'lab_hyponatremia',
    'lab_hypoglycemia',
    'lab_thrombocytopenia',
    'new_acutemi',
    'new_celiac',
    'new_hyperlipidemia',
    'new_hypertension',
    'new_lupus',
    'new_pancan'
]

def find_period_range(patient_id, prediction_time):
    """
    Find period_begin and period_end for a given patient and prediction time.
    Returns the row indices that should be included in the context.
    """
    patient_path = os.path.join(PATIENT_EHR_DIR, f"{patient_id}.csv")
    
    if not os.path.exists(patient_path):
        return None, None
    
    try:
        df = pd.read_csv(patient_path)
        df['start'] = pd.to_datetime(df['start'], errors='coerce')
        prediction_time = pd.to_datetime(prediction_time)
        
        # Filter out person table (it's always included separately)
        non_person_df = df[df['omop_table'] != 'person']
        
        if len(non_person_df) == 0:
            return None, None
        
        # Find all records before or at prediction time
        valid_records = non_person_df[non_person_df['start'] <= prediction_time]
        
        if len(valid_records) == 0:
            return None, None
        
        # Get the indices in the original dataframe
        period_begin = non_person_df.index[0]
        period_end = valid_records.index[-1]
        
        return period_begin, period_end
        
    except Exception as e:
        return None, None

def process_single_row(row, split_dict):
    """Process a single row and return sample dict or None"""
    patient_id = row['patient_id']
    prediction_time = row['prediction_time']
    label = row['value']
    task_name = row['task_name']
    
    # Get split
    split = split_dict.get(patient_id, None)
    if split is None:
        return None
    
    # Find period range
    period_begin, period_end = find_period_range(patient_id, prediction_time)
    
    if period_begin is None or period_end is None:
        return None
    
    # Create sample record
    sample = {
        'patient_id': patient_id,
        'task_name': task_name,
        'prediction_time': prediction_time,
        'label': label,
        'period_begin': period_begin,
        'period_end': period_end,
        'split': split
    }
    
    return sample

def process_task_chunk(chunk, split_dict):
    """Process a chunk of rows with multiprocessing"""
    results = []
    for _, row in chunk.iterrows():
        result = process_single_row(row, split_dict)
        if result is not None:
            results.append(result)
    return results

def generate_sample_info(num_processes=None):
    """
    Generate sample info CSV files using multiprocessing.
    
    Args:
        num_processes: Number of processes to use. If None, uses cpu_count()
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"Using {num_processes} processes")
    
    # Load split information
    print("Loading split information...")
    split_df = pd.read_csv(SPLITS_PATH)
    split_dict = dict(zip(split_df['omop_person_id'], split_df['split']))
    
    all_samples = []
    
    # Process each task
    for task_name in TASK_NAMES:
        labeled_path = os.path.join(BENCHMARK_DIR, task_name, "labeled_patients.csv")
        
        if not os.path.exists(labeled_path):
            print(f"Warning: {labeled_path} not found, skipping...")
            continue
        
        print(f"\nProcessing task: {task_name}")
        labeled_df = pd.read_csv(labeled_path)

        labeled_df.drop_duplicates(subset=['patient_id', 'prediction_time', 'value'], inplace=True)
        conflict_mask = labeled_df.duplicated(subset=['patient_id', 'prediction_time'], keep=False)

        labeled_df = labeled_df[~conflict_mask]
        labeled_df['task_name'] = task_name

        # Split dataframe into chunks for parallel processing
        chunk_size = max(1, len(labeled_df) // (num_processes * 4))
        chunks = [labeled_df[i:i+chunk_size] for i in range(0, len(labeled_df), chunk_size)]
        
        print(f"  Processing {len(labeled_df)} samples in {len(chunks)} chunks...")
        
        # Process chunks in parallel
        with Pool(num_processes) as pool:
            process_func = partial(process_task_chunk, split_dict=split_dict)
            results = list(tqdm(
                pool.imap(process_func, chunks),
                total=len(chunks),
                desc=f"  {task_name}"
            ))
        
        # Flatten results
        for chunk_result in results:
            all_samples.extend(chunk_result)
    
    # Create DataFrame
    print("\nCreating sample info dataframes...")
    all_df = pd.DataFrame(all_samples)
    
    # Separate into train, val, and test
    train_df = all_df[all_df['split'] == 'train']
    val_df = all_df[all_df['split'] == 'val']
    test_df = all_df[all_df['split'] == 'test']
    
    # Save to files
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_path = os.path.join(OUTPUT_DIR, "ehrshot_train.csv")
    val_path = os.path.join(OUTPUT_DIR, "ehrshot_val.csv")
    test_path = os.path.join(OUTPUT_DIR, "ehrshot_test.csv")
    all_path = os.path.join(OUTPUT_DIR, "ehrshot_all.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    all_df.to_csv(all_path, index=False)
    
    print(f"\n✓ Sample info files generated successfully!")
    print(f"  Total samples: {len(all_df)}")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"\nFiles saved to:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"  {all_path}")
    
    # Print statistics by task
    print("\nSamples per task:")
    task_stats = all_df.groupby(['task_name', 'split']).size().unstack(fill_value=0)
    print(task_stats)
    
    return all_df

if __name__ == "__main__":
    df = generate_sample_info()
