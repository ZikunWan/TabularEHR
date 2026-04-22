import os
import sys
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dataset.mimic_dataset import MIMICIV

# Global dataset variable for worker processes
_worker_dataset = None

def _init_worker(root_dir, sample_csv):
    """Initialize the MIMICIV dataset in each worker process."""
    global _worker_dataset
    _worker_dataset = MIMICIV(
        root_dir=root_dir,
        sample_info_path=sample_csv,
        lazy_mode=True,
        shuffle=False,
        return_table=True,
        log=False,
    )

def _get_table_length(idx):
    """Retrieve the table length for a given sample index."""
    try:
        sample = _worker_dataset[idx]
        tables = sample.get('measurement_table')
        if isinstance(tables, pd.DataFrame):
            return len(tables)
        elif isinstance(tables, dict):
            # Sum up the lengths of all individual DataFrames in the dictionary
            length = 0
            for k, v in tables.items():
                if isinstance(v, pd.DataFrame):
                    length += len(v)
            return length
        else:
            return 0
    except Exception as e:
        # In case of any error processing a particular sample, default to 0
        return 0

def main():
    parser = argparse.ArgumentParser(description="Calculate and append table lengths to the sample CSV.")
    parser.add_argument("--root_dir", type=str, default="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular",
                        help="Root directory for MIMIC-IV tabular data")
    parser.add_argument("--sample_csv", type=str, default="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/task_index/all/contrastive_learning.csv",
                        help="Path to the sample info CSV to update")
    parser.add_argument("--num_workers", type=int, default=64, help="Number of concurrent multiprocessing workers")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()

    # Back up the original CSV file before modifying it
    backup_csv = args.sample_csv + ".bak"
    if not os.path.exists(backup_csv):
        print(f"Backing up original CSV to {backup_csv}")
        import shutil
        shutil.copy2(args.sample_csv, backup_csv)

    print(f"Loading CSV: {args.sample_csv}")
    df = pd.read_csv(args.sample_csv)
    
    num_samples = len(df)
    if args.max_samples:
        num_samples = min(num_samples, args.max_samples)
        df = df.iloc[:num_samples].copy()
        
    print(f"Processing {num_samples} samples using {args.num_workers} workers...")

    lengths = []
    
    with Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(args.root_dir, args.sample_csv)
    ) as pool:
        # Map indices to the worker function and display a progress bar
        lengths = list(tqdm(
            pool.imap(_get_table_length, range(num_samples)),
            total=num_samples,
            desc="Calculating Lengths"
        ))
        
    df['table_length'] = lengths
    
    # Save the updated DataFrame back to the CSV
    # Only save if we processed the entire file, otherwise save to a test file
    if args.max_samples:
        test_csv = args.sample_csv.replace(".csv", "_test.csv")
        df.to_csv(test_csv, index=False)
        print(f"Saved subset test results to {test_csv}")
    else:
        df.to_csv(args.sample_csv, index=False)
        print(f"Successfully updated and saved {args.sample_csv}")

if __name__ == "__main__":
    main()
