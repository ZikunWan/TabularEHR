"""
Generate train/test splits for Renji dataset using labels.csv.
Uses multilabel stratified splitting based on label distribution.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


DEFAULT_DATA_DIR = "/data/EHR_data_public/Renji"
DEFAULT_SAVE_DIR = "/data/EHR_data_public/Renji/index"

LABEL_WINDOWS = ["2w-1m", "2m-6m", "7m-12m", "13m-14m", "15m-24m", "2y+"]

# Target metrics
TARGET_METRICS = [
    "ALB",
    "ALP",
    "ALT",
    "AST",
    "CMV-DNA",
    "CR",
    "DB",
    "EBV-DNA",
    "HBV-DNA",
    "HB",
    "INR",
    "N(%)",
    "PLT",
    "PT",
    "TB",
    "TP",
    "WBC",
    "γ-GT",
    "他克莫司浓度",
    "尿酸",
    "总胆固醇",
    "淋巴细胞绝对值",
    "环孢素峰浓度",
    "环孢素谷浓度",
    "甘油三脂",
    "胆汁酸",
    "血糖",
]

DEV_SAMPLE_SIZE = None  # Set to integer for quick testing


def get_label_fingerprints(labels_df):
    """
    Generate label fingerprints for stratification.
    Use all individual [window]_[metric] columns as the fingerprint.
    """
    target_cols = []
    
    # Identify all valid target columns that exist in the dataframe
    for m in TARGET_METRICS:
        for w in LABEL_WINDOWS:
            col_name = f"{w}_{m}"
            if col_name in labels_df.columns:
                target_cols.append(col_name)
    
    print(f"Using {len(target_cols)} individual label columns for stratification.")
    
    # Extract the binary matrix
    # Fill NaN with 0 for stratification purposes (treating missing as negative/ignore)
    # in this context, we just want to balance the positive labels we DO have.
    y = labels_df[target_cols].fillna(0).astype(int).values
    
    return y, target_cols


def main():
    parser = argparse.ArgumentParser(description="Generate Renji dataset train/test splits.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, 
                        help="Directory containing labels.csv")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR, 
                        help="Directory to save split JSON files.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test set ratio (default: 0.2)")
    parser.add_argument("--dev_sample", type=int, default=DEV_SAMPLE_SIZE, 
                        help="Size of dev set for testing (optional).")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility")
    
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created directory: {args.save_dir}")

    # 1. Load labels.csv
    labels_path = os.path.join(args.data_dir, "labels.csv")
    if not os.path.exists(labels_path):
        print(f"Error: labels.csv not found at {labels_path}")
        return
    
    labels_df = pd.read_csv(labels_path, encoding='utf-8-sig')
    print(f"Loaded labels.csv: {labels_df.shape[0]} patients, {labels_df.shape[1]} columns")
    
    if 'filename' not in labels_df.columns:
        print("Error: 'filename' column not found in labels.csv")
        return
    
    filenames = labels_df['filename'].tolist()
    
    # 2. Generate Fingerprints for stratification
    print("Generating label fingerprints for stratification...")
    y, target_cols = get_label_fingerprints(labels_df)
    indices = np.arange(len(filenames))
    
    # Handle edge case: if all fingerprints are zero, just do random split
    if y.sum() == 0:
        print("Warning: No positive labels found, using random split")
        np.random.seed(args.random_state)
        np.random.shuffle(indices)
        split_point = int(len(indices) * (1 - args.test_size))
        train_idx = indices[:split_point]
        test_idx = indices[split_point:]
    else:
        # Downsampling for dev if requested
        if args.dev_sample is not None and args.dev_sample < len(filenames):
            print(f"Downsampling dataset to {args.dev_sample} samples for development...")
            remaining_size = len(filenames) - args.dev_sample
            
            msss_dev = MultilabelStratifiedShuffleSplit(
                n_splits=1, 
                train_size=args.dev_sample, 
                test_size=remaining_size, 
                random_state=args.random_state
            )
            dev_idx, _ = next(msss_dev.split(indices, y))
            
            indices = indices[dev_idx]
            y = y[dev_idx]
            filenames = [filenames[i] for i in dev_idx]
            indices = np.arange(len(filenames))  # Reset indices
            
            print(f"Development set size: {len(filenames)}")

        # 3. Stratified Split (train/test only)
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, 
            test_size=args.test_size, 
            random_state=args.random_state
        )
        train_idx, test_idx = next(msss.split(indices, y))
    
    print(f"Split Result: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # 4. Save Results
    train_files = [filenames[i] for i in train_idx]
    test_files = [filenames[i] for i in test_idx]
    
    # 4.1. all_valid_renji.json (Complete dictionary)
    split_data = {
        "train_files": train_files,
        "test_files": test_files,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "total_patients": len(filenames),
        "test_size": args.test_size,
        "random_state": args.random_state
    }
    
    save_path_all = os.path.join(args.save_dir, "all_valid_renji.json")
    with open(save_path_all, "w", encoding='utf-8') as f:
        json.dump(split_data, f, indent=4, ensure_ascii=False)
    print(f"Saved: {save_path_all}")

    # 4.2. train_renji.json
    save_path_train = os.path.join(args.save_dir, "train_renji.json")
    with open(save_path_train, "w", encoding='utf-8') as f:
        json.dump(train_files, f, indent=4, ensure_ascii=False)
    print(f"Saved: {save_path_train}")

    # 4.3. test_renji.json
    save_path_test = os.path.join(args.save_dir, "test_renji.json")
    with open(save_path_test, "w", encoding='utf-8') as f:
        json.dump(test_files, f, indent=4, ensure_ascii=False)
    print(f"Saved: {save_path_test}")
    
    # Print label distribution stats
    print("\n--- Label Distribution ---")
    train_y = y[train_idx]
    test_y = y[test_idx]
    
    for i, metric in enumerate(TARGET_METRICS[:10]):  # Show first 10
        train_pos = train_y[:, i].sum()
        test_pos = test_y[:, i].sum()
        print(f"  {metric}: Train={train_pos}, Test={test_pos}")
    if len(TARGET_METRICS) > 10:
        print(f"  ... and {len(TARGET_METRICS) - 10} more metrics")


if __name__ == "__main__":
    main()
