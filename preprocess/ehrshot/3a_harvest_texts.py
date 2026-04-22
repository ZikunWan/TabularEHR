"""
Step 1: CPU Harvesting of unique texts.
Usage: python preprocess/ehrshot/3a_harvest_texts.py
"""
import os
import sys
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import pickle

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dataset.ehrshot_dataset import EHRSHOTDataset

def harvest_worker(worker_args):
    """Worker function to extract strings from a set of patient slices."""
    slice_list, data_dir, index_path = worker_args
    import pandas as pd
    from dataset.ehrshot_dataset import EHRSHOTDataset
    
    dataset = EHRSHOTDataset(
        root_dir=data_dir,
        sample_info_path=index_path,
        split=None,
        lazy_mode=True,
        return_table=True,
        shuffle=False
    )
    
    local_unique = set()
    for sample_idx in slice_list:
        try:
            sample = dataset[sample_idx]
            df = sample.get('measurement_table')
            if df is not None and not df.empty:
                local_unique.update(df['Item'].astype(str).unique())
                local_unique.update(df['Unit'].astype(str).fillna('-').unique())
                raw_values = df['Value'].astype(str).unique()
                for val in raw_values:
                    try:
                        float(val)
                    except ValueError:
                        local_unique.add(val)
        except Exception:
            continue
    return local_unique

def main():
    DATA_DIR = "/home/ma-user/sfs_turbo/sai6/zkwan/EHRSHOT"
    INDEX_FILE = os.path.join(DATA_DIR, "index", "ehrshot_all.csv")
    CACHE_DIR = "/home/ma-user/sfs_turbo/sai6/zkwan/.cache/embeddings/ehrshot"
    HARVEST_CHECKPOINT = os.path.join(CACHE_DIR, "unique_texts_harvested.pkl")
    
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Phase 1: Harvesting unique texts from dataset...")
    dataset = EHRSHOTDataset(
        root_dir=DATA_DIR,
        sample_info_path=INDEX_FILE,
        split=None,
        lazy_mode=True,
        return_table=True,
        shuffle=False
    )
    
    unique_slices = {} # (p, b, e) -> index
    for i, s in enumerate(dataset.sample_info):
        key = (s['patient_id'], s['period_begin'], s['period_end'])
        if key not in unique_slices:
            unique_slices[key] = i
    
    slice_indices = list(unique_slices.values())
    print(f"Total samples: {len(dataset)}, Unique slices to process: {len(slice_indices)}")
    
    num_workers = 32 # CPU cores
    chunks = np.array_split(slice_indices, num_workers)
    worker_args = [(list(chunk), DATA_DIR, INDEX_FILE) for chunk in chunks]
    
    all_unique = set(['[PAD]', '[EMPTY]', '[UNKNOWN]', '-', '0'])
    
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(harvest_worker, worker_args), total=num_workers, desc="CPU Harvesting"))
        for res in results:
            all_unique.update(res)
    
    unique_texts = [str(t) for t in all_unique if t and str(t).strip()]
    print(f"Harvested {len(unique_texts)} unique strings. Saving checkpoint...")
    
    with open(HARVEST_CHECKPOINT, 'wb') as f:
        pickle.dump(unique_texts, f)
    print(f"Successfully saved to {HARVEST_CHECKPOINT}")

if __name__ == "__main__":
    main()