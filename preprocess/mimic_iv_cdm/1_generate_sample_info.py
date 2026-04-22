import os
import pickle
import random
import pandas as pd
from tqdm import tqdm

def main():
    # Configuration
    root_dir = "/home/ma-user/sfs_turbo/Data/mimic-iv-cdm"
    output_index_dir = os.path.join(root_dir, "index")
    os.makedirs(output_index_dir, exist_ok=True)
    
    categories = ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']
    seed = 42
    splits = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    
    print(f"Generating splits for categories: {categories}")
    print(f"Splits: {splits}")
    
    all_index_records = []
    
    random.seed(seed)
    
    for category in categories:
        pkl_path = os.path.join(root_dir, f"{category}_hadm_info_first_diag.pkl")
        print(f"\nProcessing {category} from {pkl_path}...")
        
        if not os.path.exists(pkl_path):
            print(f"Warning: File not found: {pkl_path}, skipping.")
            continue
            
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        hadm_ids = list(data.keys())
        random.shuffle(hadm_ids)
        
        n_total = len(hadm_ids)
        n_train = int(n_total * splits['train'])
        n_val = int(n_total * splits['val'])
        # remaining for test
        
        train_ids = hadm_ids[:n_train]
        val_ids = hadm_ids[n_train:n_train+n_val]
        test_ids = hadm_ids[n_train+n_val:]
        
        print(f"  Total: {n_total}, Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
        
        # Helper to process IDs for a split
        def process_split(ids, split_name):
            for hid in ids:
                item = data[hid]
                
                # Add to index
                lbl = item.get('ICD Diagnosis', '[]')
                try:
                    if isinstance(lbl, str):
                        import ast
                        lbl_list = ast.literal_eval(lbl)
                        lbl_str = ";".join(lbl_list) if isinstance(lbl_list, list) else str(lbl_list)
                    else:
                        lbl_str = str(lbl)
                except:
                    lbl_str = str(lbl)

                all_index_records.append({
                    'hadm_id': hid,
                    'split': split_name,
                    'category': category,
                    'icd': lbl_str
                })
        
        process_split(train_ids, 'train')
        process_split(val_ids, 'val')
        process_split(test_ids, 'test')

    # Save Index CSVs
    df = pd.DataFrame(all_index_records)
    print("\nSaving index CSVs...")
    
    # Save all
    df.to_csv(os.path.join(output_index_dir, "mimiciv_cdm_all.csv"), index=False)
    
    # Save splits
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        split_path = os.path.join(output_index_dir, f"mimiciv_cdm_{split}.csv")
        split_df.to_csv(split_path, index=False)
        print(f"  Saved {split}: {len(split_df)} records to {split_path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
