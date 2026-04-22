import pandas as pd
import pickle
import os
from tqdm import tqdm

def main():
    # Paths
    micro_events_path = "/home/ma-user/sfs_turbo/sai6/yangqian/tmp_input/mimic-iv-3.1/hosp/microbiologyevents.csv.gz"
    output_path = "/home/ma-user/sfs_turbo/Data/mimic-iv-cdm/microbiology_test_mapping.pkl"
    
    print(f"Reading {micro_events_path}...")
    
    # Read relevant columns to save memory
    cols = ['test_itemid', 'test_name']
    df = pd.read_csv(micro_events_path, usecols=cols)
    
    # Create mappings
    print("Generating mappings...")
    mapping = {}
    
    # test_itemid -> test_name
    test_map = df[['test_itemid', 'test_name']].dropna().drop_duplicates()
    for _, row in test_map.iterrows():
        mapping[int(row['test_itemid'])] = row['test_name']
     
    print(f"Total mapping items: {len(mapping)}")
    
    # Save
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(mapping, f)
        
    print("Done!")

if __name__ == "__main__":
    main()
