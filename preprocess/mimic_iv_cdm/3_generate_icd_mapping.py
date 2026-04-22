import pandas as pd
import pickle
import os
import gzip

def generate_icd_mapping():
    # Paths
    mimic_hosp_path = "/home/ma-user/sfs_turbo/sai6/yangqian/tmp_input/mimic-iv-3.1/hosp"
    output_dir = "/home/ma-user/sfs_turbo/Data/mimic-iv-cdm"
    
    d_icd_path = os.path.join(mimic_hosp_path, "d_icd_diagnoses.csv.gz")
    if not os.path.exists(d_icd_path):
        # Failback to unzipped if exists
        d_icd_path = os.path.join(mimic_hosp_path, "d_icd_diagnoses.csv")
    
    print(f"Reading {d_icd_path}...")
    
    # Load d_icd_diagnoses
    # Cols: icd_code, icd_version, long_title
    df = pd.read_csv(d_icd_path)
    
    # Create mapping: Long Title -> ICD Code
    # Note: Description in icd_diagnosis.csv seems to match long_title
    # We might need to handle potential duplicates or case sensitivity
    
    mapping = {}
    for _, row in df.iterrows():
        long_title = row['long_title']
        icd_code = str(row['icd_code'])
        
        # We store the mapping. If duplicates exist, we iterate.
        # Ideally, descriptions should be unique enough or we prioritize one version.
        # But MIMIC-IV sequences are mixed.
        mapping[long_title] = icd_code
        mapping[long_title.lower()] = icd_code
        
    output_path = os.path.join(output_dir, "icd_desc_to_code_mapping.pkl")
    print(f"Saving mapping with {len(mapping)} entries to {output_path}...")
    
    with open(output_path, 'wb') as f:
        pickle.dump(mapping, f)
        
    print("Done.")

if __name__ == "__main__":
    generate_icd_mapping()
