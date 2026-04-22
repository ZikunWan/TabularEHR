"""
eICU Diagnosis Mapping Utilities
Uses CCS (Clinical Classifications Software) to map diagnoses to categories
Adapted from GenHPF's make_dx_mapping method
"""
import os
import pandas as pd
import logging
from collections import Counter
import treelib

logger = logging.getLogger(__name__)


def load_ccs_mapping(ccs_path):
    """
    Load CCS multi-level diagnosis mapping
    
    Args:
        ccs_path: Path to ccs_multi_dx_tool_2015.csv
    
    Returns:
        dict: ICD-9 code -> CCS Level 1 category (0-17)
    """
    print(f"Loading CCS mapping from {ccs_path}...")
    ccs_dx = pd.read_csv(ccs_path)
    
    # Remove quotes and convert to proper format
    ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:-1].str.strip()
    ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:-1].astype(int) - 1  # 0-indexed
    
    icd2cat = dict(zip(ccs_dx["'ICD-9-CM CODE'"], ccs_dx["'CCS LVL 1'"]))
    
    print(f"  Loaded {len(icd2cat)} ICD-9 -> CCS mappings")
    print(f"  CCS categories: 0-{max(icd2cat.values())}")
    
    return icd2cat


def load_icd10_to_icd9_gem(gem_path):
    """
    Load ICD-10 to ICD-9 GEM (General Equivalence Mappings)
    
    Args:
        gem_path: Path to icd10cmtoicd9gem.csv
    
    Returns:
        dict: ICD-10 code -> ICD-9 code
    """
    print(f"Loading ICD-10 to ICD-9 GEM from {gem_path}...")
    gem = pd.read_csv(gem_path)
    
    # Create mapping
    icd10_to_icd9 = dict(zip(gem['icd10cm'], gem['icd9cm']))
    
    print(f"  Loaded {len(icd10_to_icd9)} ICD-10 -> ICD-9 mappings")
    
    return icd10_to_icd9


def create_diagnosis_mapping(diagnosis_df, icd2cat, icd10_to_icd9):
    """
    Create mapping from diagnosis string to CCS category
    
    This implements GenHPF's make_dx_mapping logic:
    1. Map diagnosis strings to ICD codes
    2. Convert ICD-10 to ICD-9 if needed
    3. Map ICD-9 to CCS categories
    4. Use diagnosis string hierarchy for unmapped diagnoses
    
    Args:
        diagnosis_df: DataFrame with 'diagnosisstring' and 'icd9code' columns
        icd2cat: ICD-9 -> CCS category mapping
        icd10_to_icd9: ICD-10 -> ICD-9 mapping
    
    Returns:
        dict: diagnosis string -> CCS category (0-17, or -1 for unmapped)
    """
    print("\nCreating diagnosis string -> CCS category mapping...")
    
    # Step 1: Make diagnosisstring -> ICD code dictionary
    diagnosis = diagnosis_df[['diagnosisstring', 'icd9code']].copy()
    
    # Remove NaN ICD codes
    str2code = diagnosis.dropna(subset=['icd9code'])
    str2code = str2code.groupby('diagnosisstring').first().reset_index()
    
    # Handle multiple codes per diagnosis (comma-separated)
    str2code['icd9code'] = str2code['icd9code'].str.split(',')
    str2code = str2code.explode('icd9code')
    
    # Remove periods from ICD codes
    str2code['icd9code'] = str2code['icd9code'].str.replace('.', '', regex=False)
    
    print(f"  Step 1: {len(str2code)} diagnosis string -> ICD code mappings")
    
    # Step 2: Identify ICD-10 codes (codes NOT in ICD-9 CCS mapping)
    str2code_icd10 = str2code[~str2code['icd9code'].isin(icd2cat.keys())]
    
    print(f"  Step 2: Found {len(str2code_icd10)} ICD-10 codes to convert")
    
    # Convert ICD-10 to ICD-9 using GEM
    icd10_to_icd9_extended = icd10_to_icd9.copy()
    
    # For codes not in GEM, try progressive truncation
    manual_mappings = {}
    for code_10 in set(str2code_icd10['icd9code']) - set(icd10_to_icd9.keys()):
        # Try truncating from right to left
        for i in range(len(code_10), 0, -1):
            tgt_10 = code_10[:i]
            if tgt_10 in icd10_to_icd9:
                manual_mappings[code_10] = icd10_to_icd9[tgt_10]
                break
            # Try finding partial matches in GEM
            partial_matches = [k for k in icd10_to_icd9.keys() if k.startswith(tgt_10)]
            if partial_matches:
                # Use most common ICD-9 mapping
                icd9_codes = [icd10_to_icd9[k] for k in partial_matches]
                manual_mappings[code_10] = Counter(icd9_codes).most_common(1)[0][0]
                break
    
    print(f"  Step 2b: Created {len(manual_mappings)} additional ICD-10->ICD-9 mappings via truncation")
    
    icd10_to_icd9_extended.update(manual_mappings)
    
    # Step 3: Convert available diagnosis strings to CCS categories
    str2cat = {}
    for _, row in str2code.iterrows():
        diag_str = row['diagnosisstring']
        code = row['icd9code']
        
        # Direct ICD-9 match
        if code in icd2cat:
            cat = icd2cat[code]
            if diag_str in str2cat and str2cat[diag_str] != cat:
                logger.warning(f"{diag_str} has multiple categories: {cat}, {str2cat[diag_str]}")
            str2cat[diag_str] = cat
        # ICD-10 -> ICD-9 -> CCS
        elif code in icd10_to_icd9_extended:
            icd9_code = icd10_to_icd9_extended[code]
            if icd9_code in icd2cat:
                cat = icd2cat[icd9_code]
                if diag_str in str2cat and str2cat[diag_str] != cat:
                    logger.warning(f"{diag_str} has multiple categories: {cat}, {str2cat[diag_str]}")
                str2cat[diag_str] = cat
    
    print(f"  Step 3: Mapped {len(str2cat)} diagnosis strings to CCS categories")
    
    # Step 4: Use diagnosis hierarchy for unmapped diagnoses
    print("  Step 4: Using diagnosis hierarchy for unmapped diagnoses...")
    
    # Build tree structure from diagnosis strings
    tree = treelib.Tree()
    tree.create_node("root", "root")
    
    for dx, cat in str2cat.items():
        dx_parts = dx.split('|')
        
        # Build tree path
        if not tree.contains(dx_parts[0]):
            tree.create_node(-1, dx_parts[0], parent="root")
        
        for i in range(2, len(dx_parts)):
            parent_path = '|'.join(dx_parts[:i-1])
            current_path = '|'.join(dx_parts[:i])
            if not tree.contains(current_path):
                tree.create_node(-1, current_path, parent=parent_path)
        
        # Add leaf node with category
        full_path = '|'.join(dx_parts)
        if not tree.contains(full_path):
            parent_path = '|'.join(dx_parts[:-1]) if len(dx_parts) > 1 else dx_parts[0]
            tree.create_node(cat, full_path, parent=parent_path)
    
    # Update non-leaf nodes with majority vote
    nid_list = list(tree.expand_tree(mode=treelib.Tree.DEPTH))
    nid_list.reverse()  # Bottom-up
    
    for nid in nid_list:
        node = tree.get_node(nid)
        if node.is_leaf():
            continue
        elif node.tag == -1:  # Unassigned internal node
            children = tree.children(nid)
            child_tags = [child.tag for child in children if child.tag != -1]
            if child_tags:
                # Majority vote
                node.tag = Counter(child_tags).most_common(1)[0][0]
    
    # Assign categories to unmapped diagnoses using tree
    unmatched_dxs = set(diagnosis['diagnosisstring']) - set(str2cat.keys())
    newly_mapped = 0
    
    for dx in unmatched_dxs:
        dx_parts = dx.split('|')
        # Try to find in tree, going up the hierarchy
        for i in range(len(dx_parts), 1, -1):  # Don't go to root level
            path = '|'.join(dx_parts[:i])
            if tree.contains(path):
                cat = tree.get_node(path).tag
                if cat != -1:
                    str2cat[dx] = cat
                    newly_mapped += 1
                    break
    
    print(f"  Step 4: Mapped {newly_mapped} additional diagnoses using hierarchy")
    
    # Final statistics
    total_diagnoses = len(set(diagnosis['diagnosisstring']))
    mapped_diagnoses = len(str2cat)
    coverage = mapped_diagnoses / total_diagnoses * 100 if total_diagnoses > 0 else 0
    
    print(f"\n  Total unique diagnoses: {total_diagnoses}")
    print(f"  Mapped diagnoses: {mapped_diagnoses} ({coverage:.1f}%)")
    print(f"  Unmapped diagnoses: {total_diagnoses - mapped_diagnoses}")
    
    # Category distribution
    cat_dist = Counter(str2cat.values())
    print(f"\n  CCS category distribution:")
    for cat in sorted(cat_dist.keys()):
        print(f"    Category {cat}: {cat_dist[cat]} diagnoses")
    
    return str2cat


def process_diagnosis_labels(labeled_cohorts, diagnosis_path, str2cat):
    """
    Add diagnosis labels to cohorts
    
    Args:
        labeled_cohorts: DataFrame with cohort information
        diagnosis_path: Path to diagnosis.csv
        str2cat: diagnosis string -> CCS category mapping
    
    Returns:
        DataFrame with added 'diagnosis' column (list of CCS categories)
    """
    print("\nProcessing diagnosis labels for cohorts...")
    
    # Load diagnosis data
    dx = pd.read_csv(diagnosis_path, low_memory=False)
    
    # Merge with cohorts to get ICU stay IDs
    icustay_key = 'patientunitstayid'
    hadm_key = 'patienthealthsystemstayid'
    
    dx = dx.merge(
        labeled_cohorts[[icustay_key, hadm_key]],
        on=icustay_key,
        how='inner'
    )
    
    # Map diagnosis strings to categories
    dx['diagnosis'] = dx['diagnosisstring'].map(lambda x: str2cat.get(x, -1))
    
    # Ignore rare class (14) and unmapped (-1)
    dx = dx[(dx['diagnosis'] != -1) & (dx['diagnosis'] != 14)]
    
    # Adjust categories > 14 down by 1 (to account for removed class 14)
    dx.loc[dx['diagnosis'] >= 14, 'diagnosis'] -= 1
    
    # Group by hospital admission, get unique diagnoses per admission
    dx = dx.groupby(hadm_key)['diagnosis'].agg(lambda x: list(set(x))).to_frame()
    
    # Merge with cohorts
    labeled_cohorts = labeled_cohorts.merge(dx, on=hadm_key, how='left')
    
    # Fill NaN with empty lists
    labeled_cohorts['diagnosis'] = labeled_cohorts['diagnosis'].apply(
        lambda x: [] if not isinstance(x, list) else x
    )
    
    # Statistics
    has_diagnosis = labeled_cohorts['diagnosis'].apply(len) > 0
    avg_diagnoses = labeled_cohorts['diagnosis'].apply(len).mean()
    
    print(f"  Cohorts with diagnoses: {has_diagnosis.sum()} / {len(labeled_cohorts)}")
    print(f"  Average diagnoses per cohort: {avg_diagnoses:.2f}")
    
    return labeled_cohorts


if __name__ == "__main__":
    # Test the mapping creation
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python diagnosis_utils.py <ccs_path> <gem_path> <diagnosis_csv>")
        sys.exit(1)
    
    ccs_path = sys.argv[1]
    gem_path = sys.argv[2]
    diagnosis_path = sys.argv[3]
    
    # Load tools
    icd2cat = load_ccs_mapping(ccs_path)
    icd10_to_icd9 = load_icd10_to_icd9_gem(gem_path)
    
    # Load diagnosis data
    print(f"\nLoading diagnosis data from {diagnosis_path}...")
    diagnosis_df = pd.read_csv(diagnosis_path, low_memory=False)
    print(f"  Loaded {len(diagnosis_df)} diagnosis records")
    
    # Create mapping
    str2cat = create_diagnosis_mapping(diagnosis_df, icd2cat, icd10_to_icd9)
    
    # Save mapping
    output_path = "diagnosis_to_ccs_mapping.csv"
    mapping_df = pd.DataFrame([
        {'diagnosisstring': k, 'ccs_category': v}
        for k, v in str2cat.items()
    ])
    mapping_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved mapping to {output_path}")
