"""
Expand CLMBR vocabulary to include missing codes for Renji dataset.

New codes to add (with clinically-meaningful bins):
- CMV-DNA (LOINC/29604-6)
- 环孢素谷浓度 (LOINC/53828-0)  
- 环孢素峰浓度 (LOINC/32997-9)
"""

import json
import os
import sys

# Paths
model_dir = "/home/ma-user/sfs_turbo/model_weights/clmbr-t-base"
ORIGINAL_DICT = os.path.join(model_dir, "clmbr_v8_original_dictionary.json")
EXPANDED_DICT = "/home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/models/clmbr/clmbr_expanded_dictionary.json"
MAPPING_FILE =  "/home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/models/clmbr/mapping.json"

INF = 1.7976931348623157e+308

NEW_CODES = {
    # CMV-DNA: Qualitative/quantitative viral load
    # Clinically: Negative is normal, higher copies = worse
    "LOINC/29604-6": {
        "bins": [
            (-INF, 0),        # Negative/Undetected
            (0, 500),         # Very low (borderline)
            (500, 1000),      # Low
            (1000, 10000),    # Moderate
            (10000, 100000),  # High  
            (100000, INF),    # Very high
        ],
        "type": "numeric"
    },
    
    # 环孢素谷浓度 (CsA Trough): Target 100-200 ng/mL for liver transplant maintenance
    "LOINC/53828-0": {
        "bins": [
            (-INF, 50),       # Very low (subtherapeutic)
            (50, 100),        # Low
            (100, 150),       # Normal-low
            (150, 200),       # Normal-high
            (200, 300),       # High
            (300, 400),       # Very high
            (400, INF),       # Toxic range
        ],
        "type": "numeric"
    },
    
    # 环孢素峰浓度 (CsA Peak): Target ~400-800 ng/mL typically
    "LOINC/32997-9": {
        "bins": [
            (-INF, 200),      # Very low
            (200, 400),       # Low
            (400, 600),       # Normal-low
            (600, 800),       # Normal-high
            (800, 1000),      # High
            (1000, 1200),     # Very high
            (1200, INF),      # Toxic range
        ],
        "type": "numeric"
    },
}


def load_dictionary(path):
    """Load CLMBR dictionary."""
    print(f"Loading dictionary from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_dictionary(data, path):
    """Save CLMBR dictionary."""
    print(f"Saving expanded dictionary to {path}...")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def create_entries_for_code(code_string, config):
    """Create dictionary entries for a new code with bins."""
    entries = []
    bins = config["bins"]
    code_type = config["type"]
    
    for val_start, val_end in bins:
        entry = {
            "code_string": code_string,
            "text_string": "",
            "type": code_type,
            "val_start": val_start,
            "val_end": val_end,
            "weight": 0.0  # New codes start with zero weight
        }
        entries.append(entry)
    
    return entries


def main():
    # Load original dictionary
    if not os.path.exists(ORIGINAL_DICT):
        print(f"Error: Original dictionary not found at {ORIGINAL_DICT}")
        sys.exit(1)
    
    original = load_dictionary(ORIGINAL_DICT)
    
    # Check existing codes
    existing_codes = set()
    for entry in original.get("regular", []):
        existing_codes.add(entry["code_string"])
    
    print(f"Original dictionary has {len(original['regular'])} entries")
    print(f"Unique code strings: {len(existing_codes)}")
    
    # Check which new codes already exist
    for code in NEW_CODES:
        if code in existing_codes:
            print(f"  Warning: {code} already exists in dictionary (will add new bins)")
        else:
            print(f"  Adding new code: {code}")
    
    # Create new entries
    new_entries = []
    for code_string, config in NEW_CODES.items():
        entries = create_entries_for_code(code_string, config)
        new_entries.extend(entries)
        print(f"  Created {len(entries)} bin entries for {code_string}")
    
    # Append to regular list
    expanded = original.copy()
    expanded["regular"] = original["regular"] + new_entries
    
    print(f"\nExpanded dictionary has {len(expanded['regular'])} entries (+{len(new_entries)})")
    
    # Save expanded dictionary
    save_dictionary(expanded, EXPANDED_DICT)
    
    # Also update mapping.json to move discard items to mappable
    if os.path.exists(MAPPING_FILE):
        print("\nUpdating mapping.json...")
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        # Move discard items to mappable
        if "discard" in mapping:
            for key, value in mapping["discard"].items():
                if key not in mapping.get("mappable", {}):
                    mapping["mappable"][key] = value
                    print(f"  Moved {key} from discard to mappable")
            
            # Clear discard section
            mapping["discard"] = {}
        
        with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=4, ensure_ascii=False)
        
        print("Mapping file updated.")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
