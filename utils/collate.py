import torch
from typing import List, Dict, Any, Optional
import pandas as pd
from .load_embedding import get_embedding



def build_table_token_tensors(
    tables_list: List[Any],
    text_to_idx: Dict[str, int],
    pad_idx: int,
    row_block_ids_list: Optional[List[List[Any]]] = None,
    kept_block_ids_list: Optional[List[List[Any]]] = None,
    type_vocab: Optional[Dict[str, int]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build table token tensors for TableEncoder from a list of measurement tables.
    Returns tensors: item_ids/unit_ids/value_text_ids/times/numeric_values/numeric_mask/seq_mask/type_ids.
    """
    if row_block_ids_list is None:
        row_block_ids_list = [None] * len(tables_list)
    if kept_block_ids_list is None:
        kept_block_ids_list = [None] * len(tables_list)

    all_item_ids, all_unit_ids, all_value_text_ids = [], [], []
    all_times, all_numeric_values, all_numeric_masks = [], [], []
    all_type_ids = []
    seq_lens = []

    for b_idx, t in enumerate(tables_list):
        # Optional row filtering by kept block ids (for text truncation alignment).
        if isinstance(t, pd.DataFrame) and len(t) > 0:
            row_block_ids = row_block_ids_list[b_idx]
            kept_block_ids = kept_block_ids_list[b_idx]
            if isinstance(row_block_ids, list) and isinstance(kept_block_ids, list) and len(row_block_ids) == len(t):
                kept_set = set(str(x) for x in kept_block_ids)
                if len(kept_set) > 0:
                    keep_mask = [str(bid) in kept_set for bid in row_block_ids]
                    t = t.loc[keep_mask].reset_index(drop=True)

        if isinstance(t, pd.DataFrame) and len(t) > 0:
            sl = len(t)
            seq_lens.append(sl)

            item_series = t["Item"].astype(str)
            item_ids = item_series.map(text_to_idx).fillna(pad_idx).astype(int).tolist()

            if "Unit" in t.columns and not t["Unit"].isnull().all():
                unit_series = t["Unit"].astype(str).fillna("-")
                unit_ids = unit_series.map(text_to_idx).fillna(pad_idx).astype(int).tolist()
            else:
                unit_ids = [text_to_idx.get("-", pad_idx)] * sl

            val_series = t["Value"].astype(str)
            numeric_series = pd.to_numeric(val_series, errors="coerce")
            is_numeric = numeric_series.notna()
            num_mask = is_numeric.astype(float).tolist()
            num_vals = numeric_series.fillna(0.0).tolist()
            text_lookup_series = val_series.where(~is_numeric, "0")
            value_text_ids_list = text_lookup_series.map(text_to_idx).fillna(pad_idx).astype(int).tolist()

            all_item_ids.append(item_ids)
            all_unit_ids.append(unit_ids)
            all_value_text_ids.append(value_text_ids_list)
            all_numeric_values.append(num_vals)
            all_numeric_masks.append(num_mask)

            if "Time" in t.columns:
                if not pd.api.types.is_datetime64_any_dtype(t["Time"]):
                    time_col = pd.to_datetime(t["Time"], errors="coerce")
                else:
                    time_col = t["Time"]
                if len(time_col) > 0:
                    first_date = time_col.iloc[0]
                    delta_hours = (time_col - first_date).dt.total_seconds() / 3600 + 1
                    delta_hours = delta_hours.fillna(0.0).tolist()
                else:
                    delta_hours = [0.0] * sl
            else:
                delta_hours = [0.0] * sl
            all_times.append(delta_hours)

            if "Type" in t.columns:
                type_ids = t["Type"].map(type_vocab).fillna(0).astype(int).tolist()
            elif "Category" in t.columns:
                type_ids = [type_vocab.get(str(c), 0) for c in t["Category"]]
            else:
                type_ids = [0] * sl
            all_type_ids.append(type_ids)
        else:
            seq_lens.append(1)
            all_item_ids.append([text_to_idx.get("[EMPTY]", pad_idx)])
            all_unit_ids.append([text_to_idx.get("-", pad_idx)])
            all_value_text_ids.append([text_to_idx.get("0", pad_idx)])
            all_times.append([0.0])
            all_numeric_values.append([0.0])
            all_numeric_masks.append([0.0])
            all_type_ids.append([0])

    bs = len(tables_list)
    max_len = max(seq_lens)

    item_ids_t = torch.zeros(bs, max_len, dtype=torch.long)
    unit_ids_t = torch.zeros(bs, max_len, dtype=torch.long)
    value_text_ids_t = torch.zeros(bs, max_len, dtype=torch.long)
    times_t = torch.zeros(bs, max_len, dtype=torch.float)
    numeric_values_t = torch.zeros(bs, max_len, dtype=torch.float)
    numeric_mask_t = torch.zeros(bs, max_len, dtype=torch.float)
    seq_mask_t = torch.zeros(bs, max_len, dtype=torch.float)
    type_ids_t = torch.zeros(bs, max_len, dtype=torch.long)

    for i in range(bs):
        sl = seq_lens[i]
        item_ids_t[i, :sl] = torch.tensor(all_item_ids[i], dtype=torch.long)
        unit_ids_t[i, :sl] = torch.tensor(all_unit_ids[i], dtype=torch.long)
        value_text_ids_t[i, :sl] = torch.tensor(all_value_text_ids[i], dtype=torch.long)
        times_t[i, :sl] = torch.tensor(all_times[i], dtype=torch.float)
        numeric_values_t[i, :sl] = torch.tensor(all_numeric_values[i], dtype=torch.float)
        numeric_mask_t[i, :sl] = torch.tensor(all_numeric_masks[i], dtype=torch.float)
        seq_mask_t[i, :sl] = 1.0
        type_ids_t[i, :sl] = torch.tensor(all_type_ids[i], dtype=torch.long)

    return {
        "item_ids": item_ids_t,
        "unit_ids": unit_ids_t,
        "value_text_ids": value_text_ids_t,
        "times": times_t,
        "numeric_values": numeric_values_t,
        "numeric_mask": numeric_mask_t,
        "seq_mask": seq_mask_t,
        "type_ids": type_ids_t,
    }


def create_collate_fn(type_vocab=None, label_map=None):
    """
    Requires dataset __getitem__ to return a dictionary with:
    - `label`: An integer, float, or tensor representing the classification target.
    - `measurement_table`: (Optional) A pandas DataFrame containing columns ['Time', 'Item', 'Value', 'Unit', 'Category'].
    """
    if type_vocab is None: type_vocab = {}
    
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        bs = len(batch)
        all_item_embs, all_unit_embs, all_value_embs = [], [], []
        all_times, all_num_vals, all_num_masks, all_type_ids = [], [], [], []
        labels_list, seq_lens = [], []
        
        for sample in batch:
            # 1. Parse Label
            raw_label = sample['output']
            if isinstance(raw_label, torch.Tensor):
                labels_list.append(raw_label.long())
            elif isinstance(raw_label, str):
                norm_label = raw_label.strip().strip('"').strip("'").strip()
                norm_label_lower = norm_label.lower()
                if norm_label_lower == 'yes':
                    labels_list.append(1)
                elif norm_label_lower == 'no':
                    labels_list.append(0)
                elif label_map is not None and norm_label in label_map:
                    labels_list.append(label_map[norm_label])
                elif label_map is not None and norm_label_lower in label_map:
                    labels_list.append(label_map[norm_label_lower])
                else:
                    labels_list.append(int(float(norm_label)))
            else:
                labels_list.append(int(raw_label))
            
            # 2. Parse Table
            df = sample.get('measurement_table')
            if df is not None and not df.empty:
                items = df['Item'].astype(str).tolist()
                units = df['Unit'].astype(str).fillna('-').tolist() if 'Unit' in df.columns else ['-'] * len(df)
                
                num_series = pd.to_numeric(df['Value'], errors="coerce")
                num_mask = num_series.notna().astype(float).tolist()
                num_vals = num_series.fillna(0.0).tolist()
                
                value_embs = [get_embedding("0") if m else get_embedding(str(v)) for v, m in zip(df['Value'], num_mask)]
                
                # Assume dataset already converted time to relative days float. E.g. 0.0 for MIMIC
                time_vals = pd.to_numeric(df['Time'], errors="coerce").fillna(0.0).tolist() if 'Time' in df.columns else [0.0] * len(items)
                
                type_ids = [type_vocab.get(str(c), 0) for c in df["Category"]] if "Category" in df.columns else [0] * len(items)
                
                seq_lens.append(len(items))
                all_item_embs.append(torch.stack([get_embedding(x) for x in items]))
                all_unit_embs.append(torch.stack([get_embedding(x) for x in units]))
                all_value_embs.append(torch.stack(value_embs))
                all_times.append(time_vals)
                all_num_vals.append(num_vals)
                all_num_masks.append(num_mask)
                all_type_ids.append(type_ids)
            else:
                seq_lens.append(1)
                all_item_embs.append(get_embedding('[EMPTY]').unsqueeze(0))
                all_unit_embs.append(get_embedding('-').unsqueeze(0))
                all_value_embs.append(get_embedding('0').unsqueeze(0))
                all_times.append([0.0])
                all_num_vals.append([0.0])
                all_num_masks.append([1.0])
                all_type_ids.append([0])
        
        # 3. Padding tensors to max length
        max_len = max(seq_lens)
        pad_emb = get_embedding('[PAD]')
        for i in range(bs):
            p = max_len - seq_lens[i]
            if p > 0:
                pad_t = pad_emb.unsqueeze(0).repeat(p, 1)
                all_item_embs[i] = torch.cat([all_item_embs[i], pad_t])
                all_unit_embs[i] = torch.cat([all_unit_embs[i], pad_t])
                all_value_embs[i] = torch.cat([all_value_embs[i], pad_t])
                all_times[i].extend([0.0]*p)
                all_num_vals[i].extend([0.0]*p)
                all_num_masks[i].extend([0.0]*p) # padding values should have 0 mask
                all_type_ids[i].extend([0]*p)
        
        seq_mask = torch.zeros(bs, max_len)
        for i, sl in enumerate(seq_lens): seq_mask[i, :sl] = 1.0
            
        labels_tensor = torch.stack(labels_list) if isinstance(labels_list[0], torch.Tensor) else torch.tensor(labels_list)
        
        return {
            "item_emb": torch.stack(all_item_embs),
            "unit_emb": torch.stack(all_unit_embs),
            "value_emb": torch.stack(all_value_embs),
            "times": torch.tensor(all_times, dtype=torch.float32),
            "numeric_values": torch.tensor(all_num_vals, dtype=torch.float32),
            "numeric_mask": torch.tensor(all_num_masks, dtype=torch.float32),
            "type_ids": torch.tensor(all_type_ids, dtype=torch.long),
            "seq_mask": seq_mask,
            "labels": labels_tensor
        }
    return collate_fn
