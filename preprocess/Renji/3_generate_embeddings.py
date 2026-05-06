"""
Generate pre-computed BERT embeddings for all unique texts in the Renji dataset.
Run this BEFORE training to create the embedding cache.

Usage:
    python preprocess/Renji/3_generate_embeddings.py
"""
import os
import sys
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dataset.renji_dataset import RenjiDataset


def main():
    # Configuration
    BERT_MODEL = "/home/ma-user/sfs_turbo/sai6/zkwan/model_weights/PubMedBERT"
    OUTPUT_PATH = "/home/ma-user/sfs_turbo/sai6/zkwan/.cache/embeddings/renji/text_embeddings.pt"
    BATCH_SIZE = 256  # BERT encoding batch size
    MAX_TOKEN_LEN = 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    print(f"Device: {DEVICE}")
    print(f"BERT Model: {BERT_MODEL}")
    print(f"Output: {OUTPUT_PATH}")
    
    # 1. Load tokenizer and model
    print("\nLoading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModel.from_pretrained(BERT_MODEL).to(DEVICE)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    text_dim = model.config.hidden_size
    print(f"Text embedding dimension: {text_dim}")
    
    # 2. Collect all unique texts from dataset
    print("\n" + "="*50)
    print("Collecting unique texts from dataset...")
    print("="*50)
    
    unique_texts = set()
    
    for split in ['train', 'test']:
        print(f"\nProcessing {split} split...")
        dataset = RenjiDataset(
            root_dir="/home/ma-user/sfs_turbo/sai6/zkwan/随访记录_labeled_v80",
            split=split,
            table_mode="table_only",
            shuffle=False
        )
        
        # Group by (fname_key, cutoff_day) to avoid redundant loading
        sample_groups = defaultdict(list)
        for idx, s in enumerate(dataset.samples):
            key = (s['fname_key'], s['cutoff_day'])
            sample_groups[key].append(idx)
        
        print(f"  Unique (patient, cutoff) combinations: {len(sample_groups)}")
        
        for key in tqdm(sample_groups.keys(), desc=f"Loading {split} tables"):
            sample_idx = sample_groups[key][0]
            sample = dataset[sample_idx]
            
            df = sample.get('measurement_table')
            if df is not None and len(df) > 0:
                # Collect Item texts
                for item in df['Item'].astype(str).tolist():
                    unique_texts.add(item)
                
                # Collect Value texts (non-numeric only)
                for val in df['Value'].astype(str).tolist():
                    try:
                        float(val)  # Skip numeric values
                    except:
                        unique_texts.add(val)
                
                # Collect Unit texts
                if 'Unit' in df.columns:
                    for unit in df['Unit'].astype(str).fillna('-').tolist():
                        unique_texts.add(unit)
    
    # Add special tokens
    unique_texts.add('[PAD]')
    unique_texts.add('[EMPTY]')
    unique_texts.add('-')
    unique_texts.add('0')  # Default numeric text
    
    unique_texts = list(unique_texts)
    print(f"\nTotal unique texts: {len(unique_texts)}")
    
    # 3. Encode all unique texts with BERT
    print("\n" + "="*50)
    print("Encoding texts with BERT...")
    print("="*50)
    
    embeddings_dict = {}
    
    for i in tqdm(range(0, len(unique_texts), BATCH_SIZE), desc="Encoding batches"):
        batch_texts = unique_texts[i:i+BATCH_SIZE]
        
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LEN,
            return_tensors='pt'
        ).to(DEVICE)
        
        with torch.no_grad():
            out = model(**tokens)
            embs = out.last_hidden_state[:, 0, :]  # CLS token
        
        # Store on CPU
        for j, text in enumerate(batch_texts):
            embeddings_dict[text] = embs[j].cpu()
    
    print(f"Encoded {len(embeddings_dict)} unique texts")
    
    # 4. Save cache
    print(f"\nSaving to {OUTPUT_PATH}...")
    torch.save({
        'embeddings': embeddings_dict,
        'text_dim': text_dim,
        'bert_model': BERT_MODEL,
    }, OUTPUT_PATH)
    
    # Stats
    file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Cache file size: {file_size:.2f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
