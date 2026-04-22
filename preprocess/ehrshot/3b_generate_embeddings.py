"""
Step 2: Multi-GPU Encoding using torchrun.
Usage: torchrun --nproc_per_node=4 preprocess/ehrshot/3b_generate_embeddings.py
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch.distributed as dist
import pickle

def main():
    BERT_MODEL = "/home/ma-user/sfs_turbo/sai6/zkwan/model_weights/PubMedBERT"
    CACHE_DIR = "/home/ma-user/sfs_turbo/sai6/zkwan/.cache/embeddings/ehrshot"
    FINAL_OUTPUT = os.path.join(CACHE_DIR, "text_embeddings.pt")
    HARVEST_CHECKPOINT = os.path.join(CACHE_DIR, "unique_texts_harvested.pkl")

    # Initialize Distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if not os.path.exists(HARVEST_CHECKPOINT):
        if rank == 0:
            print(f"Error: {HARVEST_CHECKPOINT} not found! Please run step1_harvest_texts.py first.")
        dist.destroy_process_group()
        sys.exit(1)

    # Load unique_texts on all ranks
    with open(HARVEST_CHECKPOINT, 'rb') as f:
        unique_texts = pickle.load(f)

    print(f"Rank {rank}: Encoding Phase starting...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModel.from_pretrained(BERT_MODEL).to(device)
    model.eval()
    
    # Shard the work
    my_shard = np.array_split(unique_texts, world_size)[rank]
    print(f"Rank {rank}: Processing {len(my_shard)} strings.")
    
    PARTIAL_CHECKPOINT = os.path.join(CACHE_DIR, f"partial_embs_rank_{rank}.pt")
    embeddings_dict = {}
    BATCH_SIZE = 512
    MAX_TOKEN_LEN = 512
    
    if os.path.exists(PARTIAL_CHECKPOINT):
        print(f"Rank {rank}: Resuming from partial checkpoint...")
        embeddings_dict = torch.load(PARTIAL_CHECKPOINT, weights_only=False)
        processed_count = len(embeddings_dict)
        my_shard = my_shard[processed_count:]
        print(f"Rank {rank}: Remaining to encode: {len(my_shard)}")

    for i in tqdm(range(0, len(my_shard), BATCH_SIZE), desc=f"Rank {rank} Encoding", disable=rank != 0):
        batch_texts = list(my_shard[i:i+BATCH_SIZE])
        
        tokens = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=MAX_TOKEN_LEN,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            out = model(**tokens)
            embs = out.last_hidden_state[:, 0, :].cpu()
        
        for j, text in enumerate(batch_texts):
            embeddings_dict[text] = embs[j]
            
        if (i // BATCH_SIZE) % 50 == 0:
            torch.save(embeddings_dict, PARTIAL_CHECKPOINT)

    # Final save for this rank
    torch.save(embeddings_dict, PARTIAL_CHECKPOINT)
    
    # Wait for all GPUs to finish
    dist.barrier()

    # --- Phase 3: Merging (Rank 0 only) ---
    if rank == 0:
        print("Phase 3: Merging partial results...")
        final_embeddings = {}
        for r in range(world_size):
            p_path = os.path.join(CACHE_DIR, f"partial_embs_rank_{r}.pt")
            shard_embs = torch.load(p_path, weights_only=False)
            final_embeddings.update(shard_embs)
            
        print(f"Merged {len(final_embeddings)} embeddings. Dimension: {model.config.hidden_size}")
        torch.save({
            'embeddings': final_embeddings,
            'text_dim': model.config.hidden_size,
            'bert_model': BERT_MODEL,
        }, FINAL_OUTPUT)
        
        # Cleanup temporary files (optional but recommended)
        # os.remove(HARVEST_CHECKPOINT) 
        for r in range(world_size):
            os.remove(os.path.join(CACHE_DIR, f"partial_embs_rank_{r}.pt"))
            
        print(f"Successfully saved to {FINAL_OUTPUT}")
        print("Done!")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()