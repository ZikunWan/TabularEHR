import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import shutil
import re
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dataset.renji_dataset import RenjiDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Renji Model Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to base model")
    parser.add_argument("--root_dir", type=str, default="./data/Renji", help="Data root directory")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallelism Size")
    parser.add_argument("--max_model_len", type=int, default=32000, help="Max Model Length (Context)")
    parser.add_argument("--target_metrics", type=str, default=None, help="Comma-separated list of target metrics (e.g., 'ALT,AST,TB')")
    parser.add_argument("--target_windows", type=str, default=None, help="Comma-separated list of target windows (e.g., 'win1,win3')")
    return parser.parse_args()

def check_and_merge_model(base_model_path, adapter_path):
    if not base_model_path:
        return adapter_path
    merged_path = adapter_path.rstrip("/\\") + "_merged"
    if os.path.exists(merged_path):
        print(f"Merged model found at {merged_path}, using it.")
        return merged_path
    print(f"Merging Base ({base_model_path}) + Adapter ({adapter_path})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map="cpu", 
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(merged_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, fix_mistral_regex=True)
    tokenizer.save_pretrained(merged_path)
    del model, base_model
    import gc
    gc.collect()
    return merged_path

def truncate_history_exact(instruction, history_text, tokenizer, max_len=32000):
    dummy_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"\n{instruction}"}
    ]

    skeleton_prompt = tokenizer.apply_chat_template(
        dummy_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    skeleton_ids = tokenizer.encode(skeleton_prompt, add_special_tokens=False)
    skeleton_len = len(skeleton_ids)
    
    # 计算 Budget
    safety_buffer = 100 
    history_budget = max_len - skeleton_len - safety_buffer
    
    if history_budget < 1: history_budget = 1
    
    # Tokenize History 并截断
    history_ids = tokenizer.encode(history_text, add_special_tokens=False)
    
    if len(history_ids) <= history_budget:
        return history_text # 无需截断
    
    kept_ids = history_ids[-history_budget:]
    truncated_history = tokenizer.decode(kept_ids, skip_special_tokens=True)
    
    return truncated_history

def main():
    args = parse_args()
    print(f"Config: {args}")

    # 1. Merge Model
    final_model_path = check_and_merge_model(args.base_model_path, args.model_path)
    
    # 2. Dataset
    # Parse target_metrics and target_windows from comma-separated strings
    target_metrics = None
    if args.target_metrics:
        target_metrics = [m.strip() for m in args.target_metrics.split(',')]
        print(f"Filtering metrics: {target_metrics}")
    
    target_windows = None
    if args.target_windows:
        target_windows = [w.strip() for w in args.target_windows.split(',')]
        print(f"Filtering windows: {target_windows}")
    
    dataset = RenjiDataset(
        root_dir=args.root_dir,
        split=args.split,
        max_samples=args.max_samples,
        table_mode="text_only",
        target_metrics=target_metrics,
        target_windows=target_windows
    )
    print(f"Found {len(dataset)} samples.")

    # 3. Prompts
    print("Preparing Prompts...")
    # Load tokenizer for Truncation calculation (CPU side)
    tokenizer = AutoTokenizer.from_pretrained(final_model_path, trust_remote_code=True, use_fast=True, fix_mistral_regex=True)
    
    prompts = []
    meta_list = []
    
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        
        # --- Truncation Logic ---
        instruction = sample['instruction']
        raw_input = sample['input']
        
        truncated_input = truncate_history_exact(
            instruction, 
            raw_input, 
            tokenizer, 
            max_len=args.max_model_len
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{truncated_input}\n{instruction}"}
        ]  
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        
        prompts.append(full_prompt)
        meta_list.append({
            'metric': sample['task_info']['task'].replace("Renji_", ""),
            'window': sample['task_info']['window'],
            'label': int(sample['output'])
        })

    # 4. vLLM
    print(f"Initializing vLLM with model: {final_model_path}")
    llm = LLM(
        model=final_model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.9,
    )
    
    sampling_params = SamplingParams(
        temperature=0, 
        max_tokens=1, 
        logprobs=20
    )

    print("Running Inference...")
    outputs = llm.generate(prompts, sampling_params)

    print("Processing Outputs...")
    vllm_tokenizer = llm.get_tokenizer()

    candidates_0 = ["0", " 0"]
    candidates_1 = ["1", " 1"]
    ids_0 = []
    ids_1 = []
    for c in candidates_0:
        tids = vllm_tokenizer.encode(c, add_special_tokens=False)
        if tids: ids_0.append(tids[-1])
    for c in candidates_1:
        tids = vllm_tokenizer.encode(c, add_special_tokens=False)
        if tids: ids_1.append(tids[-1])
    ids_0 = list(set(ids_0))
    ids_1 = list(set(ids_1))
    
    preds_score = []
    labels = []
    
    for i, output in enumerate(outputs):
        if not output.outputs:
            preds_score.append(0.5)
            labels.append(meta_list[i]['label'])
            continue
            
        top_logprobs = output.outputs[0].logprobs[0]
        prob_0_sum = 0.0
        prob_1_sum = 0.0
        for tid in ids_0:
            if tid in top_logprobs: prob_0_sum += np.exp(top_logprobs[tid].logprob)
        for tid in ids_1:
            if tid in top_logprobs: prob_1_sum += np.exp(top_logprobs[tid].logprob)
        total = prob_0_sum + prob_1_sum + 1e-12
        score = prob_1_sum / total
        preds_score.append(score)
        labels.append(meta_list[i]['label'])

    results_df = pd.DataFrame([
        {"metric": m['metric'], "window": m['window'], "label": l, "score": s}
        for m, l, s in zip(meta_list, labels, preds_score)
    ])
    
    try: global_auc = roc_auc_score(results_df['label'], results_df['score'])
    except: global_auc = 0.5
    print(f"Global AUC: {global_auc:.4f}")

    metrics_summary = []
    for (metric, win), group in results_df.groupby(['metric', 'window']):
        auc = np.nan
        if len(group['label'].unique()) > 1:
            try: auc = roc_auc_score(group['label'], group['score'])
            except: pass
        metrics_summary.append({"Metric": metric, "Window": win, "Count": len(group), "AUC": auc})
    
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df = metrics_df.sort_values(by=['Metric', 'Window'])
    
    print("\n=== Macro AUC per Window ===")
    for win, group in metrics_df.groupby('Window'):
        print(f"Window: {win}, Macro AUC: {group['AUC'].mean():.4f}")

    save_dir = os.path.join(args.model_path, "eval_vllm_results")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    metrics_df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
    print(f"Results saved to {save_dir}")

if __name__ == "__main__":
    main()
