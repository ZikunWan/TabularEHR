import json
import os
import torch
from peft import PeftModel
from transformers import AutoModel
from models.encoder_classifier import LongTableEncoderClassifier
from models.TableEncoder.config import TableEncoderConfig


def infer_pretrained_dim_out(pretrained_path: str):
    if not pretrained_path:
        return None

    config_path = None
    if os.path.isdir(pretrained_path):
        candidate = os.path.join(pretrained_path, "config.json")
        if os.path.exists(candidate):
            config_path = candidate
    else:
        candidate = os.path.join(os.path.dirname(pretrained_path), "config.json")
        if os.path.exists(candidate):
            config_path = candidate

    if config_path:
        try:
            with open(config_path, 'r') as f:
                dim_out = json.load(f).get('dim_out')
            if dim_out is not None:
                return dim_out
        except Exception as exc:
            print(f"Warning: failed to read dim_out from {config_path}: {exc}")

    ckpt_path = pretrained_path
    if os.path.isdir(pretrained_path):
        safetensors_files = [f for f in os.listdir(pretrained_path) if f.endswith('.safetensors')]
        bin_files = [f for f in os.listdir(pretrained_path) if f.endswith('.bin')]
        if safetensors_files:
            ckpt_path = os.path.join(pretrained_path, 'model.safetensors' if 'model.safetensors' in safetensors_files else safetensors_files[0])
        elif bin_files:
            ckpt_path = os.path.join(pretrained_path, 'pytorch_model.bin' if 'pytorch_model.bin' in bin_files else bin_files[0])
        else:
            return None

    def _match_dim_out_key(keys):
        for key in keys:
            if key.endswith('qformer.out_proj.weight'):
                return key
        return None

    try:
        if ckpt_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

        key = _match_dim_out_key(state_dict.keys())
        if key is not None:
            return int(state_dict[key].shape[0])
    except Exception as exc:
        print(f"Warning: failed to infer dim_out from {ckpt_path}: {exc}")

    return None


def load_bert_attn_to_2d_grid(model: LongTableEncoderClassifier, pretrained_model_name: str) -> LongTableEncoderClassifier:
    """Init 2d_grid attention weights from a BERT-compatible model (TAPAS, TaBERT)."""
    print(f"Loading '{pretrained_model_name}' for 2d_grid weight init...")
    bert = AutoModel.from_pretrained(pretrained_model_name)
    bert_sd, model_sd = bert.state_dict(), model.state_dict()
    new_sd, loaded = {}, []

    for i in range(len(model.encoder.layers)):
        bp, op = f"encoder.layer.{i}", f"encoder.layers.{i}"

        q = bert_sd.get(f"{bp}.attention.self.query.weight")
        k = bert_sd.get(f"{bp}.attention.self.key.weight")
        v = bert_sd.get(f"{bp}.attention.self.value.weight")
        if q is not None:
            qkv_w = torch.cat([q, k, v], dim=0)
            for attn in ("intra_attn", "inter_attn"):
                key = f"{op}.{attn}.qkv.weight"
                if key in model_sd:
                    new_sd[key] = qkv_w.clone(); loaded.append(key)

        proj_w = bert_sd.get(f"{bp}.attention.output.dense.weight")
        proj_b = bert_sd.get(f"{bp}.attention.output.dense.bias")
        for attn in ("intra_attn", "inter_attn"):
            for sfx, val in [(".proj.weight", proj_w), (".proj.bias", proj_b)]:
                key = f"{op}.{attn}{sfx}"
                if val is not None and key in model_sd:
                    new_sd[key] = val.clone(); loaded.append(key)

        for ln_key, norms in [
            (f"{bp}.attention.output.LayerNorm.weight", ("norm1_intra", "norm1_inter")),
            (f"{bp}.output.LayerNorm.weight",           ("norm2",)),
        ]:
            w = bert_sd.get(ln_key)
            if w is not None:
                for norm in norms:
                    key = f"{op}.{norm}.weight"
                    if key in model_sd:
                        new_sd[key] = w.clone(); loaded.append(key)

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"  Loaded {len(loaded)} keys from BERT base | Missing (expected): {len(missing)} | Unexpected: {len(unexpected)}")
    del bert
    return model


def merge_hf_trainer_lora_state_dict(model: LongTableEncoderClassifier, state_dict: dict):
    merged_sd = {}
    for k, v in state_dict.items():
        if '.modules_to_save.default.' in k:
            new_k = k.replace('.modules_to_save.default.', '.')
            merged_sd[new_k] = v
            continue
        if '.original_module.' in k:
            continue
        if '.lora_A.' in k or '.lora_B.' in k:
            continue
        if '.base_layer.' in k:
            prefix = k.split('.base_layer.')[0]     # e.g. "encoder.layers.0.attn.qkv"
            param  = k.split('.base_layer.')[1]     # e.g. "weight"
            base_k = f"{prefix}.{param}"
            lora_a_k = f"{prefix}.lora_A.default.weight"
            lora_b_k = f"{prefix}.lora_B.default.weight"
            if param == 'weight' and lora_a_k in state_dict and lora_b_k in state_dict:
                r       = state_dict[lora_a_k].shape[0]
                scaling = float(r * 2) / float(r)  # Assume default alpha=2*r -> scaling=2
                merged_sd[base_k] = v.float() + scaling * (state_dict[lora_b_k].float() @ state_dict[lora_a_k].float())
            else:
                merged_sd[base_k] = v
            continue
        merged_sd[k] = v
    
    missing, unexpected = model.load_state_dict(merged_sd, strict=False)
    encoder_missing = [k for k in missing if 'encoder' in k]
    print(f"  Merged HF Trainer flat LoRA weights. Missing encoder keys: {len(encoder_missing)} (should be 0)")
    return model


def _load_raw_state_dict(model: LongTableEncoderClassifier, ckpt_path: str):
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location='cpu')

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    # Strip DDP "module." prefix
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    # Check for flat HF LoRA state_dict
    has_peft_keys = any('.lora_A.' in k or '.base_layer.' in k for k in state_dict)
    if has_peft_keys:
        print("Detected peft-format state dict (base_layer + lora_* keys). Merging LoRA weights directly...")
        return merge_hf_trainer_lora_state_dict(model, state_dict)

    model_state = model.state_dict()
    # Remap encoder-only checkpoints onto the classifier's encoder.* namespace.
    # This covers both older "tabular_model.*" checkpoints and raw encoder
    # checkpoints that save keys like "embedding.*" / "layers.*".
    new_state_dict = {}
    remapped_keys = 0
    for k, v in state_dict.items():
        if k.startswith('tabular_model.'):
            new_key = 'encoder.' + k[len('tabular_model.'):]
        elif k not in model_state and ('encoder.' + k) in model_state:
            new_key = 'encoder.' + k
        else:
            new_key = k
        if new_key != k:
            remapped_keys += 1
        new_state_dict[new_key] = v

    # Shape filtering
    shape_skipped, compatible_state_dict = [], {}
    for k, v in new_state_dict.items():
        if k in model_state and getattr(model_state[k], 'shape', None) != getattr(v, 'shape', None):
            shape_skipped.append(f"{k}: ckpt{tuple(v.shape)} vs model{tuple(model_state[k].shape)}")
        else:
            compatible_state_dict[k] = v

    if shape_skipped:
        print(f"  ⚠️  Skipping {len(shape_skipped)} shape-mismatched keys (will be randomly initialized):")
        for s in shape_skipped[:5]: print(f"       {s}")
        if len(shape_skipped) > 5: print(f"       ... and {len(shape_skipped) - 5} more.")

    matched_keys = [k for k in compatible_state_dict if k in model_state]
    if remapped_keys:
        print(f"  Remapped {remapped_keys} checkpoint keys into the classifier encoder namespace.")
    if not matched_keys:
        print("  No checkpoint tensors matched model parameters after remapping.")

    missing, unexpected = model.load_state_dict(compatible_state_dict, strict=False)
    print(
        f"  Loaded raw state_dict | Matched: {len(matched_keys)} | "
        f"Missing: {len(missing)} | Unexpected: {len(unexpected)}"
    )
    if missing:
        print(f"  Missing sample: {missing[:5]}")
    if unexpected:
        print(f"  Unexpected sample: {unexpected[:5]}")
    return model


def load_model_weights(
    model: LongTableEncoderClassifier,
    pretrained_path: str = None,
    use_lora: bool = False,
    is_trainable: bool = False
):
    if not pretrained_path:
        return model

    if not os.path.exists(pretrained_path) and (pretrained_path.startswith("google/") or pretrained_path.startswith("bert-")):
        return load_bert_attn_to_2d_grid(model, pretrained_path)

    ckpt_path = pretrained_path
    if os.path.isdir(pretrained_path):
        safetensors_files = [f for f in os.listdir(pretrained_path) if f.endswith(".safetensors")]
        bin_files = [f for f in os.listdir(pretrained_path) if f.endswith(".bin")]
        
        if safetensors_files:
            ckpt_path = os.path.join(pretrained_path, "model.safetensors" if "model.safetensors" in safetensors_files else safetensors_files[0])
        elif bin_files:
            ckpt_path = os.path.join(pretrained_path, "pytorch_model.bin" if "pytorch_model.bin" in bin_files else bin_files[0])
        elif not os.path.exists(os.path.join(pretrained_path, "adapter_config.json")):
             print(f"Warning: No weights found in {pretrained_path}, returning raw model.")
             return model

    has_adapter_config = os.path.exists(os.path.join(pretrained_path, "adapter_config.json"))
    
    if use_lora or has_adapter_config:
        print(f"Loading PEFT adapter from {pretrained_path}...")
        model = PeftModel.from_pretrained(model, pretrained_path, is_trainable=is_trainable)
        
        if not is_trainable:
            model = model.merge_and_unload()
            print("  LoRA adapters merged into base model for inference.")
        
        return model

    print(f"Loading full checkpoint from {ckpt_path}...")
    return _load_raw_state_dict(model, ckpt_path)
