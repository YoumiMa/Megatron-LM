#!/usr/bin/env python3
# compare_hf_models.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from transformers import LlamaForCausalLM

def compare():
    converted_path = "/gs/bs/tga-okazaki/ma/cache/Llama-3.1-Swallow-8B-v0.5/megatron_tp1_pp2/converted_hf"
    original_path = "/gs/bs/tga-okazaki/ma/cache/Llama-3.1-Swallow-8B-v0.5"
    
    print("Loading converted model...")
    converted = LlamaForCausalLM.from_pretrained(converted_path, torch_dtype=torch.float32)
    conv_state = converted.state_dict()
    
    print("Loading original model...")
    original = LlamaForCausalLM.from_pretrained(original_path, torch_dtype=torch.float32)
    orig_state = original.state_dict()
    
    print("\n=== Comparing weights ===\n")
    
    all_close = True
    for key in orig_state.keys():
        if key not in conv_state:
            print(f"{key}: MISSING in converted")
            all_close = False
            continue
        
        orig_w = orig_state[key]
        conv_w = conv_state[key]
        
        if orig_w.shape != conv_w.shape:
            print(f"{key}: shape mismatch {orig_w.shape} vs {conv_w.shape}")
            all_close = False
            continue
        
        diff = (orig_w - conv_w).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        match = max_diff < 1e-5
        all_close &= match
        
        # 差分がある場合のみ表示
        if not match or 'layers.0.' in key or 'layers.31.' in key or 'embed' in key or 'lm_head' in key:
            status = '✓' if match else '✗'
            print(f"{key}: {status} max={max_diff:.2e} mean={mean_diff:.2e}")
    
    print(f"\n{'='*50}")
    print(f"Result: {'✓ Models match!' if all_close else '✗ Models differ (expected after continued pretraining)'}")

if __name__ == "__main__":
    compare()