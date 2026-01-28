#!/usr/bin/env python3
# compare_logits.py
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

def compare_logits():
    # converted_path = "/gs/bs/tga-okazaki/ma/cache/Llama-3.1-8B/megatron_tp1_pp2/hf"
    converted_path = "/gs/bs/tga-ma/ma/ckpts/llama-3.1-swallow-8B-v0.5-megatron_tp1_pp2_LR1e-5_exp1/hf/iter_0003600/"
    # original_path = "/gs/bs/tga-okazaki/ma/cache/Llama-3.1-Swallow-8B-v0.5"
    
    tokenizer = AutoTokenizer.from_pretrained(converted_path)
    
    # 両モデルをロード（同じdtype）
    converted = LlamaForCausalLM.from_pretrained(converted_path, torch_dtype=torch.float32, device_map="cpu")
    # original = LlamaForCausalLM.from_pretrained(original_path, torch_dtype=torch.float32, device_map="cpu")
    
    # テスト入力
    texts = [
        "The capital of France is",
        "1 + 1 =",
        "Hello, my name is",
    ]
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            conv_out = converted(**inputs)
            # orig_out = original(**inputs)
        
        conv_logits = conv_out.logits[0, -1]  # 最後のトークンのlogits
        # orig_logits = orig_out.logits[0, -1]
        
        # Top-5予測を確認
        top5 = torch.topk(conv_logits, 5)
        # top5 = torch.topk(orig_logits, 5)
        print(f"\nInput: '{text}'")
        print("Top 5 predictions:")
        for i, (idx, score) in enumerate(zip(top5.indices, top5.values)):
            token = tokenizer.decode([idx])
            print(f"  {i+1}. '{token}' ({score:.2f})")

if __name__ == "__main__":
    compare_logits()