#!/usr/bin/env python3
# convert_torch_dist_to_hf.py
import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM

def init_distributed():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='gloo', world_size=1, rank=0)

def convert_megatron_to_hf(state_dict, num_layers=32, hidden_size=4096, num_heads=32, num_kv_heads=8):
    """Megatron-Core -> HuggingFace Llama"""
    hf = {}
    head_dim = hidden_size // num_heads
    
    for key, value in state_dict.items():
        # Embedding
        if 'embedding.word_embeddings.weight' in key:
            hf['model.embed_tokens.weight'] = value
        # LM Head
        elif 'output_layer.weight' in key:
            hf['lm_head.weight'] = value
        # Final LayerNorm
        elif 'decoder.final_layernorm.weight' in key:
            hf['model.norm.weight'] = value
        # Layers
        elif 'decoder.layers.' in key:
            parts = key.split('.')
            layer_idx = int(parts[2])
            prefix = f'model.layers.{layer_idx}'
            
            if 'self_attention.linear_qkv.weight' in key:
                q_size = num_heads * head_dim
                kv_size = num_kv_heads * head_dim
                q, k, v = torch.split(value, [q_size, kv_size, kv_size], dim=0)
                hf[f'{prefix}.self_attn.q_proj.weight'] = q
                hf[f'{prefix}.self_attn.k_proj.weight'] = k
                hf[f'{prefix}.self_attn.v_proj.weight'] = v
            elif 'self_attention.linear_proj.weight' in key:
                hf[f'{prefix}.self_attn.o_proj.weight'] = value
            elif 'mlp.linear_fc1.weight' in key:
                mid = value.shape[0] // 2
                hf[f'{prefix}.mlp.gate_proj.weight'] = value[:mid]
                hf[f'{prefix}.mlp.up_proj.weight'] = value[mid:]
            elif 'mlp.linear_fc2.weight' in key:
                hf[f'{prefix}.mlp.down_proj.weight'] = value
            elif 'input_layernorm.weight' in key:
                hf[f'{prefix}.input_layernorm.weight'] = value
            elif 'pre_mlp_layernorm.weight' in key:
                hf[f'{prefix}.post_attention_layernorm.weight'] = value
    
    return hf

def main():
    ckpt_path = Path("/gs/bs/tga-okazaki/ma/ckpts/llama-3.1-8B-megatron_tp1_pp2_LR1e-4/iter_0002400")
    hf_output = Path("/gs/bs/tga-okazaki/ma/ckpts/llama-3.1-8B-megatron_tp1_pp2_LR1e-4_hf")
    hf_tokenizer = "/gs/bs/tga-okazaki/ma/cache/Llama-3.1-8B"
    
    init_distributed()
    
    # PyTorch dcpでロード（空のstate_dictを渡すとメタデータからキーを推論）
    print("Loading checkpoint...")
    
    # まずメタデータからキー一覧を取得
    import pickle
    with open(ckpt_path / ".metadata", "rb") as f:
        metadata = pickle.load(f)
    
    print(f"Keys in checkpoint: {len(metadata.state_dict_metadata)}")
    
    # state_dictを構築
    state_dict = {}
    for key in metadata.state_dict_metadata.keys():
        state_dict[key] = torch.zeros(1)  # placeholder
    
    dcp.load(
        state_dict=state_dict,
        storage_reader=dcp.FileSystemReader(str(ckpt_path)),
    )
    
    print(f"Loaded {len(state_dict)} tensors")
    print("Sample keys:")
    for k in list(state_dict.keys())[:15]:
        v = state_dict[k]
        print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
    
    # HF形式に変換
    print("\nConverting to HF format...")
    hf_state_dict = convert_megatron_to_hf(state_dict)
    print(f"Converted {len(hf_state_dict)} keys")
    
    # 保存
    print("\nLoading HF model config...")
    config = AutoConfig.from_pretrained(hf_tokenizer)
    model = LlamaForCausalLM(config)
    
    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print(f"  Missing samples: {missing[:5]}")
    
    hf_output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(hf_output)
    
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
    tokenizer.save_pretrained(hf_output)
    
    print(f"\nSaved to: {hf_output}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()