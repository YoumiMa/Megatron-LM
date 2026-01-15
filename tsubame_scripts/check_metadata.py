import torch
import pickle

# common.ptまたはチェックポイントのargsを確認
ckpt_path = "/gs/bs/tga-okazaki/ma/ckpts/llama-3.1-8B-megatron_tp1_pp2_LR1e-4_test/iter_0000005/common.pt"
data = torch.load(ckpt_path, map_location='cpu', weights_only=False)

print("Keys:", data.keys())
if 'args' in data:
    args = data['args']
    print(f"rope_theta: {getattr(args, 'rope_theta', 'NOT FOUND')}")
    print(f"rope_scaling: {getattr(args, 'rope_scaling', 'NOT FOUND')}")
    print(f"rotary_base: {getattr(args, 'rotary_base', 'NOT FOUND')}")