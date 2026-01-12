from megatron.bridge import AutoBridge

# 1) Create a bridge from a Hugging Face model (hub or local path)
model = AutoBridge.load_megatron_model("/gs/bs/tga-okazaki/ma/ckpts/llama-3.1-8B-megatron_tp$1_pp2_LR1e-4/")

# 4a) Export Megatron → Hugging Face (full HF folder with config/tokenizer/weights)
bridge.save_hf_pretrained(model, "/gs/bs/tga-okazaki/ma/ckpts/llama-3.1-8B-megatron_tp$1_pp2_LR1e-4_hf/")

# 4b) Or stream only weights (Megatron → HF)
for name, weight in bridge.export_hf_weights(model, cpu=True):
    print(name, tuple(weight.shape))