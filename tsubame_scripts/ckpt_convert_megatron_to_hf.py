from megatron.bridge import AutoBridge

# 1) Create a bridge from a Hugging Face model (hub or local path)
bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)

# 2) Get a Megatron provider and configure parallelism before instantiation
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.finalize()
# 3) Materialize Megatron Core model(s)
model = provider.provide_distributed_model(wrap_with_ddp=False)

# 4a) Export Megatron → Hugging Face (full HF folder with config/tokenizer/weights)
bridge.save_hf_pretrained(model, "./hf_exports/llama32_1b")

# 4b) Or stream only weights (Megatron → HF)
for name, weight in bridge.export_hf_weights(model, cpu=True):
    print(name, tuple(weight.shape))