import os, yaml, torch, glob

ITER_DIR = "/home/acb11709gz/disk/ckpts/llama-3.1-8B-megatron_tp1_pp2_LR5e-5_torch/torch/iter_0000500/"
pt = torch.load(glob.glob(os.path.join(ITER_DIR, "mp_rank_00*", "model_optim_rng.pt"))[0], map_location="cpu", weights_only=False)
args = pt.get("checkpoint_args", pt.get("args"))

G = getattr(args, "num_query_groups", getattr(args, "num_key_value_heads", 1))

cfg = {
  "version": 1,
  "model": {
    "type": "gpt",
    "name": "llama3",
    "architecture": {
      "hidden_size": args.hidden_size,
      "ffn_hidden_size": args.ffn_hidden_size,
      "num_layers": args.encoder_num_layers,
      "num_attention_heads": args.num_attention_heads,
      "num_query_groups": G,
      "max_position_embeddings": args.max_position_embeddings,
      "rotary_base": getattr(args, "rotary_base", 500000),
      "attention_bias": getattr(args, "add_bias_linear", False),
      "normalization": "RMSNorm",
      "bos_token_id": 128000,
      "eos_token_id": 128001,
      "vocab_size": args.padded_vocab_size,
      "tie_word_embeddings": not getattr(args, "untie_embeddings_and_output_weights", False),
      "params_dtype": "bfloat16" if getattr(args, "bf16", True) else "float16",
    },
  },
  "megatron_config": {
    "tensor_model_parallel_size": getattr(args, "tensor_model_parallel_size", 1),
    "pipeline_model_parallel_size": getattr(args, "pipeline_model_parallel_size", 1),
    "use_mcore_models": True,
    "params_dtype": "bfloat16" if getattr(args, "bf16", True) else "float16",
    "seed": getattr(args, "seed", 1234),
  },
}


RUN_DIR = os.path.dirname(ITER_DIR)  # 1つ上に出す
with open(os.path.join(RUN_DIR, "run_config.yaml"), "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("wrote:", os.path.join(RUN_DIR, "run_config.yaml"))