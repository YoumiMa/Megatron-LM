# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import sys
import os
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, LlamaConfig, AutoTokenizer
from contextlib import contextmanager


def add_arguments(parser):
    group = parser.add_argument_group(title='Llama3_hf saver.')
    group.add_argument('--hf-tokenizer-path', type=str, default=None,
                       help='Huggingface tokenizer path. eg. /models/llama-3-hf.')
    group.add_argument('--save-dtype', type=str, default='bfloat16')


@contextmanager
def suspend_nn_inits():
    """
    create context manager for loading without init
    see https://github.com/huggingface/transformers/issues/26258
    """
    skip = lambda *args, **kwargs: None  # noqa: E731
    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_   # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip   # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring


def save_checkpoint(queue: mp.Queue, args):
    # Search in directory above this
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 os.path.pardir,
                                                 os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print("Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)

    md = queue_get()

    # Verify compatibility of args
    assert hasattr(md, 'checkpoint_args')
    assert md.model_type == 'GPT'
    mag_conf = md.checkpoint_args

    if args.save_dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif args.save_dtype == 'float32':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    llama_conf = LlamaConfig(
        vocab_size=mag_conf.padded_vocab_size,
        hidden_size=mag_conf.hidden_size,
        intermediate_size=mag_conf.ffn_hidden_size,
        num_hidden_layers=mag_conf.encoder_num_layers,
        num_attention_heads=mag_conf.num_attention_heads,
        num_key_value_heads=mag_conf.num_query_groups,
        max_position_embeddings=mag_conf.max_position_embeddings,
        rms_norm_eps=mag_conf.norm_epsilon,
        bos_token_id=128000,
        eos_token_id=128001,
        tie_word_embeddings=not mag_conf.untie_embeddings_and_output_weights,
        rope_theta=mag_conf.rotary_base,
        attention_bias=mag_conf.add_bias_linear,
        torch_dtype=torch_dtype
    )

    state_dict = {}

    def set_hf_param(name, tensor: torch.Tensor):
        weight_name = f'{name}.weight'
        state_dict[weight_name] = tensor.to(torch.bfloat16)

    set_hf_param('model.embed_tokens', queue_get("embeddings")["word embeddings"])
    for i_layer in range(llama_conf.num_hidden_layers):
        message = queue_get(f"transformer layer {i_layer}")
        suffix = f'model.layers.{i_layer}.'
        set_hf_param(suffix + 'input_layernorm', message["input norm weight"])
        set_hf_param(suffix + 'post_attention_layernorm', message["post norm weight"])
        set_hf_param(suffix + 'mlp.gate_proj', message["mlp l0 weight W"])
        set_hf_param(suffix + 'mlp.up_proj', message["mlp l0 weight V"])
        
        H  = llama_conf.hidden_size
        A  = llama_conf.num_attention_heads
        G  = llama_conf.num_key_value_heads
        d  = H // A
        grp = A // G

        
        qkv_weight = message["qkv weight"]
        qkv_weight = qkv_weight.view(llama_conf.num_key_value_heads, -1, llama_conf.hidden_size)
        qkv_weight = torch.split(qkv_weight, [
            llama_conf.hidden_size // llama_conf.num_key_value_heads,
            llama_conf.hidden_size // llama_conf.num_attention_heads,
            llama_conf.hidden_size // llama_conf.num_attention_heads,
        ], dim=1)
        set_hf_param(suffix + 'self_attn.q_proj', qkv_weight[0].reshape(-1, llama_conf.hidden_size))
        set_hf_param(suffix + 'self_attn.k_proj', qkv_weight[1].reshape(-1, llama_conf.hidden_size))
        set_hf_param(suffix + 'self_attn.v_proj', qkv_weight[2].reshape(-1, llama_conf.hidden_size))
        
        # qkv = message["qkv weight"]                  # 期待: (3H, H) または (H, 3H)
        # print(f"layer {i_layer} qkv shape: {tuple(qkv.shape)}")              # (3H, H) に正規化
        # q_full, k_full, v_full = torch.chunk(qkv, 3, dim=0)  # 各 (H, H)
        # # K/V を (A, d, H) へ
        # kA = k_full.view(A, d, H)
        # vA = v_full.view(A, d, H)
        # # グループ単位に並べ替え: [G, grp, d, H]
        # kG = torch.stack([kA[g::G] for g in range(G)], dim=0)
        # vG = torch.stack([vA[g::G] for g in range(G)], dim=0)

        # # 同一性チェック（ここで差が出るなら切り出しが誤り）
        # eps = 1e-6
        # dk = (kG - kG[:, :1]).abs().max().item()
        # dv = (vG - vG[:, :1]).abs().max().item()
        # assert dk<1e-3 and dv<1e-3, f"dup_check max |ΔK|={dk} |ΔV|={dv}"
        # # A→Gに縮約
        # k_comp = kG[:, 0].contiguous().view(G*d, H)
        # v_comp = vG[:, 0].contiguous().view(G*d, H)

        # set_hf_param(suffix + 'self_attn.q_proj', q_full)     # (H,H)
        # set_hf_param(suffix + 'self_attn.k_proj', k_full)     # (G*d,H)
        # set_hf_param(suffix + 'self_attn.v_proj', v_full)     # (G*d,H)
        set_hf_param(suffix + 'self_attn.o_proj', message["dense weight"])
        set_hf_param(suffix + 'mlp.down_proj', message["mlp l1 weight"])
        
        # ループ内の最後で
        # verify_layer(i_layer, suffix, message, state_dict, H, A, G, d)
        
    set_hf_param('model.norm', queue_get('final norm')['weight'])
    set_hf_param('lm_head', queue_get('output layer')['weight'])

    with suspend_nn_inits():
        print("Saving model to disk ...")
        model = AutoModelForCausalLM.from_config(llama_conf)
        model.load_state_dict(state_dict, strict=False)
        model.save_pretrained(
            args.save_dir,
            safe_serialization=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_tokenizer_path
    )
    tokenizer.save_pretrained(args.save_dir)