#!/bin/bash

# distributed settings
TENSOR_PARALLEL_SIZE=1   # fixed
PIPELINE_PARALLEL_SIZE=2

# model config
MEGATRON_FORMAT_DIR=/gs/bs/tga-okazaki/ma/ckpts/llama-3.1-8B-megatron_tp${TENSOR_PARALLEL_SIZE}_pp${PIPELINE_PARALLEL_SIZE}_LR1e-4/
HF_FORMAT_DIR=/gs/bs/tga-okazaki/ma/ckpts/llama-3.1-8B-megatron_tp${TENSOR_PARALLEL_SIZE}_pp${PIPELINE_PARALLEL_SIZE}_LR1e-4_hf/

mkdir -p ${HF_FORMAT_DIR}

cd ~/Megatron-LM
PROCS=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))
echo $PROCS

torchrun --nproc_per_node=$PROCS \
    tools/checkpoint/convert.py \
     --model-type GPT \
     --loader core \
     --saver llama3_hf \
     --load-dir ${MEGATRON_FORMAT_DIR} \
     --save-dir ${HF_FORMAT_DIR} \
     --megatron-path . \
     --hf-tokenizer-path /gs/bs/tga-okazaki/ma/cache/Llama-3.1-8B
