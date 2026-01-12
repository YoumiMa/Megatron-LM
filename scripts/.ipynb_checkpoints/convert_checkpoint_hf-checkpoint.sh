#!/bin/bash


# distributed settings
TENSOR_PARALLEL_SIZE=1   # fixed
PIPELINE_PARALLEL_SIZE=2

# model config
MEGATRON_FORMAT_DIR=/groups/gcb50243/ma/ckpts/llama-3.1-8B-megatron_tp${TENSOR_PARALLEL_SIZE}_pp${PIPELINE_PARALLEL_SIZE}_LR1e-4_torch/torch
HF_FORMAT_DIR=/groups/gcb50243/ma/ckpts/llama-3.1-8B-megatron_tp${TENSOR_PARALLEL_SIZE}_pp${PIPELINE_PARALLEL_SIZE}_LR1e-4_torch/hf

mkdir -p ${HF_FORMAT_DIR}

cd ~/Megatron-LM
PROCS=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))
echo $PROCS

python tools/checkpoint/convert.py \
     --model-type GPT \
     --loader core \
     --saver llama3_hf \
     --load-dir ${MEGATRON_FORMAT_DIR} \
     --save-dir ${HF_FORMAT_DIR} \
     --megatron-path . \
     --hf-tokenizer-path /groups/gcb50243/ma/cache/llama-3.1-8B
