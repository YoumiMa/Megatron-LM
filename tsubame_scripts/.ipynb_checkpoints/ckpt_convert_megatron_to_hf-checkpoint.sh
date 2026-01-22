#!/bin/bash

# distributed settings
TENSOR_PARALLEL_SIZE=1   # fixed
PIPELINE_PARALLEL_SIZE=2

MODEL_PATH=/gs/bs/tga-ma/ma/ckpts/llama-3.1-swallow-8B-v0.5-megatron_tp${TENSOR_PARALLEL_SIZE}_pp${PIPELINE_PARALLEL_SIZE}_LR1e-5_exp1/
ITER=0000600

# model config
TORCH_FORMAT_DIR=${MODEL_PATH}/torch
HF_FORMAT_DIR=${MODEL_PATH}/hf/iter_${ITER}
ls $TORCH_FORMAT_DIR
# TORCH_FORMAT_DIR=/gs/bs/tga-okazaki/ma/cache/Llama-3.1-Swallow-8B-v0.5/megatron_tp${TENSOR_PARALLEL_SIZE}_pp${PIPELINE_PARALLEL_SIZE}
# HF_FORMAT_DIR=/gs/bs/tga-okazaki/ma/cache/Llama-3.1-Swallow-8B-v0.5/megatron_tp${TENSOR_PARALLEL_SIZE}_pp${PIPELINE_PARALLEL_SIZE}/converted_hf/

mkdir -p ${HF_FORMAT_DIR}

cd ~/Megatron-LM
PROCS=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))
echo $PROCS

torchrun --nproc_per_node=$PROCS \
    tools/checkpoint/convert.py \
     --model-type GPT \
     --loader core \
     --saver llama3_hf \
     --load-dir ${TORCH_FORMAT_DIR} \
     --save-dir ${HF_FORMAT_DIR} \
     --megatron-path . \
     --hf-tokenizer-path /gs/bs/tga-okazaki/ma/cache/Llama-3.1-Swallow-8B-v0.5
