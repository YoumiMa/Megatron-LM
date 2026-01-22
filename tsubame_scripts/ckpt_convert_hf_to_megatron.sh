#!/bin/bash

# distributed settings
TENSOR_PARALLEL_SIZE=1   # fixed
PIPELINE_PARALLEL_SIZE=2

# model config
HF_FORMAT_DIR=/gs/bs/tga-okazaki/ma/cache/Llama-3.1-Swallow-8B-v0.5
MEGATRON_FORMAT_DIR=/gs/bs/tga-okazaki/ma/cache/Llama-3.1-Swallow-8B-v0.5/megatron_tp${TENSOR_PARALLEL_SIZE}_pp${PIPELINE_PARALLEL_SIZE}

mkdir -p ${MEGATRON_FORMAT_DIR}

# tokenizer config
TOKENIZER_MODEL="tokyotech-llm/Llama-3.1-Swallow-8B-v0.5"
cd ~/Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc_per_node=2 \
    tools/checkpoint/convert.py \
    --model-type GPT \
    --loader llama_mistral \
    --model-size llama3 \
    --checkpoint-type hf \
    --load-dir ${HF_FORMAT_DIR} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --saver mcore \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --target-tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --target-pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
    --bf16
