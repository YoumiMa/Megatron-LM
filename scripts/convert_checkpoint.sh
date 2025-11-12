#!/bin/bash

# distributed settings
TENSOR_PARALLEL_SIZE=1   # fixed
PIPELINE_PARALLEL_SIZE=2

# model config
HF_FORMAT_DIR=/groups/gcb50243/ma/cache/llama-3.1-8B
MEGATRON_FORMAT_DIR=/groups/gcb50243/ma/cache/llama-3.1-8B/megatron_tp${TENSOR_PARALLEL_SIZE}_pp${PIPELINE_PARALLEL_SIZE}

mkdir -p ${MEGATRON_FORMAT_DIR}

# tokenizer config
TOKENIZER_MODEL="meta-llama/Llama-3.1-8B"
cd ~/Megatron-LM
MODEL_SIZE="llama3"

python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader llama_mistral \
    --load-dir ${HF_FORMAT_DIR} \
    --model-size ${MODEL_SIZE} \
    --checkpoint-type hf \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --saver core \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --target-tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --target-pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
    --bf16
