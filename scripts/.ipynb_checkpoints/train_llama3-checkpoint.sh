#!/bin/bash
#! /bin/sh
#PBS -P gch51650  
#PBS -q rt_HF
#PBS -l select=2:mpiprocs=8  -k oe
#PBS -l walltime=12:00:00
#PBS -v USE_SSH=1

# module load
source /etc/profile.d/modules.sh

module load cuda/12.8/12.8.1
module load python/3.12/3.12.9  
module load cudnn/9.10/9.10.2
module load nccl/2.23/2.23.4-1
module load hpcx/2.20
module load gcc/13.2.0

# python virtualenv
source venv/cpt/bin/activate

pip install pybind11
# distributed settings
JOBID=${PBS_JOBID%%.*}
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOBID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile

if [[ "$PBS_QUEUE" == "rt_HF" ]]; then
  export NUM_GPU_PER_NODE=8
  NODE_TYPE="h200"
elif [[ "$PBS_QUEUE" == "rt_HG" ]]; then
  export NUM_GPU_PER_NODE=1
  NODE_TYPE="h200"
else
  echo "Unrecognized PBS_QUEUE: $PBSclear
  _QUEUE"
fi

NUM_GPUS=$(wc -l < "$PBS_NODEFILE")
NUM_NODES=$((${NUM_GPUS} / ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOBID}

while read -r line; do
  echo "${line}"
done <"$PBS_NODEFILE" >"$HOSTFILE_NAME"

# model config
# llama-2-7b: https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336 # intermediate size (HuggingFace)
NUM_LAYERS=32
NUM_HEADS=32
NUM_QUERY_GROUPS=8
SEQ_LENGTH=8192

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=2 # num layers 32: Llama-3 8B
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

echo $DATA_PARALLEL_SIZE

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=512
TRAIN_STEPS=500

LR=1e-4
MIN_LR=1e-5
LR_WARMUP_STEPS=50
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=meta-llama/Meta-Llama-3.1-8B
CHECKPOINT_DIR=/groups/gcb50243/ma/cache/llama-3.1-8B/megatron_tp1_pp2/
CHECKPOINT_SAVE_DIR=/groups/gcb50243/ma/ckpts/llama-3.1-8B-megatron_tp${TENSOR_PARALLEL_SIZE}_pp${PIPELINE_PARALLEL_SIZE}_LR${LR}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR=/groups/gcb50243/ma/data/japanese-wikipedia/processed

TRAIN_DATA_PATH=""

# japanese wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1 ${DATASET_DIR}/ja_wiki_text_document"

echo ${TRAIN_DATA_PATH}
# job name
JOB_NAME="Llama-3-8b-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu"

# checkpoint load
if [ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
else
  # first training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_DIR} --no-load-rng --no-load-optim"
fi

echo "The CHECKPOINT ARG is:  ${CHECKPOINT_ARGS}"
export WORLD_SIZE=$NUM_GPUS
echo "world size is: ${WORLD_SIZE}"
echo "hostfile: $HOSTFILE_NAME"
cd Megatron-LM/
# run
mpirun -np $WORLD_SIZE \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $PBS_NODEFILE \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x LD_LIBRARY_PATH \
  -x NCCL_P2P_LEVEL=NVL \
  -x PATH \
  -bind-to none \
  python pretrain_gpt.py \
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --use-distributed-optimizer \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_HEADS} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  ${CHECKPOINT_ARGS} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path ${TRAIN_DATA_PATH} \
  --distributed-backend nccl \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --weight-decay ${WEIGHT_DECAY} \
  --clip-grad ${GRAD_CLIP} \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --log-interval 1 \
  --log-progress \
  --save-interval 100 \
  --eval-interval 100 \
  --eval-iters 10 \
  --bf16 \
  --untie-embeddings-and-output-weights \
  --use-rotary-position-embeddings \
  --position-embedding-type rope \
  --use-mcore-models \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --no-masked-softmax-fusion \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --disable-bias-linear \
  --no-bias-gelu-fusion \
  --swiglu \
  --use-flash-attn \
  --recompute-activations \
  --recompute-granularity "selective" \
  --use-mpi \
  --log-throughput \
  --wandb-exp-name ${JOB_NAME} \
  --wandb-project "llm-cpt" \
  --exit-on-missing-checkpoint \
  --use-checkpoint-args \
  --transformer-impl transformer_engine \
  --group-query-attention \
  --rotary-base 500000 \
  --rotary-percent 1.0 \
  --use-rope-scaling \
