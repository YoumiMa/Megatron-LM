#!/bin/bash
#! /bin/sh
#PBS -P gcb50243  
#PBS -q rt_HF
#PBS -l select=1:mpiprocs=8  -k oe
#PBS -l walltime=00:10:00
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

NUM_NODES=$(($(wc -l < "$PBS_NODEFILE") / ${NUM_GPU_PER_NODE}))
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

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
SEQ_LENGTH=8192

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=2 # num layers 32: Llama-3 8B
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))


# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=25000

LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=meta-llama/Meta-Llama-3.1-8B
CHECKPOINT_DIR=/groups/gcb50243/ma/cache/llama-3.1-8B/megatron_tp{$TENSOR_PARALLEL_SIZE}_pp{$PIPELINE_PARALLEL_SIZE}
CHECKPOINT_SAVE_DIR=/groups/gcb50243/ma/ckpts/llama-3.1-8B-megatron_tp{$TENSOR_PARALLEL_SIZE}_pp{$PIPELINE_PARALLEL_SIZE}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR=/groups/gcb50243/ma/data/japanese-wikipedia/processed

TRAIN_DATA_PATH=""

# japanese wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1 ${DATASET_DIR}/ja_wiki_text_document"

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
echo "hostfile: ${PBS_NODEFILE}"

ldd $(which python) | grep mpi
python -c "import mpi4py; import mpi4py.MPI as MPI; print(MPI.Get_library_version())"

mpirun -np $WORLD_SIZE \
  --display-allocation \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x LD_LIBRARY_PATH \
  -x NCCL_P2P_LEVEL=NVL \
  -x PATH \
  -bind-to none \
  python ~/test.py \
