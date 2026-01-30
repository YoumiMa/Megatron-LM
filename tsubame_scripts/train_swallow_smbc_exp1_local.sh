MASTER_ADDR=localhost
MASTER_PORT=29500

NODE_TYPE="h100"
export NUM_GPU_PER_NODE=$(nvidia-smi -L | wc -l)

NUM_NODES=1
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

# model config
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
TRAIN_STEPS=3600

LR=1e-5
MIN_LR=1e-6
LR_WARMUP_STEPS=360
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=tokyotech-llm/Llama-3.1-Swallow-8B-v0.5
CHECKPOINT_DIR=/gs/bs/tga-okazaki/ma/cache/Llama-3.1-Swallow-8B-v0.5/megatron_tp1_pp2/
CHECKPOINT_SAVE_DIR=/gs/bs/tga-ma/ma/ckpts/llama-3.1-swallow-8B-v0.5-megatron_tp${TENSOR_PARALLEL_SIZE}_pp${PIPELINE_PARALLEL_SIZE}_LR${LR}_exp1/

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR="/gs/bs/tga-okazaki/ma/data/smbcgic_processed"

TRAIN_DATA_PATH=""

# smbc data
# DATASET_DIR配下の全てのサブディレクトリを追加
for FILE in "${DATASET_DIR}"/*; do
    if [[ "$FILE" == *.idx ]]; then
            BASENAME=$(basename "$FILE")
            
            # Remove _text_document.idx suffix
            NAME="${BASENAME%_text_document.idx}"
            # echo "Found dataset: $NAME"
            
            # Add to blended dataset path with weight 1
            TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1 ${DATASET_DIR}/${NAME}_text_document"
        fi
done

echo "TRAIN_DATA_PATH=$TRAIN_DATA_PATH"

# job name
JOB_NAME="Llama-3.1-Swallow-8b-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-exp1"

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
cd ~/fs/Megatron-LM/

CONTAINER_IMAGE="/gs/fs/tga-ma/ma/megatron-container"
# run
  apptainer run --nv \
  --env WORLD_SIZE=$WORLD_SIZE \
  --env MASTER_ADDR=$MASTER_ADDR \
  --env MASTER_PORT=$MASTER_PORT \
  -w -B /gs -B /apps -B /home -B /gs/fs/tga-okazaki/ma:/root ${CONTAINER_IMAGE} \
  torchrun --nproc_per_node=${NUM_GPUS} \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
  pretrain_gpt.py \
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
  --split 970,30,0 \
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
  --log-interval 10 \
  --log-progress \
  --save-interval 600 \
  --eval-interval 10 \
  --eval-iters 1 \
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
