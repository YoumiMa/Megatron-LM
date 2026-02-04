#!/bin/bash

# テキストファイルのパス
OUTPUT_ROOT="/gs/bs/tga-okazaki/ma/data/smbcgic_processed/"
mkdir -p $OUTPUT_ROOT

BASE_PATH="/gs/bs/tga-okazaki/ma/data/smbcgic/"
CONTAINER_IMAGE="/gs/fs/tga-ma/ma/megatron-container"

for DIR in "$BASE_PATH"*/; do
    FILE_PATHS=("${DIR}"*)
    echo "Processing directory: $DIR"
    # FILE_PATHSを使った処理
    for FILE_PATH in "${FILE_PATHS[@]}"; do
        echo "  Processing $FILE_PATH"
        FILE_NAME=$(basename "$FILE_PATH")
        OUTPUT_PREFIX="${OUTPUT_ROOT}/${FILE_NAME%.jsonl.gz}"

        echo $OUTPUT_PREFIX
        apptainer run --nv \
  -w -f -B /gs -B /apps -B /home -B /gs/fs/tga-okazaki/ma:/root ${CONTAINER_IMAGE} \
        python tools/preprocess_data.py \
            --input "$FILE_PATH" \
            --output-prefix "$OUTPUT_PREFIX" \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model tokyotech-llm/Llama-3.1-Swallow-8B-v0.5 \
            --append-eod \
            --use-fast-tokenizer \
            --workers 64
        sleep 1
    done
done
