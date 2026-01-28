#!/bin/bash

# テキストファイルのパス
OUTPUT_ROOT="/gs/bs/tga-okazaki/ma/data/smbcgic_translated_processed/"
mkdir -p $OUTPUT_ROOT

BASE_PATH="/gs/bs/tga-okazaki/ma/data/smbcgic_translated/"

for DIR in "$BASE_PATH"*/; do
    FILE_PATHS=("${DIR}"*)
    echo "Processing directory: $DIR"
    # FILE_PATHSを使った処理
    for FILE_PATH in "${FILE_PATHS[@]}"; do
        echo "  Processing $FILE_PATH"
        FILE_NAME=$(basename "$FILE_PATH")
        OUTPUT_PREFIX="${OUTPUT_ROOT}/${FILE_NAME%.jsonl.gz}"

        echo $OUTPUT_PREFIX
        python tools/preprocess_data.py \
            --input "$FILE_PATH" \
            --output-prefix "$OUTPUT_PREFIX" \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model meta-llama/Llama-3.1-8B \
            --append-eod \
            --workers 64
        sleep 1
    done
done
