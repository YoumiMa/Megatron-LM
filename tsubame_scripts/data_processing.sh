#!/bin/bash

# テキストファイルのパス
FILE_PATHS="/gs/bs/tga-okazaki/ma/data/japanese-wikipedia/ja_wiki.jsonl"
BASE_PATH="/gs/bs/tga-okazaki/ma/data/japanese-wikipedia/"
# FILE_PATHS="/gs/bs/tga-okazaki/ma/data/smbcgic/test.jsonl"
# BASE_PATH="/gs/bs/tga-okazaki/ma/data/smbcgic/"
OUTPUT_ROOT="${BASE_PATH}/processed"

# ファイルが存在するか確認
if [ ! -f "$FILE_PATHS" ]; then
    echo "$FILE_PATHS ファイルが見つかりません。"
    exit 1
fi

cd ~/Megatron-LM/ 

for FILE_PATH in $FILE_PATHS; do
    echo "Processing $FILE_PATH"
    python tools/preprocess_data.py \
        --input "$FILE_PATH" \
        --output-prefix "${OUTPUT_ROOT}" \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model meta-llama/Llama-3.1-8B \
        --append-eod \
        --workers 64
    sleep 1
done