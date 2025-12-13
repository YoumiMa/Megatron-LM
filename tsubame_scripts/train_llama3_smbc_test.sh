# data config
DATASET_DIR="/gs/bs/tga-okazaki/ma/data/smbcgic_processed"

TRAIN_DATA_PATH=""

# smbc data
# DATASET_DIR配下の全てのサブディレクトリを追加
for FILE in "${DATASET_DIR}"/*; do
    echo $FILE
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
# echo ${TRAIN_DATA_PATH}
