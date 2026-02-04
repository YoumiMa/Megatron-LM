## コンテナ環境の用意

環境構築に[Apptainer](https://apptainer.org/docs/user/main/build_env.html)（Singularityの進化形）を使う。tsubameは個々のホームディレクトリが25Gしかないので、コンテナイメージ（30Gくらい）をグループ領域に配置する。

```
export APPTAINER_CACHEDIR=/path/to/cache

# writable container (sandbox)
apptainer build -s /path/to/container docker:nvcr.io/nvidia/pytorch:25.08-py3
```

コンテナを起動してみる。コンテナ環境でもアクセスしたいファイル・フォルダーは作成した上で`-B`でバインドしておく。TSUBAMEは`/home`よりも`/gs/fs/tga-okazaki/ma` の方が色々置いてあるので、コンテナ環境の`/root`を`/gs/fs/tga-okazaki/ma` にバインドした。

```jsx
# グループ領域をアクセスするために/gsをバインド
# module loadを可能とするために/appsをバインド
# /home領域をアクセスするために/homeをバインド
# Megatron-LMの実装が/gs/fs/tga-okazaki/maにあるためバインド

cd /path/to/container/megatron-container
mkdir gs 
mkdir apps

apptainer shell -B /gs -B /apps -B /home -B /gs/fs/tga-okazaki/ma:/root --nv -f -w /path/to/container
```

学習のログ記録にはwandbを使うのでインストールしておく。他にも使うライブラリがあるのでインストールしておく。

```jsx
Apptainer > pip install wandb 
Apptainer > pip install transformers
Apptainer > pip install accelerate
```

## データの前処理

datasetは`.jsonl.gz`形式のファイルの集まりであり、`"text"`というキーを含むことを前提とする。

そして以下のスクリプトを用いてデータを処理する（tsubame_scripts/data_processing.sh）

```jsx
#!/bin/bash

# テキストファイルのパス
OUTPUT_ROOT="/gs/bs/tga-okazaki/ma/data/smbcgic_processed/"
mkdir -p $OUTPUT_ROOT

BASE_PATH="/gs/bs/tga-okazaki/ma/data/smbcgic/"

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
```

## チェックポイントの前処理

チェックポイントをhf形式→megatron形式に変換する(tsubame_scripts/ckpt_convert_hf_to_megatron.sh)

```jsx
bash tsubame_scripts/ckpt_convert_hf_to_megatron.sh
```

## 学習

### ローカル計算機環境

```jsx
bash tsubame_scripts/train_swallow_smbc_exp1_local.sh
```


### クラウド計算機環境

ジョブを投げるコマンド

```jsx
qsub -g tga-okazaki tsubame_scripts/train_swallow_smbc_exp1.sh
```

wandbの`llm-cpt`というプロジェクトにログが残る

## チェックポイント後処理

torch_dist→torch→huggingface形式で変換

### ローカル計算機環境

```jsx
## torch_dist to torch to huggingface
bash tsubame_scripts/ckpt_convert_swallow_smbc_exp1_local.sh
```

### クラウド計算機環境


```jsx
## torch_dist to torch to huggingface
qsub -g tga-okazaki tsubame_scripts/ckpt_convert_swallow_smbc_exp1.sh
```