import os
from megatron.core.datasets import indexed_dataset
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-prefix', type=str, required=True,
                        help='Path prefix to the .bin/.idx files')
    args = parser.parse_args()

    # bin/idx ファイルの prefix （拡張子を除いた部分）
    prefix = args.data_prefix

    # dataset をロード
    dataset = indexed_dataset.IndexedDataset(prefix)

    # 何件あるか（＝文書数）
    print("Number of documents:", len(dataset))

    # 最初のサンプルのトークン列を確認
    print("First sample token ids:", dataset[0])

    # 全体のトークン数を集計
    total_tokens = sum(len(dataset[i]) for i in range(len(dataset)))
    print("Total tokens:", total_tokens)