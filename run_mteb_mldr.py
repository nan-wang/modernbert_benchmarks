import mteb
from pathlib import Path
import argparse
# CUDA_VISIBLE_DEVICES=7 python run_mteb_mldr.py


# Define the sentence-transformers model name
# model_name = "/home/nwang/modernbert_benchmarks/output/train_bi-encoder-margin_mse-sentence-transformers-nli-roberta-large-batch_size_64-2025-01-12_11-07-50"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/msmarco-bert-base-dot-v5")
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int
    )
    args = parser.parse_args()
    model_name = args.model_name
    batch_size = args.batch_size

    model = mteb.get_model(model_name)
    tasks = mteb.get_tasks(tasks=["MultiLongDocRetrieval"])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=f"results/{model_name}", encode_kwargs={'batch_size': batch_size})
    print(results)


if __name__ == '__main__':
    main()