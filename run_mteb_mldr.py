import mteb
from mteb.models.cache_wrapper import CachedEmbeddingWrapper
from pathlib import Path
import argparse
# CUDA_VISIBLE_DEVICES=7 python run_mteb_mldr.py


# HF_HOME=/home/jinaai/nanw/modernbert_benchmarks/.cache HTTPS_PROXY=http://127.0.0.1:7890 CUDA_VISIBLE_DEVICES=0 python run_mteb_mldr.py --model_name bwang0911/test-roberta-ft --batch_size 128
# HF_HOME=/home/jinaai/nanw/modernbert_benchmarks/.cache HTTPS_PROXY=http://127.0.0.1:7890 CUDA_VISIBLE_DEVICES=1 python run_mteb_mldr.py --model_name bwang0911/test-modernbert-ft --batch_size 128
# HF_HOME=/home/jinaai/nanw/modernbert_benchmarks/.cache HTTPS_PROXY=http://127.0.0.1:7890 CUDA_VISIBLE_DEVICES=2 python run_mteb_mldr.py --model_name bwang0911/test-jina-xlm-roberta-ft --batch_size 128

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

    model = mteb.get_model(model_name, trust_remote_code=True)
    # model_with_cached_emb = CachedEmbeddingWrapper(model, cache_path=f'.cache/cache_embeddings/{model_name}')
    tasks = mteb.get_tasks(tasks=["MultiLongDocRetrieval",], languages=["eng-Latn",], eval_splits=["test",])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        # model_with_cached_emb,
        model,
        output_folder=f"results/{model_name}",
        encode_kwargs={'batch_size': batch_size},
        verbosity=2,
        overwrite_results=True,
        save_predictions=True
    )
    print(results)


if __name__ == '__main__':
    main()