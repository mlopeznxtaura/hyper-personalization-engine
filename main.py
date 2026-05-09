"""
hyper-personalization-engine — Entry Point

Real-time recommendation: feature engineering, model training, vector indexing, serving.

Usage:
  python main.py --mode preprocess --data ./data/interactions.csv
  python main.py --mode train --model two-tower --epochs 10
  python main.py --mode index --n-items 100000
  python main.py --mode serve --port 8000
  python main.py --mode pipeline
"""
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Hyper-Personalization Engine")
    parser.add_argument("--mode", required=True,
                        choices=["preprocess", "train", "index", "serve", "pipeline", "demo"])
    parser.add_argument("--data", default="./data/interactions.csv")
    parser.add_argument("--model", default="two-tower", choices=["two-tower"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-items", type=int, default=100_000)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--output", default="./output")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def mode_preprocess(args):
    from features.nvtabular_pipeline import InteractionFeaturePipeline
    pipeline = InteractionFeaturePipeline(output_dir=f"{args.output}/processed")
    output_path = pipeline.fit_transform(args.data)
    stats = pipeline.get_feature_stats(output_path)
    print(f"
Preprocessing complete:")
    print(f"  Rows: {stats['n_rows']:,}")
    print(f"  Columns: {stats['n_cols']}")
    print(f"  Positive rate: {stats['positive_rate']:.2%}")
    print(f"  Output: {output_path}")


def mode_train(args):
    from models.two_tower import TwoTowerTrainer, TwoTowerConfig
    import os
    cfg = TwoTowerConfig(epochs=args.epochs)
    trainer = TwoTowerTrainer(cfg)

    processed_path = f"{args.output}/processed/processed.parquet"
    if not __import__("os").path.exists(processed_path):
        print("Processed data not found. Running preprocessing first...")
        from features.nvtabular_pipeline import InteractionFeaturePipeline
        pipeline = InteractionFeaturePipeline(output_dir=f"{args.output}/processed")
        processed_path = pipeline.fit_transform(args.data)

    result = trainer.train(processed_path)
    model_path = f"{args.output}/models/two_tower.pt"
    __import__("pathlib").Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(model_path)
    print(f"
Training complete: final_loss={result['final_loss']:.4f}")


def mode_index(args):
    from retrieval.qdrant_store import QdrantItemStore, ItemIndexer
    import numpy as np

    store = QdrantItemStore(embedding_dim=64)
    indexer = ItemIndexer(store=store)
    item_ids = [f"item_{i:07d}" for i in range(args.n_items)]
    n = indexer.index_all_items(item_ids)
    print(f"
Indexed {n:,} items into Qdrant")


def mode_serve(args):
    import uvicorn
    from api.server import app
    from prometheus_client import start_http_server
    start_http_server(9090)
    print(f"[Server] Starting on {args.host}:{args.port} | Metrics on :9090")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def mode_demo(args):
    """Run a quick demo of the full recommendation flow."""
    import numpy as np
    print("Running recommendation demo...
")

    # Preprocess
    from features.nvtabular_pipeline import InteractionFeaturePipeline
    pipeline = InteractionFeaturePipeline(output_dir=f"{args.output}/processed")
    data_path = pipeline.fit_transform(args.data)
    stats = pipeline.get_feature_stats(data_path)
    print(f"Processed {stats['n_rows']:,} interactions
")

    # Simulate recommendation
    from retrieval.redis_cache import UserVectorCache
    cache = UserVectorCache()
    user_emb = np.random.randn(64).astype(np.float32)
    user_emb /= np.linalg.norm(user_emb)
    cache.set_embedding("user_001", user_emb)
    cache.push_interaction("user_001", "item_042", "click")
    cache.push_interaction("user_001", "item_017", "purchase")

    state = cache.get_user_state("user_001")
    print(f"User state: {state}")
    print("
Demo complete. Start serve mode for live API.")


def main():
    args = parse_args()
    print("=" * 60)
    print("  Hyper-Personalization Engine")
    print(f"  Mode: {args.mode.upper()}")
    print("=" * 60)

    dispatch = {
        "preprocess": mode_preprocess,
        "train": mode_train,
        "index": mode_index,
        "serve": mode_serve,
        "demo": mode_demo,
        "pipeline": mode_demo,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
