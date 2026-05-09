# Hyper-Personalization Engine

Cluster 18 of the NextAura 500 SDKs / 25 Clusters project.

Real-time recommendation and content adaptation for every individual user. GPU-accelerated end-to-end: user event → Kafka → Flink features → Redis vector update → real-time re-rank.

## Architecture

- NVIDIA Merlin + NVTabular for GPU-accelerated feature engineering
- RAPIDS cuDF + cuML for GPU dataframe operations and embedding training
- HugeCTR for distributed CTR model training
- HuggingFace Transformers for semantic item embeddings (two-tower model)
- Qdrant + LanceDB for vector similarity search
- Kafka + Flink for real-time event streaming and feature aggregation
- Redis for sub-millisecond user vector cache
- Prefect for training pipeline orchestration
- W&B + MLflow for experiment tracking
- Prometheus + Grafana for recommendation system observability

## SDKs Used

NVIDIA Merlin SDK, RAPIDS cuML, RAPIDS cuDF, HugeCTR, HuggingFace Transformers, Pinecone SDK, Qdrant SDK, LanceDB SDK, Kafka SDK, Apache Flink SDK, Redis SDK, FastAPI, Polars, DuckDB, Weights & Biases, MLflow, Prometheus Client, Grafana SDK, Prefect SDK, OpenTelemetry SDK

## Quickstart

```bash
pip install -r requirements.txt
docker-compose up -d  # Kafka, Redis, Qdrant, Prometheus

# Preprocess interaction data
python main.py --mode preprocess --data ./data/interactions.csv

# Train two-tower retrieval model
python main.py --mode train --model two-tower --epochs 10

# Start real-time recommendation API
python main.py --mode serve --port 8000

# Run full pipeline
python main.py --mode pipeline
```

## Structure

```
features/
  nvtabular_pipeline.py  NVTabular GPU feature engineering
  flink_aggregator.py    Apache Flink real-time feature aggregation
models/
  two_tower.py           HuggingFace two-tower retrieval model
  ranking.py             cuML + HugeCTR ranking model
retrieval/
  qdrant_store.py        Qdrant vector similarity search
  lance_store.py         LanceDB embedding storage
  redis_cache.py         Redis user vector cache
streaming/
  kafka_events.py        Kafka user event ingestion
pipeline/
  prefect_pipeline.py    Prefect training orchestration
api/
  server.py              FastAPI recommendation endpoints
main.py                  Entry point
```
