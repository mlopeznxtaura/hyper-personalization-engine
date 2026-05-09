"""
Qdrant vector similarity search for item retrieval.
Store item embeddings, search by user vector, filter by metadata.
SDKs: qdrant-client
"""
import os
import time
import uuid
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    SearchRequest, ScoredPoint,
)


class QdrantItemStore:
    """
    Qdrant vector store for item embeddings.
    Fast ANN search to retrieve top-K candidate items for a user.
    """

    def __init__(
        self,
        collection_name: str = "items",
        embedding_dim: int = 64,
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        qdrant_url = url or os.environ.get("QDRANT_URL")
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url)
        else:
            self.client = QdrantClient(host=host, port=port)

        self._ensure_collection()
        print(f"[Qdrant] Store ready: {collection_name} ({embedding_dim}D)")

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            )
            print(f"[Qdrant] Created collection: {self.collection_name}")

    def upsert_items(
        self,
        item_ids: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 512,
    ) -> int:
        """Upsert item embeddings into Qdrant. Returns number of points upserted."""
        metadata = metadata or [{} for _ in item_ids]
        total = 0
        for i in range(0, len(item_ids), batch_size):
            batch_ids = item_ids[i:i+batch_size]
            batch_embs = embeddings[i:i+batch_size]
            batch_meta = metadata[i:i+batch_size]
            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, iid)),
                    vector=emb.tolist(),
                    payload={"item_id": iid, **meta},
                )
                for iid, emb, meta in zip(batch_ids, batch_embs, batch_meta)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            total += len(points)

        print(f"[Qdrant] Upserted {total} items")
        return total

    def search(
        self,
        user_embedding: np.ndarray,
        top_k: int = 100,
        category_filter: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        ANN search: find top_k items closest to user embedding.
        Supports metadata filters (category, price range).
        """
        query_filter = None
        conditions = []
        if category_filter:
            conditions.append(FieldCondition(key="category", match=MatchValue(value=category_filter)))
        if min_price is not None or max_price is not None:
            conditions.append(FieldCondition(
                key="price",
                range=Range(gte=min_price, lte=max_price),
            ))
        if conditions:
            query_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=user_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        return [
            {
                "item_id": r.payload.get("item_id", str(r.id)),
                "score": r.score,
                **{k: v for k, v in r.payload.items() if k != "item_id"},
            }
            for r in results
        ]

    def get_collection_size(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)
        print(f"[Qdrant] Deleted collection: {self.collection_name}")


class ItemIndexer:
    """
    Build and maintain the item embedding index from a trained model.
    """

    def __init__(self, store: QdrantItemStore, model=None):
        self.store = store
        self.model = model

    def index_all_items(
        self,
        item_ids: List[str],
        item_features: Optional[np.ndarray] = None,
        batch_size: int = 1024,
    ) -> int:
        """Generate and index embeddings for all items."""
        total = 0
        for i in range(0, len(item_ids), batch_size):
            batch_ids = item_ids[i:i+batch_size]
            batch_feats = item_features[i:i+batch_size] if item_features is not None else None

            if self.model is not None:
                embs = self.model.get_item_embeddings_batch(
                    list(range(i, min(i+batch_size, len(item_ids)))),
                    batch_feats,
                )
            else:
                # Random embeddings for testing
                embs = np.random.randn(len(batch_ids), self.store.embedding_dim).astype(np.float32)
                embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

            self.store.upsert_items(batch_ids, embs)
            total += len(batch_ids)
            if (i // batch_size) % 10 == 0:
                print(f"[Indexer] {total}/{len(item_ids)} items indexed")

        print(f"[Indexer] Complete: {total} items in {self.store.collection_name}")
        return total
