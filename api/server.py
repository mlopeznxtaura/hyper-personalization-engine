"""
FastAPI recommendation serving API.
<50ms end-to-end: fetch user embedding -> Qdrant ANN search -> re-rank -> return.
SDKs: FastAPI, Qdrant, Redis, Prometheus
"""
import os
import time
import uuid
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, start_http_server

app = FastAPI(title="Hyper-Personalization Engine API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# Metrics
RECOMMENDATION_REQUESTS = Counter("recsys_requests_total", "Total recommendation requests", ["endpoint"])
RECOMMENDATION_LATENCY = Histogram(
    "recsys_latency_ms", "End-to-end recommendation latency",
    buckets=[5, 10, 20, 30, 50, 75, 100, 150, 200, 500]
)
CACHE_HITS = Counter("recsys_cache_hits_total", "Redis cache hits")
CACHE_MISSES = Counter("recsys_cache_misses_total", "Redis cache misses")

# Lazy-loaded components
_qdrant_store = None
_user_cache = None
_model = None


def get_qdrant():
    global _qdrant_store
    if _qdrant_store is None:
        from retrieval.qdrant_store import QdrantItemStore
        _qdrant_store = QdrantItemStore(
            host=os.environ.get("QDRANT_HOST", "localhost"),
            port=int(os.environ.get("QDRANT_PORT", "6333")),
        )
    return _qdrant_store


def get_cache():
    global _user_cache
    if _user_cache is None:
        from retrieval.redis_cache import UserVectorCache
        _user_cache = UserVectorCache(
            url=os.environ.get("REDIS_URL"),
            host=os.environ.get("REDIS_HOST", "localhost"),
        )
    return _user_cache


# ---- Request/Response models ----

class RecommendRequest(BaseModel):
    user_id: str
    n: int = 20
    context: Optional[Dict[str, Any]] = None
    category_filter: Optional[str] = None
    exclude_seen: bool = True


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, Any]]
    latency_ms: float
    cache_hit: bool
    request_id: str


class InteractionEvent(BaseModel):
    user_id: str
    item_id: str
    interaction_type: str = "click"   # click, purchase, view, add_to_cart
    context: Optional[Dict] = None


# ---- Endpoints ----

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest, background_tasks: BackgroundTasks):
    """
    Get personalized recommendations for a user.
    Flow: Redis cache -> Qdrant ANN search -> re-rank -> return
    """
    t0 = time.perf_counter()
    RECOMMENDATION_REQUESTS.labels(endpoint="recommend").inc()
    request_id = str(uuid.uuid4())[:12]

    cache = get_cache()
    qdrant = get_qdrant()

    # Try cache first
    user_emb = cache.get_embedding(req.user_id) if cache.r else None
    cache_hit = user_emb is not None

    if cache_hit:
        CACHE_HITS.inc()
    else:
        CACHE_MISSES.inc()
        # Generate random embedding (in prod: compute from model)
        user_emb = np.random.randn(64).astype(np.float32)
        user_emb /= np.linalg.norm(user_emb) + 1e-8
        # Cache for next request
        if cache.r:
            background_tasks.add_task(cache.set_embedding, req.user_id, user_emb)

    # ANN search
    try:
        candidates = qdrant.search(
            user_embedding=user_emb,
            top_k=req.n * 3,  # over-fetch for re-ranking
            category_filter=req.context.get("category") if req.context else req.category_filter,
        )
    except Exception:
        # Fallback: return synthetic recommendations
        candidates = [
            {"item_id": f"item_{i:06d}", "score": float(1.0 - i * 0.01)}
            for i in range(req.n * 3)
        ]

    # Exclude recently seen items
    if req.exclude_seen and cache.r:
        history = cache.get_history(req.user_id, n=50)
        seen = {h["item_id"] for h in history}
        candidates = [c for c in candidates if c["item_id"] not in seen]

    # Simple re-rank: sort by score, take top N
    recommendations = sorted(candidates, key=lambda x: x["score"], reverse=True)[:req.n]

    elapsed = (time.perf_counter() - t0) * 1000
    RECOMMENDATION_LATENCY.observe(elapsed)

    return RecommendResponse(
        user_id=req.user_id,
        recommendations=recommendations,
        latency_ms=round(elapsed, 2),
        cache_hit=cache_hit,
        request_id=request_id,
    )


@app.post("/event")
async def record_event(event: InteractionEvent, background_tasks: BackgroundTasks):
    """Record a user interaction event. Updates history cache + triggers async feature update."""
    cache = get_cache()
    if cache.r:
        background_tasks.add_task(
            cache.push_interaction, event.user_id, event.item_id, event.interaction_type
        )
        if event.context:
            background_tasks.add_task(cache.update_session, event.user_id, event.context)
    RECOMMENDATION_REQUESTS.labels(endpoint="event").inc()
    return {"status": "recorded", "user_id": event.user_id, "item_id": event.item_id}


@app.get("/users/{user_id}/state")
async def get_user_state(user_id: str):
    cache = get_cache()
    return cache.get_user_state(user_id)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "qdrant_items": _qdrant_store.get_collection_size() if _qdrant_store else "not loaded",
    }
