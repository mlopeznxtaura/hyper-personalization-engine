"""
Redis sub-millisecond user vector cache.
Cache user embeddings and personalization state for <5ms recommendation serving.
SDKs: redis
"""
import os
import json
import time
import struct
import numpy as np
from typing import Optional, Dict, Any, List

import redis


class UserVectorCache:
    """
    Redis cache for user embeddings and personalization state.
    Target: <5ms p99 latency for cached user lookups.
    """

    EMBEDDING_TTL = 3600        # 1 hour
    SESSION_TTL = 1800          # 30 minutes
    HISTORY_MAX = 100           # max items in click history

    def __init__(
        self,
        url: Optional[str] = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        embedding_dim: int = 64,
    ):
        redis_url = url or os.environ.get("REDIS_URL")
        self.embedding_dim = embedding_dim
        if redis_url:
            self.r = redis.from_url(redis_url, decode_responses=False)
        else:
            self.r = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        try:
            self.r.ping()
            print("[Redis] User cache connected")
        except Exception as e:
            print(f"[Redis] Connection failed: {e}")
            self.r = None

    def _emb_key(self, user_id: str) -> str:
        return f"user:emb:{user_id}"

    def _session_key(self, user_id: str) -> str:
        return f"user:session:{user_id}"

    def _history_key(self, user_id: str) -> str:
        return f"user:history:{user_id}"

    def set_embedding(self, user_id: str, embedding: np.ndarray):
        """Store user embedding as packed float32 bytes."""
        if self.r is None:
            return
        packed = struct.pack(f"{self.embedding_dim}f", *embedding.astype(np.float32).tolist())
        self.r.setex(self._emb_key(user_id), self.EMBEDDING_TTL, packed)

    def get_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Retrieve user embedding. Returns None if not cached."""
        if self.r is None:
            return None
        data = self.r.get(self._emb_key(user_id))
        if data is None:
            return None
        return np.array(struct.unpack(f"{self.embedding_dim}f", data), dtype=np.float32)

    def update_session(self, user_id: str, context: Dict[str, Any]):
        """Update user session context (device, location, time-of-day, etc.)."""
        if self.r is None:
            return
        context["updated_at"] = time.time()
        self.r.setex(
            self._session_key(user_id),
            self.SESSION_TTL,
            json.dumps(context).encode(),
        )

    def get_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        if self.r is None:
            return None
        data = self.r.get(self._session_key(user_id))
        return json.loads(data) if data else None

    def push_interaction(self, user_id: str, item_id: str, interaction_type: str = "click"):
        """Append an interaction to user history (capped at HISTORY_MAX)."""
        if self.r is None:
            return
        entry = json.dumps({"item_id": item_id, "type": interaction_type, "ts": time.time()})
        pipe = self.r.pipeline()
        pipe.lpush(self._history_key(user_id), entry.encode())
        pipe.ltrim(self._history_key(user_id), 0, self.HISTORY_MAX - 1)
        pipe.expire(self._history_key(user_id), self.SESSION_TTL)
        pipe.execute()

    def get_history(self, user_id: str, n: int = 20) -> List[Dict]:
        if self.r is None:
            return []
        items = self.r.lrange(self._history_key(user_id), 0, n - 1)
        return [json.loads(item) for item in items]

    def get_user_state(self, user_id: str) -> Dict[str, Any]:
        """Get full user state: embedding + session + recent history."""
        emb = self.get_embedding(user_id)
        session = self.get_session(user_id) or {}
        history = self.get_history(user_id, n=10)
        return {
            "user_id": user_id,
            "has_embedding": emb is not None,
            "embedding_dim": len(emb) if emb is not None else 0,
            "session": session,
            "recent_items": [h["item_id"] for h in history],
        }

    def invalidate(self, user_id: str):
        """Clear all cached state for a user (e.g., after explicit preference update)."""
        if self.r is None:
            return
        self.r.delete(self._emb_key(user_id))
        self.r.delete(self._session_key(user_id))
        self.r.delete(self._history_key(user_id))
