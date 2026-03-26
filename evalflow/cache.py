"""
evalflow.cache — Request caching and rate limiting for LLM API calls.

Provides a disk-backed response cache (SQLite) and a token-bucket rate limiter
to avoid redundant API calls and stay within provider rate limits.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = ".evalflow_cache.db"


class ResponseCache:
    """
    Disk-backed LLM response cache using SQLite.

    Caches are keyed by (model_id, messages_hash, temperature). This means
    identical prompts to the same model at the same temperature return cached
    results — useful for re-running evaluations without burning API credits.
    """

    def __init__(self, db_path: str = DEFAULT_CACHE_PATH, ttl_hours: float = 168.0):
        # ttl_hours default: 1 week
        self.db_path = Path(db_path)
        self.ttl_seconds = ttl_hours * 3600
        self._init_db()
        self._hits = 0
        self._misses = 0

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_model ON cache(model_id)")
        conn.commit()
        conn.close()  # explicit close for WAL mode

    def _make_key(self, model_id: str, messages: list, temperature: float) -> str:
        content = json.dumps({"model": model_id, "messages": messages, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, model_id: str, messages: list, temperature: float) -> Optional[str]:
        key = self._make_key(model_id, messages, temperature)
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute(
            "SELECT response_json, created_at FROM cache WHERE cache_key=?", (key,)
        ).fetchone()
        conn.close()  # explicit close for WAL mode

        if row is None:
            self._misses += 1
            return None

        response_json, created_at = row
        if time.time() - created_at > self.ttl_seconds:
            self._misses += 1
            return None

        self._hits += 1
        return json.loads(response_json)

    def put(self, model_id: str, messages: list, temperature: float, response: str) -> None:
        key = self._make_key(model_id, messages, temperature)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT OR REPLACE INTO cache (cache_key, model_id, response_json, created_at) VALUES (?, ?, ?, ?)",
            (key, model_id, json.dumps(response), time.time()),
        )
        conn.commit()
        conn.close()  # explicit close for WAL mode

    def clear(self, model_id: Optional[str] = None) -> int:
        conn = sqlite3.connect(str(self.db_path))
        if model_id:
            cursor = conn.execute("DELETE FROM cache WHERE model_id=?", (model_id,))
        else:
            cursor = conn.execute("DELETE FROM cache")
        count = cursor.rowcount
        conn.commit()
        conn.close()  # explicit close for WAL mode
        return count

    def evict_expired(self) -> int:
        cutoff = time.time() - self.ttl_seconds
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
        count = cursor.rowcount
        conn.commit()
        conn.close()  # explicit close for WAL mode
        return count

    @property
    def stats(self) -> Dict[str, int]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }


class RateLimiter:
    """
    Token-bucket rate limiter for API calls.

    Ensures we don't exceed provider rate limits (e.g., OpenAI 500 RPM,
    HuggingFace 100 RPM on free tier). Thread-safe.
    """

    def __init__(self, requests_per_minute: float = 60.0, burst: int = 10):
        self._rate = requests_per_minute / 60.0  # tokens per second
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._total_waits = 0
        self._total_wait_time = 0.0

    def acquire(self) -> float:
        """
        Block until a token is available. Returns the wait time in seconds.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0

            # Need to wait
            wait_time = (1.0 - self._tokens) / self._rate
            self._tokens = 0.0

        time.sleep(wait_time)

        with self._lock:
            self._total_waits += 1
            self._total_wait_time += wait_time

        return wait_time

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_waits": self._total_waits,
            "total_wait_time_s": round(self._total_wait_time, 2),
            "rate_rpm": self._rate * 60,
        }

