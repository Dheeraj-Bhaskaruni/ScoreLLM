"""Tests for evalflow.cache — response caching and rate limiting."""
import os
import tempfile
import time

import pytest

from evalflow.cache import RateLimiter, ResponseCache


@pytest.fixture
def cache():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    c = ResponseCache(db_path=path, ttl_hours=1.0)
    yield c
    os.unlink(path)


class TestResponseCache:
    def test_cache_miss_returns_none(self, cache):
        result = cache.get("model", [{"role": "user", "content": "hi"}], 0.1)
        assert result is None

    def test_cache_hit_after_put(self, cache):
        messages = [{"role": "user", "content": "hello"}]
        cache.put("model-a", messages, 0.1, "response text")
        result = cache.get("model-a", messages, 0.1)
        assert result == "response text"

    def test_different_models_separate_cache(self, cache):
        messages = [{"role": "user", "content": "same prompt"}]
        cache.put("model-a", messages, 0.1, "response A")
        cache.put("model-b", messages, 0.1, "response B")
        assert cache.get("model-a", messages, 0.1) == "response A"
        assert cache.get("model-b", messages, 0.1) == "response B"

    def test_different_temperature_separate_cache(self, cache):
        messages = [{"role": "user", "content": "prompt"}]
        cache.put("model", messages, 0.0, "cold")
        cache.put("model", messages, 1.0, "hot")
        assert cache.get("model", messages, 0.0) == "cold"
        assert cache.get("model", messages, 1.0) == "hot"

    def test_stats_tracking(self, cache):
        messages = [{"role": "user", "content": "test"}]
        cache.get("m", messages, 0.1)  # miss
        cache.put("m", messages, 0.1, "resp")
        cache.get("m", messages, 0.1)  # hit
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_clear_by_model(self, cache):
        m1 = [{"role": "user", "content": "a"}]
        m2 = [{"role": "user", "content": "b"}]
        cache.put("model-a", m1, 0.1, "resp-a")
        cache.put("model-b", m2, 0.1, "resp-b")
        cache.clear(model_id="model-a")
        assert cache.get("model-a", m1, 0.1) is None
        assert cache.get("model-b", m2, 0.1) == "resp-b"

    def test_clear_all(self, cache):
        cache.put("m", [{"role": "user", "content": "x"}], 0.1, "r")
        count = cache.clear()
        assert count == 1


class TestRateLimiter:
    def test_burst_allows_immediate(self):
        limiter = RateLimiter(requests_per_minute=60, burst=5)
        for _ in range(5):
            wait = limiter.acquire()
            assert wait == 0.0

    def test_stats_tracked(self):
        limiter = RateLimiter(requests_per_minute=6000, burst=2)
        limiter.acquire()
        stats = limiter.stats
        assert "total_waits" in stats
        assert "rate_rpm" in stats
        assert stats["rate_rpm"] == 6000
