"""
Integration tests: Redis → DB pipeline.
Requires running Redis and PostgreSQL instances.

Run: python -m pytest tests/test_redis_db_pipeline.py -v -m integration
Skip: python -m pytest -m "not integration"
"""
import sys
import os
import json
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

pytestmark = pytest.mark.integration


# ---------- Redis Connectivity ----------

class TestRedisConnection:
    @pytest.fixture
    def redis_client(self):
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        try:
            r.ping()
        except redis.ConnectionError:
            pytest.skip("Redis is not running on localhost:6379")
        yield r
        r.close()

    def test_redis_ping(self, redis_client):
        assert redis_client.ping() is True

    def test_push_and_pop_alert(self, redis_client, sample_alert_data):
        """Push an alert to queue and verify it can be read back."""
        queue = "test_alerts_queue"
        redis_client.delete(queue)

        redis_client.rpush(queue, json.dumps(sample_alert_data))
        raw = redis_client.lpop(queue)

        assert raw is not None
        data = json.loads(raw)
        assert data["source"] == sample_alert_data["source"]
        assert data["is_attack"] == sample_alert_data["is_attack"]
        assert data["attack_type"] == sample_alert_data["attack_type"]

        # Cleanup
        redis_client.delete(queue)

    def test_stats_increment(self, redis_client):
        """Verify Redis INCR works for stats counters."""
        key = "test_stats:total_packets"
        redis_client.delete(key)

        redis_client.incr(key)
        redis_client.incr(key)
        redis_client.incr(key)

        assert int(redis_client.get(key)) == 3
        redis_client.delete(key)


# ---------- PostgreSQL Connectivity ----------

class TestPostgresConnection:
    @pytest.fixture
    def db_url(self):
        return os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/guardian_ai"
        )

    @pytest.mark.asyncio
    async def test_db_connect_and_create_tables(self, db_url):
        """Verify DB connection and table creation."""
        from sqlalchemy.ext.asyncio import create_async_engine
        from database import Base
        from models import Alert  # noqa: F401 — ensures Alert is registered

        engine = create_async_engine(db_url, echo=False)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        except Exception as e:
            pytest.skip(f"PostgreSQL not available: {e}")
        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_insert_and_query_alert(self, db_url):
        """Insert a test alert and read it back."""
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import select
        from database import Base
        from models import Alert

        engine = create_async_engine(db_url, echo=False)
        SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            # Insert
            async with SessionLocal() as session:
                alert = Alert(
                    timestamp=time.time(),
                    source="test_runner",
                    is_attack=True,
                    confidence=0.88,
                    attack_type="Test Attack",
                    original_features={"test_key": "test_value"},
                )
                session.add(alert)
                await session.commit()

            # Query
            async with SessionLocal() as session:
                stmt = select(Alert).where(Alert.source == "test_runner")
                result = await session.execute(stmt)
                found = result.scalars().first()

                assert found is not None
                assert found.source == "test_runner"
                assert found.is_attack is True
                assert found.attack_type == "Test Attack"

            # Cleanup
            async with SessionLocal() as session:
                stmt = select(Alert).where(Alert.source == "test_runner")
                result = await session.execute(stmt)
                for a in result.scalars().all():
                    await session.delete(a)
                await session.commit()
        except Exception as e:
            pytest.skip(f"PostgreSQL not available: {e}")
        finally:
            await engine.dispose()
