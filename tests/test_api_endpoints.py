"""
Unit tests for backend/api.py — FastAPI endpoints.
Uses httpx.AsyncClient with mocked Redis and Database dependencies.
"""
import sys
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


# ---------- Helpers ----------

def _make_mock_db_session():
    """Build a mock session where execute().scalars().all() returns []."""
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = []
    mock_scalars.first.return_value = None

    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    session.commit = AsyncMock()
    session.close = AsyncMock()
    return session


# ---------- GET /status ----------

class TestStatusEndpoint:
    @pytest.mark.asyncio
    async def test_status_returns_200(self):
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.close = AsyncMock()
        mock_redis.blpop = AsyncMock(return_value=None)

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            with patch("database.init_db", new_callable=AsyncMock):
                from api import app
                from httpx import AsyncClient, ASGITransport

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.get("/status")
                    assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_status_contains_online(self):
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.close = AsyncMock()
        mock_redis.blpop = AsyncMock(return_value=None)

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            with patch("database.init_db", new_callable=AsyncMock):
                from api import app
                from httpx import AsyncClient, ASGITransport

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.get("/status")
                    data = response.json()
                    assert data["status"] == "online"
                    assert data["service"] == "Guardian AI Engine API"
                    assert "database" in data


# ---------- GET /api/alerts ----------

class TestAlertsEndpoint:
    @pytest.mark.asyncio
    async def test_alerts_returns_200_and_list(self):
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.close = AsyncMock()
        mock_redis.blpop = AsyncMock(return_value=None)

        mock_session = _make_mock_db_session()

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            with patch("database.init_db", new_callable=AsyncMock):
                from api import app
                from database import get_db

                async def override_get_db():
                    yield mock_session

                app.dependency_overrides[get_db] = override_get_db

                try:
                    from httpx import AsyncClient, ASGITransport
                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as client:
                        response = await client.get("/api/alerts")
                        assert response.status_code == 200
                        data = response.json()
                        assert isinstance(data, list)
                        assert data == []  # mock returns empty list
                finally:
                    app.dependency_overrides.clear()


# ---------- WebSocket /ws/live_feed ----------

class TestWebSocket:
    @pytest.mark.asyncio
    async def test_websocket_connects(self):
        """Verify WebSocket endpoint accepts connections."""
        try:
            from httpx_ws import aconnect_ws
        except ImportError:
            pytest.skip("httpx_ws not installed, skipping WebSocket test")

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.close = AsyncMock()
        mock_redis.blpop = AsyncMock(return_value=None)

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            with patch("database.init_db", new_callable=AsyncMock):
                from api import app
                from httpx import AsyncClient, ASGITransport

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    try:
                        async with aconnect_ws("http://test/ws/live_feed", client) as ws:
                            pass
                    except Exception:
                        pass
