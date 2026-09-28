import pytest
from httpx import ASGITransport, AsyncClient
from qdrant_client import AsyncQdrantClient

from warden_retrieval.collection import init_qdrant_collection
from warden_retrieval.indexer import BatchIndexer
from warden_retrieval.laya_client import SpeculativeLayaClient
from warden_retrieval.main import app
from warden_retrieval.qdrant_client import QdrantClientManager
from warden_retrieval.search import HybridSearchEngine


@pytest.mark.integration
@pytest.mark.asyncio
async def test_main_app_lifespan_and_endpoints():
    """Verifies that the production application entrypoint works end-to-end with active lifespan."""
    test_client = AsyncQdrantClient(":memory:")
    await init_qdrant_collection(test_client, "warden_hr_policies")

    client_mgr = QdrantClientManager()
    client_mgr._client = test_client

    app.state.client_manager = client_mgr
    app.state.search_engine = HybridSearchEngine(test_client, "warden_hr_policies")
    app.state.indexer = BatchIndexer(test_client, "warden_hr_policies")
    app.state.laya_client = SpeculativeLayaClient()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Health check should be healthy and return vectors_count
        health_resp = await ac.get("/health")
        assert health_resp.status_code == 200
        data = health_resp.json()
        assert data["status"] == "HEALTHY"
        assert data["qdrant_connected"] is True
        assert "vectors_count" in data

        # 2. Retrieve endpoint should work with dependencies dynamically resolved from app.state
        retrieve_resp = await ac.post(
            "/retrieve",
            json={"query_text": "PTO policy", "enable_reranker": False},
            headers={"X-User-Role": "Employee"},
        )
        assert retrieve_resp.status_code == 200
        ret_data = retrieve_resp.json()
        assert ret_data["caller_role_applied"] == "Employee"
        assert "passages" in ret_data

    await test_client.close()
