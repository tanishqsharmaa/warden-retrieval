from unittest.mock import AsyncMock

import pytest
from warden_shared.errors import QdrantUnavailableError

from warden_retrieval.collection import init_qdrant_collection
from warden_retrieval.config import Settings
from warden_retrieval.qdrant_client import QdrantClientManager


@pytest.mark.asyncio
async def test_init_collection_creates_schema_and_indexes():
    mock_client = AsyncMock()
    mock_client.collection_exists.return_value = False

    result = await init_qdrant_collection(mock_client, "warden_hr_policies")
    assert result is True
    mock_client.create_collection.assert_awaited_once()
    assert mock_client.create_payload_index.await_count == 2

@pytest.mark.asyncio
async def test_init_collection_skips_if_already_exists():
    mock_client = AsyncMock()
    mock_client.collection_exists.return_value = True

    result = await init_qdrant_collection(mock_client, "warden_hr_policies")
    assert result is False
    mock_client.create_collection.assert_not_awaited()

@pytest.mark.asyncio
async def test_qdrant_client_health_check_failure():
    # Review Focus 2: Qdrant unreachable raises or returns unhealthy
    mock_client = AsyncMock()
    mock_client.get_collection.side_effect = Exception("Connection refused")

    manager = QdrantClientManager(settings=Settings())
    manager._client = mock_client

    is_healthy = await manager.is_healthy()
    assert is_healthy is False

@pytest.mark.asyncio
async def test_init_collection_propagates_qdrant_error():
    mock_client = AsyncMock()
    mock_client.collection_exists.side_effect = Exception("Qdrant TCP fault")

    with pytest.raises(QdrantUnavailableError):
        await init_qdrant_collection(mock_client, "warden_hr_policies")
