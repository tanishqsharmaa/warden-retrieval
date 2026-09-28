from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient


@pytest.fixture
def mock_qdrant_client():
    client = AsyncMock()
    client.collection_exists = AsyncMock(return_value=False)
    client.create_collection = AsyncMock()
    client.create_payload_index = AsyncMock()
    client.get_collection = AsyncMock()
    client.close = AsyncMock()
    return client

@pytest.fixture
async def live_qdrant_client():
    """Provides an isolated AsyncQdrantClient instance for integration tests."""
    client = AsyncQdrantClient(":memory:")
    yield client
    await client.close()

