import logging
from typing import Optional

from qdrant_client import AsyncQdrantClient
from warden_shared.errors import QdrantUnavailableError

from warden_retrieval.config import Settings, get_settings

logger = logging.getLogger("warden.retrieval.client")

class QdrantClientManager:
    """Manages AsyncQdrantClient connection pool and health checks."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._client: Optional[AsyncQdrantClient] = None

    async def get_client(self) -> AsyncQdrantClient:
        """Returns the pooled AsyncQdrantClient instance."""
        if self._client is None:
            try:
                self._client = AsyncQdrantClient(
                    host=self.settings.QDRANT_HOST,
                    grpc_port=self.settings.QDRANT_GRPC_PORT,
                    port=self.settings.QDRANT_HTTP_PORT,
                    prefer_grpc=True,
                    timeout=5.0,
                )
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant cluster: {e}")
                err_msg = f"Failed to initialize Qdrant client: {e}"
                raise QdrantUnavailableError(detail=err_msg) from e
        return self._client

    async def is_healthy(self) -> bool:
        """Performs a readiness health check against the Qdrant cluster."""
        try:
            client = await self.get_client()
            await client.get_collection(self.settings.QDRANT_COLLECTION)
            return True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False

    async def close(self) -> None:
        """Closes the underlying client connection pool."""
        if self._client is not None:
            await self._client.close()
            self._client = None
