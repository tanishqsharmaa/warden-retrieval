import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from warden_shared.logging import configure_logging

from warden_retrieval.api import create_app
from warden_retrieval.collection import init_qdrant_collection
from warden_retrieval.config import get_settings
from warden_retrieval.grpc_server import RetrievalServiceServicerImpl, create_grpc_server
from warden_retrieval.indexer import BatchIndexer
from warden_retrieval.laya_client import SpeculativeLayaClient
from warden_retrieval.qdrant_client import QdrantClientManager
from warden_retrieval.search import HybridSearchEngine

logger = logging.getLogger("warden.retrieval")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and graceful shutdown for both gRPC and REST runtimes."""
    settings = get_settings()
    configure_logging(service_name="warden-retrieval", level=settings.LOG_LEVEL)
    logger.info("Initializing warden-retrieval subsystem dependencies...")

    client_mgr = getattr(app.state, "client_manager", None) or QdrantClientManager(settings)
    qdrant_client = await client_mgr.get_client()

    # Ensure collection and payload indexes exist
    await init_qdrant_collection(qdrant_client, settings.QDRANT_COLLECTION)

    search_engine = getattr(app.state, "search_engine", None) or HybridSearchEngine(
        qdrant_client, settings.QDRANT_COLLECTION
    )
    indexer = getattr(app.state, "indexer", None) or BatchIndexer(
        qdrant_client, settings.QDRANT_COLLECTION
    )
    laya_client = getattr(app.state, "laya_client", None) or SpeculativeLayaClient(
        target_url=settings.LAYA_GRPC_URL
    )

    # Initialize and start gRPC server
    servicer = RetrievalServiceServicerImpl(
        search_engine=search_engine,
        laya_client=laya_client,
        indexer=indexer,
        settings=settings,
    )
    grpc_server = create_grpc_server(servicer=servicer, port=settings.GRPC_PORT)
    await grpc_server.start()
    logger.info(f"gRPC RetrievalService active on port {settings.GRPC_PORT}")

    # Attach live dependencies to app state
    app.state.client_manager = client_mgr
    app.state.search_engine = search_engine
    app.state.indexer = indexer
    app.state.laya_client = laya_client
    app.state.grpc_server = grpc_server

    yield

    logger.info("Shutting down warden-retrieval services...")
    await grpc_server.stop(grace=5.0)
    await laya_client.close()
    await client_mgr.close()
    logger.info("Shutdown complete.")

def get_application() -> FastAPI:
    """Creates the default FastAPI application with lifespan management."""
    settings = get_settings()
    client_mgr = QdrantClientManager(settings)
    app = create_app(client_manager=client_mgr, settings=settings)
    app.router.lifespan_context = lifespan
    return app

app = get_application()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "warden_retrieval.main:app",
        host="0.0.0.0",
        port=8000,
        log_level=settings.LOG_LEVEL.lower(),
    )
