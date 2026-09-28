import logging

from qdrant_client import AsyncQdrantClient, models
from warden_shared.errors import QdrantUnavailableError

logger = logging.getLogger("warden.retrieval.collection")

async def init_qdrant_collection(client: AsyncQdrantClient, collection_name: str) -> bool:
    """Initializes the Qdrant collection with 768-dim INT8 scalar quantization,

    sparse BM25 vectors, and payload keyword indexes on role_tags and doc_id.
    Returns True if collection was created, False if it already existed.
    """
    try:
        exists = await client.collection_exists(collection_name)
        if exists:
            logger.info(f"Qdrant collection '{collection_name}' already exists. Skipping creation.")
            return False

        logger.info(f"Creating Qdrant collection '{collection_name}'...")
        await client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                        full_scan_threshold=10000,
                        max_indexing_threads=2,
                        on_disk=False,
                    ),
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True,
                        )
                    ),
                )
            },
            sparse_vectors_config={
                "sparse_bm25": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
            optimizers_config=models.OptimizersConfigDiff(
                deleted_threshold=0.2,
                vacuum_min_vector_number=1000,
                default_segment_number=2,
                flush_interval_sec=5,
            ),
        )

        logger.info("Creating payload indexes on 'role_tags' and 'doc_id'...")
        await client.create_payload_index(
            collection_name=collection_name,
            field_name="role_tags",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await client.create_payload_index(
            collection_name=collection_name,
            field_name="doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        logger.info(
            f"Collection '{collection_name}' initialized successfully with payload indexes."
        )
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant collection '{collection_name}': {e}")
        err_msg = f"Failed to initialize Qdrant collection '{collection_name}': {e}"
        raise QdrantUnavailableError(detail=err_msg) from e
