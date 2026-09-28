from warden_retrieval.config import Settings, get_settings


def test_default_settings():
    s = Settings()
    assert s.QDRANT_HOST == "localhost"
    assert s.QDRANT_HTTP_PORT == 6333
    assert s.QDRANT_GRPC_PORT == 6334
    assert s.QDRANT_COLLECTION == "warden_hr_policies"
    assert s.QDRANT_POOL_SIZE == 128
    assert s.GRPC_PORT == 50051
    assert s.LAYA_GRPC_URL == "warden-laya-service:50051"
    assert s.RETRIEVAL_TOP_K_DEFAULT == 30
    assert s.RERANK_LIMIT_DEFAULT == 5
    assert s.CANDIDATE_PRUNE_RATIO == 0.40
    assert s.PRUNED_CANDIDATE_MAX == 10
    assert s.LAYA_DEADLINE_MS == 150

def test_get_settings_cached():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
