from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    WARDEN_ENV: str = "development"
    QDRANT_HOST: str = "localhost"
    QDRANT_HTTP_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_COLLECTION: str = "warden_hr_policies"
    QDRANT_POOL_SIZE: int = 128
    GRPC_PORT: int = 50051
    LAYA_GRPC_URL: str = "warden-laya-service:50051"
    RETRIEVAL_TOP_K_DEFAULT: int = 30
    RERANK_LIMIT_DEFAULT: int = 5
    CANDIDATE_PRUNE_RATIO: float = 0.40
    PRUNED_CANDIDATE_MAX: int = 10
    LAYA_DEADLINE_MS: int = 150
    LOG_LEVEL: str = "INFO"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
