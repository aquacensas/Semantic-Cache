"""All settings in one place. Values come from the .env file"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    redis_url: str = "redis://localhost:6379"

    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536          # text-embedding-3-small always gives 1536 numbers

    similarity_threshold: float = 0.95  # cosine similarity needed for a cache hit
    default_ttl_seconds: int = 24 * 60 * 60  # entries live for 24 hours

    index_name: str = "llm_cache"
    key_prefix: str = "cache:entry"

    demo_model: str = "gpt-4o-mini"


@lru_cache
def get_settings() -> Settings:
    return Settings()
