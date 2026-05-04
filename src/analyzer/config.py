from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str = "sk-ant-placeholder"
    model_id: str = "claude-sonnet-4-6"
    llm_timeout_ms: int = 2000
    llm_max_retries: int = 3
    confidence_threshold: float = 0.65

    db_path: str = "analyzer.db"

    max_concurrent_llm: int = 5
    max_traceback_tokens: int = 1800

    eval_dataset_path: str = "data/eval_dataset/labeled_failures.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
