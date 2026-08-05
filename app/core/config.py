from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Taleem AI Service"
    APP_ENV: str = "development"
    FIREBASE_PROJECT_ID: str = ""
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    FIREBASE_ADMIN_PROJECT_ID: str = ""
    FIREBASE_ADMIN_CLIENT_EMAIL: str = ""
    FIREBASE_ADMIN_PRIVATE_KEY: str = ""
    INTERNAL_JWT_PUBLIC_KEYS_JSON: str = "{}"
    REDIS_URL: str = "redis://localhost:6379/0"
    ACTIVE_CORPUS_CACHE_TTL_SECONDS: int = 300
    USAGE_POLICY_CACHE_TTL_SECONDS: int = 300
    PROMPT_CACHE_TTL_SECONDS: int = 300
    USAGE_UID_HMAC_SECRET: str = ""
    INTERNAL_JTI_HMAC_SECRET: str = ""
    BUSINESS_TIMEZONE: str = "Asia/Karachi"
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/chat/completions"
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"
    DEEPSEEK_TIMEOUT_SECONDS: float = 20.0
    DEEPSEEK_MAX_RETRIES: int = 2
    DEEPSEEK_MAX_OUTPUT_TOKENS: int = 2400
    DEEPSEEK_MAX_INPUT_CHARACTERS: int = 32000
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/taleem_dev"
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_MODEL_REVISION: str = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    EMBEDDING_DIM: int = 768
    WORKER_MODE: str = ""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
