from functools import lru_cache

from pydantic import Field
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
    TALEEM_PROCESS_ROLE: str = "api"
    WORKER_MODE: str = ""
    EMBEDDING_MODEL: str = "voyage-4-lite"
    EMBEDDING_MODEL_REVISION: str = "voyage-4-lite-512-v1"
    EMBEDDING_DIM: int = 512
    EMBEDDING_OUTPUT_DTYPE: str = "float"
    VOYAGE_ADMIN_API_KEY: str = ""
    VOYAGE_API_KEY: str = ""
    VOYAGE_EMBED_BATCH_SIZE: int = 64
    GEMINI_API_KEY: str = ""
    GEMINI_OCR_MODEL: str = "gemini-3.6-flash"
    MULTIPLE_ASK_ANSWER_BATCH_SIZE: int = 5
    # Module 5 Run 1 is intentionally dark until its complete worker flow is
    # introduced and separately verified. This gate applies to all internals.
    MULTIPLE_ASK_RUN1_ENABLED: bool = False
    MULTIPLE_ASK_TEMPORARY_BUCKET: str = "multiple-ask-temporary"
    MULTIPLE_ASK_SESSION_TTL_SECONDS: int = 900
    # Raw source follows finalization, not the short upload capability.
    MULTIPLE_ASK_RAW_SOURCE_RETENTION_HOURS: int = 24
    MULTIPLE_ASK_JOB_RETENTION_DAYS: int = 7
    MULTIPLE_ASK_MAX_IMAGE_BYTES: int = 8 * 1024 * 1024
    MULTIPLE_ASK_MAX_PDF_BYTES: int = 15 * 1024 * 1024
    MULTIPLE_ASK_MAX_TEXT_CHARACTERS: int = 30_000
    MULTIPLE_ASK_MAX_PDF_PAGES: int = 10
    # A compressed image can expand far beyond its 8 MB upload size.  Keep the
    # validation worker within a predictable memory budget before OCR exists.
    MULTIPLE_ASK_MAX_IMAGE_PIXELS: int = 20_000_000
    # PDFs with embedded text are checked before quota is committed.  A scanned
    # PDF legitimately has no embedded text and is still eligible for Run 2 OCR.
    MULTIPLE_ASK_MAX_PDF_EXTRACTED_CHARACTERS: int = 30_000
    # This is a hard safety ceiling, not merely a UI preference. Deployments
    # may choose a lower value but cannot increase a student paper past 60.
    MULTIPLE_ASK_MAX_EXTRACTED_QUESTIONS: int = Field(default=60, ge=1, le=60)
    # Gemini structured OCR can take longer than a conventional text request,
    # particularly for a high-resolution exam-paper image.  The runtime
    # setting remains authoritative, but keep the environment default aligned
    # with the safe production default used for a missing runtime row.
    MULTIPLE_ASK_OCR_TIMEOUT_SECONDS: int = 90
    # PDF page dimensions are attacker-controlled. Rendering one page at a
    # time is not sufficient unless its rendered pixel count is bounded too.
    MULTIPLE_ASK_MAX_RENDERED_PDF_PAGE_PIXELS: int = 12_000_000
    MULTIPLE_ASK_CLEANUP_BATCH_SIZE: int = 100
    MULTIPLE_ASK_CLEANUP_INTERVAL_SECONDS: int = 300

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
