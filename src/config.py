"""Configuration management using Pydantic Settings."""

import os
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field
except ImportError:
    # Graceful fallback if pydantic_settings is not yet installed
    from pydantic import BaseModel as BaseSettings, Field
    SettingsConfigDict = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

class Settings(BaseSettings):
    """Application settings and environment variables."""
    # API Keys
    GOOGLE_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    GROQ_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    # Provider & Model selection
    LLM_PROVIDER: str = Field(default="gemini")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile")
    EMBEDDING_MODEL: str = Field(default="models/text-embedding-004")

    # Guardrails
    GUARDRAILS_ENABLED: bool = Field(default=True)
    GUARDRAIL_STRICT_MODE: bool = Field(default=True)
    SIMILARITY_THRESHOLD: float = Field(default=0.30)
    TOP_K_RESULTS: int = Field(default=4)

    # Persistence & Paths
    CHROMA_PERSIST_DIR: str = Field(default=str(BASE_DIR / "chroma_db"))
    DATA_PATH: str = Field(default=str(DATA_DIR / "profile_data.json"))
    MD_DATA_PATH: str = Field(default=str(DATA_DIR / "profile_data.md"))

    # Server settings
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)

    if SettingsConfigDict:
        model_config = SettingsConfigDict(
            env_file=str(BASE_DIR / ".env"),
            env_file_encoding="utf-8",
            extra="ignore"
        )
    else:
        class Config:
            env_file = str(BASE_DIR / ".env")
            extra = "ignore"

settings = Settings()

