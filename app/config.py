"""Settings read from the environment (and an optional, git-ignored .env file)."""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Common install locations, tried when TESSERACT_CMD is empty and tesseract is not on PATH
# (for example a non-login shell on macOS that never loaded Homebrew's PATH).
_TESSERACT_FALLBACKS = (
    "/opt/homebrew/bin/tesseract",  # Homebrew, Apple Silicon
    "/usr/local/bin/tesseract",  # Homebrew, Intel
    "/usr/bin/tesseract",  # Linux packages
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",  # UB Mannheim, Windows
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # LLM via OpenRouter (OpenAI-compatible API)
    llm_provider: str = "openrouter"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""

    # OCR
    tesseract_cmd: str = ""
    ocr_dpi: int = 300  # PDF render resolution
    ocr_max_pages: int = 3

    # Web
    cors_origin: str = ""

    # Uploads
    max_upload_bytes: int = 10 * 1024 * 1024
    upload_dir: Path = Field(default=PROJECT_ROOT / "uploads")

    # Gates and guards (see docs/ARCHITECTURE.md §3–4)
    ocr_min_chars: int = 40  # letters/digits needed before we trust the text path
    ocr_min_mean_conf: float = 60.0
    field_min_conf: float = 80.0  # per-word confidence for the percentage
    cutoff_percentage: float = 60.0
    near_cutoff_band: float = 2.0

    def resolved_tesseract_cmd(self) -> str:
        """Path to the tesseract binary: TESSERACT_CMD, then PATH, then common locations."""
        if self.tesseract_cmd:
            return self.tesseract_cmd
        found = shutil.which("tesseract")
        if found:
            return found
        for candidate in _TESSERACT_FALLBACKS:
            if Path(candidate).exists():
                return candidate
        return "tesseract"


@lru_cache
def get_settings() -> Settings:
    return Settings()
