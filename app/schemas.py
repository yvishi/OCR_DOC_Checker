"""Shared data contracts between pipeline stages.

Increment 1 covers upload -> OCR -> PII masking. The LLM output and API response models
(LearnSphereFields, LLMOutput, ExtractResponse) are added in later phases.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DocumentType(str, Enum):
    MARK_SHEET = "mark_sheet"
    STUDENT_ID_CARD = "student_id_card"
    FEE_RECEIPT = "fee_receipt"
    OTHER = "other"


# ---------- upload ----------


class StoredUpload(BaseModel):
    """An accepted upload, stored under a random UUID. The original filename is never kept."""

    model_config = ConfigDict(frozen=True)

    upload_id: str  # uuid4 hex
    path: Path
    extension: str  # "pdf" | "png" | "jpg", derived from magic bytes, not the filename
    size_bytes: int


# ---------- OCR ----------


class OcrWord(BaseModel):
    """One Tesseract word (image_to_data level 5). Coordinates are pixels in the processed page image."""

    model_config = ConfigDict(frozen=True)

    text: str
    conf: float  # 0-100
    left: int
    top: int
    width: int
    height: int
    page: int = 0  # 0-based page index
    line_key: tuple[int, int, int] = (0, 0, 0)  # (block_num, par_num, line_num) within the page


class OcrPage(BaseModel):
    page: int
    width: int
    height: int
    rotation_applied: int = 0  # degrees clockwise the page was rotated to make it upright
    stamp_pixels_removed: int = 0


class OcrResult(BaseModel):
    words: list[OcrWord]
    text: str  # words joined: spaces within a line, newlines between lines, blank line between pages
    pages: list[OcrPage]
    mean_conf: float  # mean of word confidences, 0 if no words
    char_count: int  # letters and digits only

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def has_boxes(self) -> bool:
        return bool(self.words)


# ---------- PII ----------


class MaskResult(BaseModel):
    """Output of the masker. token_map stays in process memory only and is cleared in `finally`."""

    masked_text: str
    entity_counts: dict[str, int] = Field(default_factory=dict)  # entity type -> distinct placeholder count
    token_map: dict[str, str] = Field(default_factory=dict, repr=False)  # "<PERSON_1>" -> original value

    def clear(self) -> None:
        self.token_map.clear()
