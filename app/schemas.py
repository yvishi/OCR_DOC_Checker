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
    # Stamp/seal regions (left, top, width, height) in processed-image pixels, same space as OcrWord.
    stamp_boxes: list[tuple[int, int, int, int]] = Field(default_factory=list)


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


# ---------- LLM output (phase 4) ----------


class LearnSphereFields(BaseModel):
    """The five extracted fields. All optional: null when absent or unreadable.

    On the text route, student_name and roll_number come back as placeholders (e.g. "<PERSON_1>");
    they are restored locally from the token map in phase 5.
    """

    model_config = ConfigDict(extra="forbid")

    student_name: str | None = Field(default=None, max_length=120)
    roll_number: str | None = Field(default=None, max_length=60)
    board_or_school: str | None = Field(default=None, max_length=120)
    passing_year: int | None = Field(default=None, ge=1950, le=2100)
    percentage: float | None = Field(default=None, ge=0, le=100)


class LLMOutput(BaseModel):
    """What the single classify-and-extract call must return. Anything else is schema_invalid."""

    model_config = ConfigDict(extra="forbid")

    document_type: DocumentType
    fields: LearnSphereFields
    # The percentage exactly as printed (e.g. "76.40%"), copied from the input. The grounding
    # guard (phase 5) checks it really occurs in the OCR text; null when there is no percentage.
    percentage_as_printed: str | None = Field(default=None, max_length=20)
    # True when the document shows only CGPA / grade points. CGPA is never converted.
    cgpa_only: bool = False


# ---------- API response (docs/ARCHITECTURE.md §6) ----------


class CheckStatus(str, Enum):
    ELIGIBLE = "Eligible"
    NOT_ELIGIBLE = "Not eligible"
    NEEDS_REVIEW = "Needs review"
    NOT_APPLICABLE = "Not applicable"
    REJECTED = "Rejected"


class Check(BaseModel):
    status: CheckStatus
    reason: str  # never contains names, IDs or numbers that identify a student


class PiiSummary(BaseModel):
    masked_before_llm: bool
    entity_counts: dict[str, int] = Field(default_factory=dict)


class PipelineSummary(BaseModel):
    ocr_engine: str = "tesseract"
    used_vision_fallback: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)  # ocr, pii, llm, rule


class ExtractResponse(BaseModel):
    """The one object returned to both the web page and POST /v1/extract."""

    request_id: str
    document_type: DocumentType
    fields: LearnSphereFields  # restored locally: real name and roll number, never placeholders
    check: Check
    pii: PiiSummary
    pipeline: PipelineSummary
