"""Error classes. Clients only ever see GENERIC_MESSAGE; logs only ever see the class."""

from __future__ import annotations

from enum import Enum

GENERIC_MESSAGE = "Could not read this document"


class ErrorClass(str, Enum):
    UPLOAD_TOO_LARGE = "upload_too_large"
    UPLOAD_BAD_TYPE = "upload_bad_type"
    UPLOAD_EMPTY = "upload_empty"
    RENDER_FAILED = "render_failed"
    OCR_FAILED = "ocr_failed"
    OCR_EMPTY = "ocr_empty"  # low text and no word boxes: manual review
    PII_FAILED = "pii_failed"
    LLM_FAILED = "llm_failed"
    SCHEMA_INVALID = "schema_invalid"
    INTERNAL = "internal"


class PipelineError(Exception):
    """Raised by any stage. The message is always generic, so it can never leak document text."""

    def __init__(self, error_class: ErrorClass) -> None:
        super().__init__(GENERIC_MESSAGE)
        self.error_class = error_class

    @property
    def client_message(self) -> str:
        return GENERIC_MESSAGE
