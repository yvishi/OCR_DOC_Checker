"""Tesseract word extraction and the text gate.

Privacy: this module never logs or prints OCR text.
"""

from __future__ import annotations

import numpy as np
import pytesseract
from pytesseract import Output

from app.config import Settings, get_settings
from app.errors import ErrorClass, PipelineError
from app.ocr.preprocess import preprocess
from app.ocr.render import load_pages
from app.schemas import OcrPage, OcrResult, OcrWord, StoredUpload

TESSERACT_CONFIG = "--oem 1 --psm 6"  # uniform block: keeps "Label: value" columns on one line
OCR_TIMEOUT_S = 120

_configured_cmd: str | None = None


def ensure_tesseract_cmd(settings: Settings | None = None) -> None:
    """Point pytesseract at the binary from settings (once, lazily)."""
    global _configured_cmd
    if _configured_cmd is not None and settings is None:
        return
    cmd = (settings or get_settings()).resolved_tesseract_cmd()
    pytesseract.pytesseract.tesseract_cmd = cmd
    _configured_cmd = cmd


def ocr_image(gray: np.ndarray, page_index: int = 0) -> list[OcrWord]:
    """Words (level 5) with non-blank text and conf >= 0."""
    ensure_tesseract_cmd()
    try:
        data = pytesseract.image_to_data(
            gray, config=TESSERACT_CONFIG, output_type=Output.DICT, timeout=OCR_TIMEOUT_S
        )
    except Exception:
        # from None: Tesseract stderr must never ride along in a traceback.
        raise PipelineError(ErrorClass.OCR_FAILED) from None
    words: list[OcrWord] = []
    for i, level in enumerate(data["level"]):
        if int(level) != 5:
            continue
        text = str(data["text"][i]).strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if not text or conf < 0:
            continue
        words.append(
            OcrWord(
                text=text,
                conf=conf,
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
                page=page_index,
                line_key=(int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i])),
            )
        )
    return words


def build_text(words: list[OcrWord]) -> str:
    """Spaces within a line, newline between lines, blank line between pages."""
    pages: dict[int, list[list[str]]] = {}
    last_key: dict[int, tuple[int, int, int]] = {}
    for w in words:
        lines = pages.setdefault(w.page, [])
        if not lines or last_key.get(w.page) != w.line_key:
            lines.append([])
            last_key[w.page] = w.line_key
        lines[-1].append(w.text)
    return "\n\n".join(
        "\n".join(" ".join(line) for line in pages[p]) for p in sorted(pages)
    )


def count_chars(text: str) -> int:
    return sum(1 for c in text if c.isalnum())


def run_ocr_with_images(
    upload: StoredUpload, settings: Settings | None = None
) -> tuple[OcrResult, list[np.ndarray]]:
    """OCR an upload and also return the processed page images.

    Word boxes are in the coordinates of these processed images (after rotation and any
    upscaling), so the vision fallback must redact these images, not the raw renders.
    """
    settings = settings or get_settings()
    ensure_tesseract_cmd(settings)
    images = load_pages(upload, dpi=settings.ocr_dpi, max_pages=settings.ocr_max_pages)
    words: list[OcrWord] = []
    pages: list[OcrPage] = []
    processed_images: list[np.ndarray] = []
    try:
        for index, img in enumerate(images):
            processed, meta = preprocess(img, index)
            pages.append(meta)
            processed_images.append(processed)
            words.extend(ocr_image(processed, index))
    except PipelineError:
        raise
    except Exception:
        raise PipelineError(ErrorClass.OCR_FAILED) from None
    text = build_text(words)
    mean_conf = sum(w.conf for w in words) / len(words) if words else 0.0
    result = OcrResult(
        words=words, text=text, pages=pages, mean_conf=mean_conf, char_count=count_chars(text)
    )
    return result, processed_images


def run_ocr(upload: StoredUpload, settings: Settings | None = None) -> OcrResult:
    return run_ocr_with_images(upload, settings)[0]


def passes_text_gate(result: OcrResult, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return result.char_count >= settings.ocr_min_chars and result.mean_conf >= settings.ocr_min_mean_conf
