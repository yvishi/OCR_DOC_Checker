"""Paint out PII word boxes before a page image may leave the machine (vision fallback)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.errors import ErrorClass, PipelineError
from app.pii.masker import find_pii_words
from app.schemas import OcrResult, OcrWord


def redact_boxes(img: np.ndarray, words: Sequence[OcrWord], pad: int = 6) -> np.ndarray:
    """Return a copy of ``img`` with a solid black, padded rectangle over each word box.

    Boxes are clipped to the image bounds. Works for grayscale (H, W) and colour (H, W, C)
    arrays; every channel is set to 0.
    """
    out = np.array(img, copy=True)
    if out.ndim < 2:
        raise PipelineError(ErrorClass.PII_FAILED)
    h, w = out.shape[:2]
    pad = max(0, int(pad))
    for word in words:
        x0 = max(0, word.left - pad)
        y0 = max(0, word.top - pad)
        x1 = min(w, word.left + word.width + pad)
        y1 = min(h, word.top + word.height + pad)
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = 0
    return out


def redact_for_vision(page_images: Sequence[np.ndarray], ocr: OcrResult, pad: int = 6) -> list[np.ndarray]:
    """Redacted copies of ``page_images`` (index == ``OcrWord.page``), PII boxes blacked out.

    Without word boxes we cannot know where the PII is, so the hosted model must not be called:
    raise ``OCR_EMPTY`` (manual review). The same holds per page: a page with no word boxes
    cannot be redacted, so it is left out of the returned list.
    """
    if not ocr.has_boxes:
        raise PipelineError(ErrorClass.OCR_EMPTY)
    pages_with_boxes = {word.page for word in ocr.words}
    pii_words = find_pii_words(ocr)
    by_page: dict[int, list[OcrWord]] = {}
    for word in pii_words:
        by_page.setdefault(word.page, []).append(word)
    return [
        redact_boxes(img, by_page.get(i, []), pad=pad)
        for i, img in enumerate(page_images)
        if i in pages_with_boxes
    ]
