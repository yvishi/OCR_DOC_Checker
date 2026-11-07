"""Stage orchestration for increment 1: upload -> OCR -> route -> PII masking / image redaction.

The LLM call (phase 4), restore + rules (phase 5) and API (phase 6) plug in after `prepare`.
Until then the request ends here, so the token map is cleared and the upload deleted in `finally`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from app import telemetry as tel
from app.config import Settings, get_settings
from app.errors import ErrorClass, PipelineError
from app.ocr.tesseract import passes_text_gate, run_ocr_with_images
from app.pii.masker import mask_text
from app.pii.redact_image import redact_for_vision
from app.upload import delete_upload, store_upload


class Route(str, Enum):
    TEXT = "text"  # enough confident text: masked text goes to the LLM
    VISION = "vision"  # weak text but word boxes exist: redacted image goes to the LLM


@dataclass
class PreparedRequest:
    """Everything the LLM stage is allowed to see, plus local-only stats.

    Holds no unmasked text and no token map: those never leave `prepare`.
    """

    request_id: str
    route: Route
    file_extension: str
    page_count: int
    char_count: int
    mean_conf: float
    low_conf_word_count: int
    rotation_applied: list[int]
    masked_text: str | None = None
    redacted_images: list[np.ndarray] = field(default_factory=list, repr=False)
    entity_counts: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def used_vision_fallback(self) -> bool:
        return self.route is Route.VISION


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 1)


def prepare(data: bytes, settings: Settings | None = None) -> PreparedRequest:
    """Run every local stage for one uploaded document. Raises PipelineError (generic message)."""
    settings = settings or get_settings()
    request_id = uuid.uuid4().hex
    upload = None
    mask = None
    timings: dict[str, float] = {}
    try:
        with tel.request_span(request_id=request_id) as req:
            upload = store_upload(data, settings)
            req.set(file_extension=upload.extension)

            start = time.perf_counter()
            with tel.stage_span("ocr", ocr_engine="tesseract") as span:
                ocr, images = run_ocr_with_images(upload, settings)
                low_conf = sum(1 for w in ocr.words if w.conf < settings.field_min_conf)
                span.set(
                    page_count=ocr.page_count,
                    char_count=ocr.char_count,
                    mean_conf=round(ocr.mean_conf, 2),
                    low_conf_word_count=low_conf,
                    rotation_applied=ocr.pages[0].rotation_applied if ocr.pages else 0,
                    stamp_pixels_removed=sum(p.stamp_pixels_removed for p in ocr.pages),
                )
            timings["ocr"] = _ms(start)

            common = dict(
                request_id=request_id,
                file_extension=upload.extension,
                page_count=ocr.page_count,
                char_count=ocr.char_count,
                mean_conf=round(ocr.mean_conf, 2),
                low_conf_word_count=low_conf,
                rotation_applied=[p.rotation_applied for p in ocr.pages],
            )

            start = time.perf_counter()
            if passes_text_gate(ocr, settings):
                with tel.stage_span("pii") as span:
                    mask = mask_text(ocr.text)
                    span.set(
                        entity_counts=mask.entity_counts,
                        placeholder_count=sum(mask.entity_counts.values()),
                    )
                tel.record_entities(mask.entity_counts)
                prepared = PreparedRequest(
                    route=Route.TEXT,
                    masked_text=mask.masked_text,
                    entity_counts=dict(mask.entity_counts),
                    **common,
                )
            else:
                if not ocr.has_boxes:
                    raise PipelineError(ErrorClass.OCR_EMPTY)
                with tel.stage_span("vision", used_vision_fallback=True) as span:
                    redacted = redact_for_vision(images, ocr)
                tel.record_fallback()
                prepared = PreparedRequest(route=Route.VISION, redacted_images=redacted, **common)
            timings["pii"] = _ms(start)
            prepared.timings_ms = timings

            req.set(used_vision_fallback=prepared.used_vision_fallback, outcome="success")
        tel.record_request("success")
        tel.log_event(
            "request_completed",
            request_id=request_id,
            used_vision_fallback=prepared.used_vision_fallback,
            page_count=prepared.page_count,
        )
        return prepared
    except PipelineError as exc:
        tel.record_request("error")
        tel.log_event("request_failed", request_id=request_id, error_class=exc.error_class.value)
        raise
    except Exception:
        tel.record_request("error")
        tel.log_event("request_failed", request_id=request_id, error_class=ErrorClass.INTERNAL.value)
        raise PipelineError(ErrorClass.INTERNAL) from None
    finally:
        if mask is not None:
            mask.clear()
        if upload is not None:
            delete_upload(upload)


def main(argv: list[str] | None = None) -> int:
    """`python -m app.pipeline <file> [--show-masked]`: run the local stages and print stats."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="python -m app.pipeline", description=main.__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--show-masked", action="store_true", help="print the masked text the LLM would see")
    args = parser.parse_args(argv)
    try:
        prepared = prepare(args.file.read_bytes())
    except PipelineError as exc:
        print(f"error: {exc.client_message} ({exc.error_class.value})")
        return 1
    print(f"route:            {prepared.route.value}")
    print(f"pages:            {prepared.page_count}  rotation={prepared.rotation_applied}")
    print(f"char_count:       {prepared.char_count}")
    print(f"mean_conf:        {prepared.mean_conf:.1f}  (words < 80: {prepared.low_conf_word_count})")
    print(f"entity_counts:    {prepared.entity_counts}")
    print(f"timings_ms:       {prepared.timings_ms}")
    if args.show_masked and prepared.masked_text:
        print("\n--- masked text (what the LLM will see) ---\n" + prepared.masked_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
