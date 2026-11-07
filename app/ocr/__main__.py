"""Dev CLI: `python -m app.ocr <file> [--show-text]`. Prints OCR stats; text only on request."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.errors import PipelineError
from app.ocr.tesseract import passes_text_gate, run_ocr
from app.upload import delete_upload, store_upload

LOW_CONF = 80.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ocr", description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--show-text", action="store_true", help="also print the OCR text (local dev only)")
    args = parser.parse_args(argv)

    settings = get_settings()
    upload = None
    try:
        upload = store_upload(args.file.read_bytes(), settings)
        t0 = time.perf_counter()
        result = run_ocr(upload, settings)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except PipelineError as err:
        print(f"error: {err.client_message} (class={err.error_class.value})", file=sys.stderr)
        return 1
    except OSError:
        print("error: cannot read input file", file=sys.stderr)
        return 1
    finally:
        if upload is not None:
            delete_upload(upload)

    low = sum(1 for w in result.words if w.conf < LOW_CONF)
    print(f"extension:        {upload.extension}")
    print(f"pages:            {result.page_count}")
    for p in result.pages:
        print(
            f"  page {p.page}: {p.width}x{p.height}  rotation={p.rotation_applied}  "
            f"stamp_pixels_removed={p.stamp_pixels_removed}"
        )
    print(f"words:            {len(result.words)}")
    print(f"char_count:       {result.char_count}")
    print(f"mean_conf:        {result.mean_conf:.1f}")
    print(f"words_conf<{LOW_CONF:.0f}:    {low}")
    print(f"text_gate:        {'pass' if passes_text_gate(result, settings) else 'fail'}")
    print(f"ocr_ms:           {elapsed_ms:.0f}")
    if args.show_text:
        print("----- text -----")
        print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
