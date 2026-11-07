"""Turn a stored upload into a list of BGR page images (numpy uint8, HxWx3)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from app.errors import ErrorClass, PipelineError
from app.schemas import StoredUpload

MAX_PIXELS = 60_000_000  # per page; guards against decompression bombs


def _rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _render_pdf(path, dpi: int, max_pages: int) -> list[np.ndarray]:
    import pymupdf

    pages: list[np.ndarray] = []
    with pymupdf.open(path) as doc:
        if doc.page_count == 0:
            raise ValueError("no pages")
        for i in range(min(doc.page_count, max_pages)):
            page = doc[i]
            scale = dpi / 72.0
            w, h = page.rect.width * scale, page.rect.height * scale
            if w * h > MAX_PIXELS:
                raise ValueError("page too large")
            pix = page.get_pixmap(dpi=dpi, alpha=False, colorspace=pymupdf.csRGB)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
            arr = arr[:, : pix.width * pix.n].reshape(pix.height, pix.width, pix.n)
            pages.append(_rgb_to_bgr(arr[:, :, :3]))
    return pages


def _render_image(path) -> list[np.ndarray]:
    with Image.open(path) as im:
        if im.width * im.height > MAX_PIXELS:
            raise ValueError("image too large")
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
            im = bg
        else:
            im = im.convert("RGB")
        return [_rgb_to_bgr(np.asarray(im, dtype=np.uint8))]


def load_pages(upload: StoredUpload, dpi: int = 300, max_pages: int = 3) -> list[np.ndarray]:
    """Render up to `max_pages` pages. Any failure raises PipelineError(RENDER_FAILED)."""
    try:
        if upload.extension == "pdf":
            pages = _render_pdf(upload.path, dpi, max_pages)
        elif upload.extension in ("png", "jpg"):
            pages = _render_image(upload.path)
        else:
            raise ValueError("unsupported kind")
    except PipelineError:
        raise
    except Exception as exc:  # includes Pillow DecompressionBombError
        raise PipelineError(ErrorClass.RENDER_FAILED) from exc
    if not pages:
        raise PipelineError(ErrorClass.RENDER_FAILED)
    return pages
