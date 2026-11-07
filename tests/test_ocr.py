import io
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.config import Settings
from app.errors import GENERIC_MESSAGE, ErrorClass, PipelineError
from app.ocr.preprocess import binarize, fix_rotation, remove_stamp, stamp_boxes
from app.ocr.render import load_pages
from app.ocr.tesseract import ocr_image, passes_text_gate, run_ocr
from app.schemas import StoredUpload
from app.upload import store_upload

LINES = [
    "LearnSphere College Pune",
    "Board Mark Sheet 2026",
    "Student Roll Number 4471",
    "Percentage 76.40",
    "Result Pass with distinction",
    "Total Marks 382 out of 500",
]


def _font(size: int = 40):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    pytest.skip("no TrueType font available")


def _page(lines=LINES) -> Image.Image:
    im = Image.new("RGB", (1600, 1000), "white")
    d = ImageDraw.Draw(im)
    font = _font()
    for i, line in enumerate(lines):
        d.text((100, 100 + i * 130), line, fill="black", font=font)
    return im


def _png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _bgr(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"))[:, :, ::-1].copy()


def _stamped(im: Image.Image) -> Image.Image:
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    # Translucent red seal over the "Percentage" and "Result" lines.
    d.ellipse((250, 380, 750, 780), fill=(220, 20, 30, 100), outline=(200, 0, 0, 140), width=8)
    return Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")


@pytest.fixture
def settings(tmp_path):
    return Settings(upload_dir=tmp_path)


def _words(text: str) -> str:
    return " ".join(text.split())


def test_ocr_reads_fixture(settings):
    up = store_upload(_png(_page()), settings)
    result = run_ocr(up, settings)
    flat = _words(result.text)
    assert "LearnSphere" in flat
    assert "Percentage 76.40" in flat
    assert result.mean_conf > 80
    assert result.char_count > 100
    assert result.page_count == 1
    assert result.pages[0].rotation_applied == 0
    assert passes_text_gate(result, settings)
    # one line per drawn line, words carry boxes and line keys
    assert len(result.text.splitlines()) == len(LINES)
    assert all(w.width > 0 and w.height > 0 for w in result.words)


@pytest.mark.parametrize(
    "transpose,expected",
    [(Image.Transpose.ROTATE_90, 90), (Image.Transpose.ROTATE_180, 180), (Image.Transpose.ROTATE_270, 270)],
)
def test_rotation_corrected(settings, transpose, expected):
    # PIL ROTATE_90 is counter-clockwise, so the fix is 90 degrees clockwise.
    up = store_upload(_png(_page().transpose(transpose)), settings)
    result = run_ocr(up, settings)
    assert result.pages[0].rotation_applied == expected
    flat = _words(result.text)
    assert "LearnSphere" in flat
    assert "Percentage 76.40" in flat


def test_fix_rotation_leaves_blank_page_alone():
    blank = np.full((600, 800, 3), 255, np.uint8)
    out, deg = fix_rotation(blank)
    assert deg == 0
    assert out is blank


def test_stamp_removed_and_text_kept(settings):
    stamped = _bgr(_stamped(_page()))
    cleaned, removed = remove_stamp(stamped)
    assert removed > 10_000
    # stamp area away from text is now near-white
    assert cleaned[600, 650].min() > 230
    # black glyph pixels outside the stamp are untouched
    untouched = _bgr(_page())
    black = (untouched.max(axis=2) < 50)
    black[380:790, 240:760] = False
    assert np.array_equal(cleaned[black], stamped[black])

    up = store_upload(_png(_stamped(_page())), settings)
    result = run_ocr(up, settings)
    assert result.pages[0].stamp_pixels_removed > 0
    flat = _words(result.text)
    assert "Percentage 76.40" in flat
    assert "LearnSphere" in flat


def test_remove_stamp_keeps_grey_and_dark_blue_text():
    img = np.full((50, 50, 3), 255, np.uint8)
    img[10:20, 10:20] = (40, 40, 40)  # dark grey text
    img[30:40, 10:20] = (90, 20, 10)  # navy (BGR) printed text
    out, _ = remove_stamp(img)
    assert np.array_equal(out[10:20, 10:20], img[10:20, 10:20])
    assert out[30:40, 10:20].max() <= 90  # stays dark enough to read


def test_no_stamp_no_change():
    img = _bgr(_page())
    out, removed = remove_stamp(img)
    assert removed == 0
    assert out is img


def test_binarize_inverts_dark_background():
    img = np.full((100, 200, 3), 20, np.uint8)
    cv2.putText(img, "Hi", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (230, 230, 230), 5)
    bw = binarize(img)
    assert set(np.unique(bw)) <= {0, 255}
    assert bw.mean() > 127  # background is white after inversion


def test_pdf_with_text(settings):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for i, line in enumerate(LINES):
        page.insert_text((60, 90 + i * 40), line, fontsize=18)
    data = doc.tobytes()
    doc.close()
    up = store_upload(data, settings)
    assert up.extension == "pdf"
    result = run_ocr(up, settings)
    flat = _words(result.text)
    assert "LearnSphere" in flat
    assert "Percentage 76.40" in flat
    assert result.mean_conf > 80


def test_pdf_with_embedded_image_and_page_cap(settings, tmp_path):
    doc = pymupdf.open()
    png = _png(_page())
    for _ in range(5):
        page = doc.new_page(width=800, height=500)
        page.insert_image(page.rect, stream=png)
    data = doc.tobytes()
    doc.close()
    up = store_upload(data, Settings(upload_dir=tmp_path, ocr_max_pages=2, ocr_dpi=200))
    pages = load_pages(up, dpi=200, max_pages=2)
    assert len(pages) == 2
    assert pages[0].dtype == np.uint8 and pages[0].ndim == 3 and pages[0].shape[2] == 3
    result = run_ocr(up, Settings(upload_dir=tmp_path, ocr_max_pages=2, ocr_dpi=200))
    assert result.page_count == 2
    assert "\n\n" in result.text  # blank line between pages
    assert {w.page for w in result.words} == {0, 1}


def test_exif_orientation_honoured(tmp_path):
    im = _page().transpose(Image.Transpose.ROTATE_90)  # stored sideways
    exif = Image.Exif()
    exif[0x0112] = 8  # "rotate 90 CW to view"
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif, quality=95)
    up = store_upload(buf.getvalue(), Settings(upload_dir=tmp_path))
    (page,) = load_pages(up)
    assert page.shape[:2] == (1000, 1600)


def test_blank_image_fails_gate(settings):
    im = Image.new("RGB", (1600, 1000), "white")
    ImageDraw.Draw(im).text((100, 100), "ok", fill="black", font=_font())
    up = store_upload(_png(im), settings)
    result = run_ocr(up, settings)
    assert result.char_count < settings.ocr_min_chars
    assert not passes_text_gate(result, settings)


def test_empty_result_mean_conf_zero(settings):
    up = store_upload(_png(Image.new("RGB", (800, 600), "white")), settings)
    result = run_ocr(up, settings)
    assert result.words == [] and result.mean_conf == 0 and not result.has_boxes
    assert not passes_text_gate(result, settings)


@pytest.mark.parametrize(
    "data,ext",
    [
        (b"%PDF-1.4\nthis is not really a pdf", "pdf"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00garbage" * 20, "png"),
        (b"\xff\xd8\xff" + b"garbage" * 20, "jpg"),
    ],
)
def test_garbage_bytes_render_failed(settings, data, ext):
    up = store_upload(data, settings)
    assert up.extension == ext
    with pytest.raises(PipelineError) as err:
        run_ocr(up, settings)
    assert err.value.error_class is ErrorClass.RENDER_FAILED
    assert str(err.value) == GENERIC_MESSAGE
    assert err.value.client_message == GENERIC_MESSAGE


def test_decompression_bomb_rejected(tmp_path, monkeypatch):
    import app.ocr.render as render

    monkeypatch.setattr(render, "MAX_PIXELS", 1000)
    up = store_upload(_png(Image.new("RGB", (100, 100), "white")), Settings(upload_dir=tmp_path))
    with pytest.raises(PipelineError) as err:
        load_pages(up)
    assert err.value.error_class is ErrorClass.RENDER_FAILED


def test_tesseract_failure_is_ocr_failed(monkeypatch):
    import app.ocr.tesseract as tess

    def boom(*a, **k):
        raise tess.pytesseract.TesseractError(1, "secret text from stderr")

    monkeypatch.setattr(tess.pytesseract, "image_to_data", boom)
    with pytest.raises(PipelineError) as err:
        ocr_image(np.full((10, 10), 255, np.uint8))
    assert err.value.error_class is ErrorClass.OCR_FAILED
    assert str(err.value) == GENERIC_MESSAGE
    assert err.value.__cause__ is None and err.value.__suppress_context__


def test_unknown_extension_render_failed(tmp_path):
    up = StoredUpload(upload_id="x", path=tmp_path / "x.gif", extension="gif", size_bytes=1)
    with pytest.raises(PipelineError) as err:
        load_pages(up)
    assert err.value.error_class is ErrorClass.RENDER_FAILED


def test_stamp_boxes_find_seal_but_not_borders_or_bands():
    im = _page()
    d = ImageDraw.Draw(im)
    d.rectangle((10, 10, im.width - 10, im.height - 10), outline=(20, 40, 200), width=6)  # blue page border
    d.rectangle((40, 30, im.width - 40, 70), fill=(200, 30, 30))  # red header band
    boxes = stamp_boxes(_bgr(_stamped(im)))
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert 230 <= x <= 270 and 360 <= y <= 400 and 0.6 <= w / h <= 1.6


def test_stamp_boxes_reported_in_processed_coordinates(settings):
    upload = store_upload(_png(_stamped(_page())), settings)
    page = run_ocr(upload, settings).pages[0]
    assert page.stamp_boxes, "seal not reported"
    x, y, w, h = page.stamp_boxes[0]
    assert x + w <= page.width and y + h <= page.height
