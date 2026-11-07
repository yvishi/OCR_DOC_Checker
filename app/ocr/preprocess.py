"""Page preprocessing: rotation fix, stamp-hue removal, binarization.

Each step is a small, pure function on numpy images (BGR uint8 in, BGR or gray uint8 out).
Nothing here logs or prints document content.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from app.schemas import OcrPage

# OSD orientation confidence needed before we rotate. Tesseract reports ~0.5-1.5 on pages
# where it is guessing and well above 5 on clearly rotated text pages; our upright and
# rotated fixtures land at roughly 5-30. 2.0 keeps false rotations on sparse pages rare.
OSD_MIN_CONF = 2.0
OSD_TIMEOUT_S = 30
# OSD only needs layout, not detail: downscale big pages to this long side to keep it fast.
OSD_MAX_SIDE = 2000

# Stamp mask (OpenCV HSV: H 0-179, S/V 0-255).
STAMP_MIN_SAT = 70  # grey/black text has S near 0; ink stamps are strongly saturated
STAMP_MIN_VAL = 60  # very dark pixels are text, even when slightly tinted
RED_HUE_LOW_MAX = 10
RED_HUE_HIGH_MIN = 160
BLUE_HUE = (95, 135)

# Upscale 2x when the median glyph is shorter than this (screenshots at screen resolution).
# Tesseract's LSTM models do best with ~20-40 px capitals; upscaling text that is already
# that size hurt accuracy on our fixtures ("76.40" -> "/6.40"), so it is conditional.
UPSCALE_MAX_GLYPH_PX = 18
UPSCALE_MAX_SIDE = 5000  # never upscale past this long side

_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_polarity(gray: np.ndarray) -> np.ndarray:
    """Invert light-on-dark pages (dark themes, negatives) so text is dark on light."""
    return cv2.bitwise_not(gray) if float(gray.mean()) < 127.0 else gray


def rotate(img: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate clockwise by 0/90/180/270 degrees."""
    code = _ROTATE_CODES.get(degrees % 360)
    return img if code is None else cv2.rotate(img, code)


def detect_rotation(img: np.ndarray) -> int:
    """Clockwise degrees needed to make the page upright (0 when unsure or OSD fails)."""
    from app.ocr.tesseract import ensure_tesseract_cmd

    ensure_tesseract_cmd()
    gray = normalize_polarity(to_gray(img))
    h, w = gray.shape[:2]
    scale = OSD_MAX_SIDE / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    try:
        osd = pytesseract.image_to_osd(
            gray, config="--psm 0", output_type=Output.DICT, timeout=OSD_TIMEOUT_S
        )
    except Exception:  # too little text, timeout, or OSD data missing
        return 0
    try:
        degrees = int(osd.get("rotate", 0)) % 360
        conf = float(osd.get("orientation_conf", 0.0))
    except (TypeError, ValueError):
        return 0
    if degrees not in _ROTATE_CODES or conf < OSD_MIN_CONF:
        return 0
    return degrees


def fix_rotation(img: np.ndarray) -> tuple[np.ndarray, int]:
    degrees = detect_rotation(img)
    return rotate(img, degrees), degrees


def stamp_mask(img: np.ndarray) -> np.ndarray:
    """uint8 mask (255 = stamp ink) of saturated red or blue pixels, slightly dilated."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    colourful = (s >= STAMP_MIN_SAT) & (v >= STAMP_MIN_VAL)
    red = (h <= RED_HUE_LOW_MAX) | (h >= RED_HUE_HIGH_MIN)
    blue = (h >= BLUE_HUE[0]) & (h <= BLUE_HUE[1])
    mask = (colourful & (red | blue)).astype(np.uint8) * 255
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)


def remove_stamp(img: np.ndarray) -> tuple[np.ndarray, int]:
    """Lift red/blue stamp ink off the page, keeping black/grey text underneath.

    Masked pixels are replaced by their brightest channel (HSV value) as grey. Stamp ink on
    white paper becomes near-white, while black text under a translucent stamp stays dark,
    and dark-blue printed text stays a readable mid-grey rather than being wiped out.
    Returns the cleaned image and the number of pixels that were lightened.
    """
    if img.ndim != 3:
        return img, 0
    mask = stamp_mask(img) > 0
    if not mask.any():
        return img, 0
    value = img.max(axis=2)
    gray_before = to_gray(img)
    out = img.copy()
    out[mask] = np.repeat(value[mask][:, None], 3, axis=1)
    removed = int(np.count_nonzero(value[mask].astype(np.int16) - gray_before[mask] > 20))
    return out, removed


STAMP_MIN_SIDE_PX = 60  # a seal is a large blob; coloured letters of printed text are small


def stamp_boxes(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of stamp-sized red/blue ink regions (pieces merged by a closing pass)."""
    if img.ndim != 3:
        return []
    mask = stamp_mask(img)
    if not mask.any():
        return []
    merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    page_h, page_w = mask.shape[:2]
    max_side = 0.6 * min(page_h, page_w)
    boxes = []
    for i in range(1, n):
        x, y, w, h, _area = (int(v) for v in stats[i])
        # Seals are roughly round and well under page size; coloured page borders, header bands
        # and rules are long and thin, or span the page.
        if min(w, h) < STAMP_MIN_SIDE_PX or max(w, h) > max_side or not 0.6 <= w / h <= 1.6:
            continue
        boxes.append((x, y, w, h))
    return boxes


def _otsu(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def median_glyph_height(bw: np.ndarray) -> float:
    """Median height of dark connected components (0 if none) in a dark-on-light 0/255 image."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(bw), connectivity=8)
    if n <= 1:
        return 0.0
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    heights = heights[(areas >= 8) & (heights >= 4) & (heights <= bw.shape[0] // 4)]
    return float(np.median(heights)) if heights.size else 0.0


def upscale_small(img: np.ndarray) -> np.ndarray:
    """2x cubic upscale when the text is small; otherwise unchanged."""
    h, w = img.shape[:2]
    if max(h, w) * 2 > UPSCALE_MAX_SIDE:
        return img
    glyph = median_glyph_height(_otsu(normalize_polarity(to_gray(img))))
    if glyph == 0.0 or glyph >= UPSCALE_MAX_GLYPH_PX:
        return img
    return cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)


def binarize(img: np.ndarray) -> np.ndarray:
    """Grayscale, fix polarity, light denoise, Otsu threshold. Returns 0/255 uint8."""
    gray = normalize_polarity(to_gray(img))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return _otsu(gray)


def preprocess(img: np.ndarray, page_index: int = 0) -> tuple[np.ndarray, OcrPage]:
    """Run rotation fix, stamp removal, upscaling and binarization on one page."""
    img, degrees = fix_rotation(img)
    boxes = stamp_boxes(img)
    img, removed = remove_stamp(img)
    before = img.shape[1]
    img = upscale_small(img)
    scale = img.shape[1] // before  # 1, or 2 when upscaled
    bw = binarize(img)
    h, w = bw.shape[:2]
    meta = OcrPage(
        page=page_index,
        width=w,
        height=h,
        rotation_applied=degrees,
        stamp_pixels_removed=removed,
        stamp_boxes=[(x * scale, y * scale, bw_ * scale, bh * scale) for x, y, bw_, bh in boxes],
    )
    return bw, meta
