"""Local checks between the LLM output and the OCR evidence. Pure functions, no network.

- restore: placeholders back to real values from the token map (name and roll number only).
- grounding: the printed percentage must really occur in the OCR text, and match the number.
- confidence: every OCR word that makes up that percentage must have conf >= field_min_conf.
- total cross-check: "Total: X / Y" printed on the sheet must agree with the percentage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas import LearnSphereFields, LLMOutput, OcrResult, OcrWord

_PLACEHOLDER = re.compile(r"^<([A-Z_]+)_(\d+)>$")
_ANY_PLACEHOLDER = re.compile(r"<[A-Z_]+_\d+>")
_NUMBER = re.compile(r"\d{1,3}(?:\.\d{1,3})?")
_TOTAL = re.compile(r"\btotal\b[^\n\d]{0,30}(\d{2,5})\s*/\s*(\d{2,5})\b", re.IGNORECASE)
_PCT_LINE = re.compile(r"percent(?!ile)|%(?!\s*t?ile\b)", re.IGNORECASE)

# What a figure on a line is labeled as. "other" is tried first so "percentile" is never a percentage.
_LABEL = re.compile(
    r"(?P<other>percentile|%\s*t?ile\b|\b[cso]?gpa\b|\bc\.\s?g\.\s?p\.\s?a\b|\bgrade\s*points?\b)"
    r"|(?P<percentage>\bpercentage\b|\bper\s?cent\b|\bpct\b)",
    re.IGNORECASE,
)
_PCT_FIELD = re.compile(r"\bpercentage\s*[:\-]", re.IGNORECASE)  # a field, not "33 percent" in the footnotes
_RESULT_ONLY = re.compile(
    r"\b(?:first|second|third|1st|2nd|3rd|pass)\s+(?:class|division)\b"
    r"|\bdivision\s*[:\-]?\s*(?:first|second|third|1st|2nd|3rd|i{1,3}\b)"
    r"|\b(?:overall|final|result)\s+grade\b",
    re.IGNORECASE,
)
_GRADE_TABLE_HEAD = re.compile(r"\b(?:subject|code)\b.*\bgrade\b", re.IGNORECASE)
# "<code> <SUBJECT NAME> <letter grade>" with no marks. OCR often drops the shaded table header.
_GRADE_ROW = re.compile(r"^\s*\S{1,4}\s+[A-Za-z][A-Za-z &.\-]{1,40}?\s+[A-E][12+]?\s*$")
_MIN_GRADE_ROWS = 3

# Which placeholder types may fill which field. A <PARENT_NAME_1> is never the student's name.
_ALLOWED = {"student_name": {"PERSON"}, "roll_number": {"ROLL_NUMBER"}}


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def restore_fields(fields: LearnSphereFields, token_map: dict[str, str], ocr_text: str | None) -> LearnSphereFields:
    """Swap placeholders for real values; drop anything that isn't grounded or is the wrong type.

    - "<PERSON_1>" in student_name -> token_map value.
    - A placeholder of the wrong type, or one not in the map -> None.
    - Free text that still contains a placeholder -> None (half-masked values are never shown).
    - A raw value (masking missed it, so the model saw it) is kept only if it occurs in the OCR text.
    """
    data = fields.model_dump()
    haystack = _norm(ocr_text) if ocr_text else None
    for key, allowed in _ALLOWED.items():
        value = data.get(key)
        if value is None:
            continue
        m = _PLACEHOLDER.match(value.strip())
        if m:
            data[key] = token_map.get(value.strip()) if m.group(1) in allowed else None
        elif _ANY_PLACEHOLDER.search(value):
            data[key] = None
        elif haystack is None or _norm(value) not in haystack:
            data[key] = None
    if data.get("board_or_school") and _ANY_PLACEHOLDER.search(data["board_or_school"]):
        data["board_or_school"] = None
    return LearnSphereFields.model_validate(data)


@dataclass(frozen=True)
class PercentageEvidence:
    grounded: bool  # printed form found in OCR text and equal to the extracted number
    min_conf: float | None  # lowest OCR confidence among the words carrying the number
    total_matches: bool | None  # None when no readable "Total: X / Y" line exists
    total_pct: float | None
    stamp_over_figures: bool = False  # a stamp overlaps the percentage or total words
    label: str | None = None  # what the document calls this figure: "percentage", "other" or None


def _core(printed: str) -> str | None:
    m = _NUMBER.search(printed)
    return m.group(0) if m else None


def _line_of(words: list[OcrWord], w: OcrWord) -> str:
    return " ".join(x.text for x in words if x.page == w.page and x.line_key == w.line_key)


def percentage_evidence(llm: LLMOutput, ocr: OcrResult | None) -> PercentageEvidence:
    pct = llm.fields.percentage
    printed = llm.percentage_as_printed
    if pct is None or not printed or ocr is None:
        return PercentageEvidence(False, None, None, None)

    core = _core(printed)
    grounded = bool(core) and abs(float(core) - pct) < 0.005 and _norm(printed) in _norm(ocr.text)

    min_conf = None
    chosen: list[OcrWord] = []
    if core:
        # Words whose digits are exactly the number; prefer ones on a "percentage" line.
        hits = [w for w in ocr.words if _NUMBER.fullmatch(w.text.strip("%:;,()")) and w.text.strip("%:;,()") == core]
        on_pct_line = [w for w in hits if _PCT_LINE.search(_line_of(ocr.words, w))]
        chosen = on_pct_line or hits
        if chosen:
            min_conf = max(w.conf for w in chosen)  # best reading of the number; each is one word
    total_matches, total_pct = _total_check(ocr.text, pct)
    figure_words = list(chosen) if core and chosen else []
    figure_words += [w for w in ocr.words if re.search(r"total|percent", _line_of(ocr.words, w), re.IGNORECASE)]
    label = figure_label(core, ocr.text) if core else None
    return PercentageEvidence(grounded, min_conf, total_matches, total_pct, _stamped(figure_words, ocr), label)


def _label_at(line: str, start: int, end: int) -> str | None:
    """The nearest label before the figure, else a % sign or label right after it."""
    before = list(_LABEL.finditer(line[:start]))
    if before:
        return before[-1].lastgroup
    after = line[end:].lstrip(" :=-(")
    following = _LABEL.match(after)
    if following:
        return following.lastgroup
    return "percentage" if after.startswith("%") else None


def figure_label(core: str, text: str) -> str | None:
    """"percentage" if the figure is labeled as one anywhere; "other" if only as another metric."""
    figure = re.compile(rf"(?<![\d.]){re.escape(core)}(?!\d|\.\d)")
    found = None
    for line in text.splitlines():
        for m in figure.finditer(line):
            kind = _label_at(line, m.start(), m.end())
            if kind == "percentage":
                return kind
            found = kind or found
    return found


def non_percentage_metric_shown(text: str) -> bool:
    """True when a sheet with no percentage field reports its result some other way."""
    lines = text.splitlines()
    if any(_PCT_FIELD.search(line) for line in lines):
        return False
    if any(m.lastgroup == "other" for line in lines for m in _LABEL.finditer(line)):
        return True
    if _RESULT_ONLY.search(text):
        return True
    if sum(1 for line in lines if _GRADE_ROW.match(line)) >= _MIN_GRADE_ROWS:
        return True
    return any(_GRADE_TABLE_HEAD.search(line) and "marks" not in line.lower() for line in lines)


def _overlaps(w: OcrWord, box: tuple[int, int, int, int]) -> bool:
    x, y, bw, bh = box
    return w.left < x + bw and x < w.left + w.width and w.top < y + bh and y < w.top + w.height


def _stamped(words: list[OcrWord], ocr: OcrResult) -> bool:
    boxes = {p.page: p.stamp_boxes for p in ocr.pages}
    return any(_overlaps(w, b) for w in words for b in boxes.get(w.page, []))


def _total_check(text: str, pct: float) -> tuple[bool | None, float | None]:
    for line in text.splitlines():
        m = _TOTAL.search(line)
        if not m:
            continue
        got, maximum = int(m.group(1)), int(m.group(2))
        if maximum == 0 or got > maximum:
            return None, None  # unreadable (e.g. a stamp turned "/ 500" into "/ ENO" or worse)
        total_pct = round(got / maximum * 100, 2)
        return abs(total_pct - pct) <= 0.05, total_pct
    return None, None
