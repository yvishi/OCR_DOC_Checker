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
        on_pct_line = [w for w in hits if re.search(r"percent|%", _line_of(ocr.words, w), re.IGNORECASE)]
        chosen = on_pct_line or hits
        if chosen:
            min_conf = max(w.conf for w in chosen)  # best reading of the number; each is one word
    total_matches, total_pct = _total_check(ocr.text, pct)
    figure_words = list(chosen) if core and chosen else []
    figure_words += [w for w in ocr.words if re.search(r"total|percent", _line_of(ocr.words, w), re.IGNORECASE)]
    return PercentageEvidence(grounded, min_conf, total_matches, total_pct, _stamped(figure_words, ocr))


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
