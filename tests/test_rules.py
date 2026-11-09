"""Eligibility rule, guards and restore: boundary cases on hand-built OCR results."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.ocr.tesseract import build_text, count_chars
from app.rules import subjects
from app.rules.eligibility import REASONS, decide
from app.rules.guards import percentage_evidence, restore_fields
from app.schemas import CheckStatus, DocumentType, LearnSphereFields, LLMOutput, OcrPage, OcrResult, OcrWord
from app.telemetry import ALLOWED_KEYS

S = Settings()


def ocr_from_lines(lines: list[str], conf: float = 95.0, overrides: dict[str, float] | None = None) -> OcrResult:
    overrides = overrides or {}
    words = []
    for i, line in enumerate(lines):
        for j, token in enumerate(line.split()):
            words.append(OcrWord(text=token, conf=overrides.get(token, conf), left=j * 50, top=i * 30, width=40, height=20, line_key=(1, 1, i)))
    text = build_text(words)
    return OcrResult(words=words, text=text, pages=[OcrPage(page=0, width=1000, height=1000)], mean_conf=conf, char_count=count_chars(text))


def sheet(pct_printed: str | None, total: str | None = None, subject_rows: list[str] | None = None, **ov) -> OcrResult:
    lines = ["CENTRAL BOARD OF SECONDARY EDUCATION", "Name of Candidate: <PERSON_1>"]
    lines += subject_rows or []
    if total:
        lines.append(f"Grand Total: {total}")
    if pct_printed:
        lines.append(f"Percentage: {pct_printed}")
    return ocr_from_lines(lines, **ov)


def llm(pct: float | None, printed: str | None, doc=DocumentType.MARK_SHEET, cgpa=False, **fields) -> LLMOutput:
    return LLMOutput(
        document_type=doc,
        fields=LearnSphereFields(percentage=pct, **fields),
        percentage_as_printed=printed,
        cgpa_only=cgpa,
    )


def status(llm_out, ocr) -> CheckStatus:
    return decide(llm_out, ocr, S).status


# ---------- the 60% boundary ----------


@pytest.mark.parametrize(
    "printed,value,total,expected",
    [
        ("76.40%", 76.4, "382 / 500", CheckStatus.ELIGIBLE),
        ("48.20%", 48.2, "241 / 500", CheckStatus.NOT_ELIGIBLE),
        ("60%", 60.0, "300 / 500", CheckStatus.ELIGIBLE),
        ("60.0%", 60.0, "300 / 500", CheckStatus.ELIGIBLE),
        ("60.00%", 60.0, "300 / 500", CheckStatus.ELIGIBLE),
        ("59.90%", 59.9, "599 / 1000", CheckStatus.NOT_ELIGIBLE),
        ("59.99%", 59.99, "5999 / 10000", CheckStatus.NOT_ELIGIBLE),
        ("61.40%", 61.4, "307 / 500", CheckStatus.ELIGIBLE),
        ("58.60%", 58.6, "293 / 500", CheckStatus.NOT_ELIGIBLE),
    ],
)
def test_cutoff_boundaries(printed, value, total, expected):
    assert status(llm(value, printed), sheet(printed, total)) is expected


def test_reasons_never_contain_student_figures():
    check = decide(llm(76.4, "76.40%"), sheet("76.40%", "382 / 500"), S)
    assert "76" not in check.reason and "<" not in check.reason


# ---------- Needs review paths ----------


def test_cgpa_only_needs_review_and_is_never_converted():
    check = decide(llm(None, None, cgpa=True), ocr_from_lines(["CGPA: 8.2"]), S)
    assert check.status is CheckStatus.NEEDS_REVIEW and check.reason == REASONS["cgpa"]


@pytest.mark.parametrize(
    "line,value,printed",
    [
        ("Percentile: 92.50", 92.5, "92.50"),
        ("Percentile Rank: 92.50", 92.5, "92.50"),
        ("Percentile: 92.50%", 92.5, "92.50%"),
        ("Score: 92.50 %ile", 92.5, "92.50"),
        ("CGPA: 8.2", 8.2, "8.2"),
        ("SGPA: 8.20", 8.2, "8.20"),
        ("GPA: 3.80", 3.8, "3.80"),
        ("Grade Points: 9.1", 9.1, "9.1"),
    ],
)
def test_other_metric_extracted_as_percentage_is_never_decided(line, value, printed):
    check = decide(llm(value, printed), ocr_from_lines(["CENTRAL BOARD", line]), S)
    assert check.status is CheckStatus.NEEDS_REVIEW and check.reason == REASONS["other_metric"]


@pytest.mark.parametrize(
    "lines",
    [
        ["Percentile: 92.50"],
        ["SGPA: 8.20"],
        ["GPA: 3.80"],
        ["Code Subject Grade", "301 ENGLISH CORE A1", "041 MATHEMATICS B2"],
        ["Code Subject Grade", "301 ENGLISH CORE A1", "Note: 33 percent to pass in each subject"],
        ["ENG ENGLISH B1", "MAT MATHEMATICS C1", "PHY PHYSICS A2", "Result: PASS"],  # header lost by OCR
        ["Result: First Division"],
        ["Division: Second"],
    ],
)
def test_other_metric_with_no_percentage_gets_its_own_reason(lines):
    check = decide(llm(None, None), ocr_from_lines(["CENTRAL BOARD", *lines]), S)
    assert check.status is CheckStatus.NEEDS_REVIEW and check.reason == REASONS["other_metric"]


def test_marks_table_without_a_percentage_is_missing_not_other_metric():
    lines = ["Code Subject Max Marks Marks Obtained Grade", "301 ENGLISH CORE 100 076 B1"]
    assert decide(llm(None, None), ocr_from_lines(lines), S).reason == REASONS["missing"]


def test_unreadable_percentage_next_to_cgpa_is_missing():
    lines = ["CGPA: 8.2", "Percentage: ####"]
    assert decide(llm(None, None), ocr_from_lines(lines), S).reason == REASONS["missing"]


@pytest.mark.parametrize(
    "lines",
    [
        ["CGPA: 8.2", "Grand Total: 382 / 500", "Percentage: 76.40%"],
        ["CGPA: 8.2 Percentage: 76.40%", "Grand Total: 382 / 500"],
        ["Percentile: 91.00", "Grand Total: 382 / 500", "Percentage: 76.40%"],
    ],
)
def test_real_percentage_alongside_another_metric_is_decided_normally(lines):
    assert status(llm(76.4, "76.40%"), ocr_from_lines(lines)) is CheckStatus.ELIGIBLE


def test_percent_sign_after_the_figure_counts_as_a_label():
    assert status(llm(76.4, "76.40%"), ocr_from_lines(["Overall Result: 76.40%"])) is CheckStatus.ELIGIBLE


def test_figure_with_no_percentage_label_needs_review():
    # A subject mark picked up as the percentage on a sheet that prints none.
    ocr = ocr_from_lines(["301 ENGLISH CORE 100 82 B1"])
    assert decide(llm(82.0, "82"), ocr, S).reason == REASONS["unlabeled"]


def test_percentile_word_is_not_treated_as_the_percentage_line():
    ocr = ocr_from_lines(["Percentile: 76.40", "Percentage: 76.40"], overrides={"76.40": 95.0})
    assert percentage_evidence(llm(76.4, "76.40"), ocr).label == "percentage"


def test_other_metric_on_vision_route_stays_needs_review():
    assert decide(llm(92.5, "92.50"), None, S).reason == REASONS["vision"]


def test_missing_percentage_needs_review():
    assert decide(llm(None, None), sheet(None), S).reason == REASONS["missing"]


def test_hallucinated_percentage_not_in_text_needs_review():
    assert decide(llm(76.4, "76.40%"), sheet("67.40%", "337 / 500"), S).reason == REASONS["ungrounded"]


def test_printed_and_value_disagree_needs_review():
    assert decide(llm(76.4, "67.40%"), sheet("67.40%", "337 / 500"), S).reason == REASONS["ungrounded"]


def test_low_confidence_percentage_word_needs_review():
    # The stamped sample: OCR read "72.50%" (true 72.6) at conf 76.
    ocr = sheet("72.50%", "3263 / ENO", overrides={"72.50%": 76.0})
    assert decide(llm(72.5, "72.50%"), ocr, S).reason == REASONS["low_conf"]


def test_total_disagreeing_with_percentage_needs_review():
    assert decide(llm(76.4, "76.40%"), sheet("76.40%", "372 / 500"), S).reason == REASONS["total_mismatch"]


@pytest.mark.parametrize("value,printed", [(58.0, "58.00%"), (59.9, "59.90%"), (60.0, "60.00%"), (62.0, "62.00%")])
def test_near_cutoff_without_readable_total_needs_review(value, printed):
    assert decide(llm(value, printed), sheet(printed, total=None), S).reason == REASONS["near_cutoff"]


def test_outside_band_without_total_still_decided():
    assert status(llm(76.4, "76.40%"), sheet("76.40%", total=None)) is CheckStatus.ELIGIBLE
    assert status(llm(48.2, "48.20%"), sheet("48.20%", total=None)) is CheckStatus.NOT_ELIGIBLE


def test_subject_recompute_disagreement_needs_review():
    rows = ["$1 ENGLISH 100 75 B1", "$2 MATHEMATICS 100 51 C1", "$3 PHYSICS 100 38 D", "$4 CHEMISTRY 100 62 B2", "$5 BIOLOGY 100 90 A2"]
    ocr = sheet("66.00%", "330 / 500", subject_rows=rows)
    assert subjects.recompute(ocr.text).percentage == 63.2
    check = decide(llm(66.0, "66.00%"), ocr, S)
    assert check.reason == REASONS["subjects"]
    assert (check.printed_percentage, check.recomputed_percentage) == (66.0, 63.2)
    assert not any(ch.isdigit() for ch in check.reason)


def test_subject_mismatch_near_cutoff_without_total_still_shows_both_figures():
    rows = ["$1 ENGLISH 100 75 B1", "$2 MATHEMATICS 100 51 C1", "$3 PHYSICS 100 38 D", "$4 CHEMISTRY 100 62 B2", "$5 BIOLOGY 100 90 A2"]
    check = decide(llm(61.0, "61.00%"), sheet("61.00%", subject_rows=rows), S)
    assert check.reason == REASONS["subjects"] and check.recomputed_percentage == 63.2


def test_check_omits_figures_when_absent_and_keeps_them_out_of_telemetry():
    check = decide(llm(76.4, "76.40%"), sheet("76.40%", "382 / 500"), S)
    assert set(check.model_dump()) == {"status", "reason"}
    assert not {"printed_percentage", "recomputed_percentage"} & ALLOWED_KEYS


def test_subject_recompute_agreeing_passes():
    rows = ["301 ENGLISH CORE 100 076 B1", "041 MATHEMATICS 100 081 A2", "042 PHYSICS 100 083 A2", "043 CHEMISTRY 100 087 A2", "083 COMPUTER SCIENCE 100 055 C1"]
    ocr = sheet("76.40%", "382 / 500", subject_rows=rows)
    assert subjects.recompute(ocr.text).rows == 5
    assert status(llm(76.4, "76.40%"), ocr) is CheckStatus.ELIGIBLE


def test_too_few_subject_rows_are_ignored():
    assert subjects.recompute("301 ENGLISH CORE 100 076 B1\n041 MATHEMATICS 100 081 A2") is None


def test_vision_route_mark_sheet_always_needs_review():
    assert decide(llm(76.4, "76.40%"), None, S).reason == REASONS["vision"]


# ---------- document types ----------


def test_other_rejected_and_cards_receipts_not_applicable():
    o = ocr_from_lines(["Grocery bill"])
    assert status(llm(None, None, doc=DocumentType.OTHER), o) is CheckStatus.REJECTED
    assert status(llm(None, None, doc=DocumentType.STUDENT_ID_CARD), o) is CheckStatus.NOT_APPLICABLE
    assert status(llm(None, None, doc=DocumentType.FEE_RECEIPT), o) is CheckStatus.NOT_APPLICABLE


def test_evidence_prefers_word_on_percentage_line():
    ocr = sheet("60.00%", "300 / 500", subject_rows=["X ENGLISH 100 60.00 B1"], overrides={"60.00": 40.0})
    ev = percentage_evidence(llm(60.0, "60.00%"), ocr)
    assert ev.grounded and ev.min_conf == 95.0 and ev.total_matches is True


# ---------- restore ----------

TOKENS = {"<PERSON_1>": "GAGAN CHAUDHURI", "<ROLL_NUMBER_1>": "CBSE23-739457", "<PARENT_NAME_1>": "XAVIER CHAUDHURI"}
TEXT = "Name of Candidate: GAGAN CHAUDHURI\nRoll No.: CBSE23-739457\nFather's Name: XAVIER CHAUDHURI"


def test_restore_placeholders():
    f = restore_fields(LearnSphereFields(student_name="<PERSON_1>", roll_number="<ROLL_NUMBER_1>"), TOKENS, TEXT)
    assert (f.student_name, f.roll_number) == ("GAGAN CHAUDHURI", "CBSE23-739457")


@pytest.mark.parametrize(
    "name,roll",
    [
        ("<PARENT_NAME_1>", "<PERSON_1>"),  # wrong placeholder type for the field
        ("<PERSON_9>", "<ROLL_NUMBER_9>"),  # not in the map
        ("<PERSON_1> CHAUDHURI", "<ROLL_NUMBER_1>/989"),  # half-masked
        ("Ravi Menon", "XYZ-000"),  # raw values that never occur in the OCR text
    ],
)
def test_restore_drops_invalid_values(name, roll):
    f = restore_fields(LearnSphereFields(student_name=name, roll_number=roll), TOKENS, TEXT)
    assert f.student_name is None and f.roll_number is None


def test_restore_keeps_raw_value_grounded_in_ocr_text():
    f = restore_fields(LearnSphereFields(student_name="gagan  chaudhuri"), {}, TEXT)
    assert f.student_name == "gagan  chaudhuri"


def test_restore_vision_route_without_text_drops_raw_values():
    assert restore_fields(LearnSphereFields(student_name="Someone"), {}, None).student_name is None


def test_stamp_over_percentage_needs_review_even_when_read_cleanly():
    ocr = sheet("64.20%", "321 / 500")
    pct_word = next(w for w in ocr.words if w.text == "64.20%")
    seal = (pct_word.left - 20, pct_word.top - 20, 200, 200)
    ocr = ocr.model_copy(update={"pages": [OcrPage(page=0, width=1000, height=1000, stamp_boxes=[seal])]})
    assert decide(llm(64.2, "64.20%"), ocr, S).reason == REASONS["stamp"]


def test_stamp_elsewhere_on_page_is_ignored():
    ocr = sheet("64.20%", "321 / 500")
    ocr = ocr.model_copy(update={"pages": [OcrPage(page=0, width=1000, height=1000, stamp_boxes=[(900, 0, 90, 90)])]})
    assert status(llm(64.2, "64.20%"), ocr) is CheckStatus.ELIGIBLE
