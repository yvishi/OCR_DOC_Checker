"""Phase 3: local PII masking, span -> OCR word mapping and image redaction. Synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest

from app.errors import ErrorClass, PipelineError
from app.pii import masker
from app.pii.masker import build_text_with_offsets, find_pii_words, mask_text
from app.pii.redact_image import redact_boxes, redact_for_vision
from app.schemas import OcrPage, OcrResult, OcrWord


@pytest.fixture(scope="session", autouse=True)
def analyzer():
    """Load spaCy lg / Presidio once for the whole session."""
    return masker.get_analyzer()


MARK_SHEET = """CENTRAL BOARD OF SECONDARY EDUCATION
Central Board of Secondary Education
Statement of Marks
Secondary School Examination 2024
Name of Candidate: RAVI MENON
Roll No: CBSE24-104482
Father's Name: Suresh Menon
Mother's Name: Lakshmi Menon
Date of Birth: 14/08/2008
Aadhaar No: 2345 6789 0123
PAN: ABCDE1234F
Mobile: +91 98765 43210
Email: ravi.menon@example.com
Address: 12 MG Road, Kothrud, Pune 411038
School: LearnSphere College
Subject Marks
English 78
Hindi 72
Mathematics 81
Science 76
Social Science 75
Total 382 / 500
Percentage: 76.40%
Year of Passing: 2024
Result: PASS
Ravi  Menon has passed the examination.
Date of Issue: 20/05/2024
"""

PII_STRINGS = [
    "RAVI MENON",
    "Ravi  Menon",
    "Menon",
    "CBSE24-104482",
    "Suresh",
    "Lakshmi",
    "14/08/2008",
    "2345 6789 0123",
    "ABCDE1234F",
    "98765 43210",
    "ravi.menon@example.com",
    "12 MG Road",
    "Kothrud",
    "411038",
]

KEEP_STRINGS = [
    "CENTRAL BOARD OF SECONDARY EDUCATION",
    "Central Board of Secondary Education",
    "Statement of Marks",
    "Secondary School Examination 2024",
    "LearnSphere College",
    "English 78",
    "Hindi 72",
    "Mathematics 81",
    "Science 76",
    "Social Science 75",
    "Total 382 / 500",
    "Percentage: 76.40%",
    "Year of Passing: 2024",
    "Result: PASS",
    "Date of Issue: 20/05/2024",
]


# ---------- mask_text ----------


def test_mark_sheet_pii_masked_and_facts_survive():
    r = mask_text(MARK_SHEET)
    for s in PII_STRINGS:
        assert s not in r.masked_text, s
    for s in KEEP_STRINGS:
        assert s in r.masked_text, s
    # labels survive, only the values are replaced
    assert "Roll No: <ROLL_NUMBER_1>" in r.masked_text
    assert "Name of Candidate: <PERSON_1>" in r.masked_text
    assert "Father's Name: <PARENT_NAME_1>" in r.masked_text
    assert "Mother's Name: <PARENT_NAME_2>" in r.masked_text
    assert "Date of Birth: <DATE_OF_BIRTH_1>" in r.masked_text
    assert "Aadhaar No: <IN_AADHAAR_1>" in r.masked_text
    assert "PAN: <IN_PAN_1>" in r.masked_text
    assert "Mobile: <IN_PHONE_1>" in r.masked_text
    assert "Email: <EMAIL_ADDRESS_1>" in r.masked_text
    assert "Address: <ADDRESS_1>" in r.masked_text


def test_token_map_and_counts():
    r = mask_text(MARK_SHEET)
    assert r.token_map["<PERSON_1>"] == "RAVI MENON"  # first surface form
    assert r.token_map["<ROLL_NUMBER_1>"] == "CBSE24-104482"
    assert r.token_map["<PARENT_NAME_1>"] == "Suresh Menon"
    assert r.token_map["<PARENT_NAME_2>"] == "Lakshmi Menon"
    assert r.token_map["<IN_AADHAAR_1>"] == "2345 6789 0123"
    assert r.entity_counts == {
        "PERSON": 1,
        "ROLL_NUMBER": 1,
        "PARENT_NAME": 2,
        "DATE_OF_BIRTH": 1,
        "IN_AADHAAR": 1,
        "IN_PAN": 1,
        "IN_PHONE": 1,
        "EMAIL_ADDRESS": 1,
        "ADDRESS": 1,
    }
    # counts are distinct placeholders, and every placeholder is in the map
    assert sum(r.entity_counts.values()) == len(r.token_map)
    for token in r.token_map:
        assert token in r.masked_text
    r.clear()
    assert r.token_map == {}


def test_same_name_different_case_and_spacing_same_token():
    text = "Name: RAVI MENON\nRavi  Menon signed here.\nRavi Menon, again.\nRoll No: 104482"
    r = mask_text(text)
    assert r.masked_text.count("<PERSON_1>") == 3
    assert "<PERSON_2>" not in r.masked_text
    assert r.entity_counts["PERSON"] == 1


def test_two_names_get_different_tokens():
    r = mask_text("Name: Aisha Rahman\nFather's Name: Imran Rahman\nGuardian: Farah Khan")
    assert r.token_map["<PERSON_1>"] == "Aisha Rahman"
    assert r.token_map["<PARENT_NAME_1>"] == "Imran Rahman"
    assert r.token_map["<PARENT_NAME_2>"] == "Farah Khan"
    assert r.entity_counts == {"PERSON": 1, "PARENT_NAME": 2}


def test_numbering_restarts_per_call():
    a = mask_text("Name: Aisha Rahman\nRoll No: M123456")
    b = mask_text("Name: Kiran Rao\nRoll No: M654321")
    assert a.token_map == {"<PERSON_1>": "Aisha Rahman", "<ROLL_NUMBER_1>": "M123456"}
    assert b.token_map == {"<PERSON_1>": "Kiran Rao", "<ROLL_NUMBER_1>": "M654321"}


def test_id_normalisation_groups_aadhaar_variants():
    r = mask_text("Aadhaar: 2345 6789 0123\nUID 2345-6789-0123\nAadhaar 234567890123")
    assert r.masked_text.count("<IN_AADHAAR_1>") == 3
    assert r.entity_counts == {"IN_AADHAAR": 1}


def test_receipt_style_identifiers():
    text = (
        "Student Name: Aisha Rahman Roll No: M123456\n"
        "Seat No. 4412873  Enrollment No 20241234\n"
        "Bank A/c No: 123456789012 IFSC: SBIN0001234 UPI aisha.r@okaxis\n"
        "Policy No: LIC/2024/778899\n"
        "Contact 9876543210  DOB 03 Mar 2007\n"
        "Amount Paid: Rs. 45,000  Date: 12/06/2024"
    )
    r = mask_text(text)
    for s in ["Aisha", "M123456", "4412873", "20241234", "123456789012", "SBIN0001234",
              "aisha.r@okaxis", "LIC/2024/778899", "9876543210", "03 Mar 2007"]:
        assert s not in r.masked_text, s
    assert "Rs. 45,000" in r.masked_text
    assert "Date: 12/06/2024" in r.masked_text  # not a DOB
    assert r.entity_counts["ROLL_NUMBER"] == 3
    assert r.entity_counts["BANK_ACCOUNT"] == 3
    assert r.entity_counts["POLICY_NUMBER"] == 1


def test_no_pii_text_unchanged():
    text = "Statement of Marks\nMaharashtra State Board of Secondary and Higher Secondary Education\nPercentage: 81.20%\nYear of Passing: 2023\nICSE"
    r = mask_text(text)
    assert r.masked_text == text
    assert r.entity_counts == {} and r.token_map == {}


def test_empty_text():
    r = mask_text("")
    assert r.masked_text == "" and r.token_map == {}


def test_analyzer_failure_is_pii_failed(monkeypatch):
    class Boom:
        def analyze(self, **kwargs):
            raise RuntimeError("secret text RAVI MENON")

    monkeypatch.setattr(masker, "get_analyzer", lambda: Boom())
    with pytest.raises(PipelineError) as ei:
        mask_text("Name: Ravi Menon")
    assert ei.value.error_class is ErrorClass.PII_FAILED
    assert "RAVI" not in str(ei.value)
    assert ei.value.__cause__ is None


# ---------- OCR fixtures ----------

_CHAR_W = 10


def _ocr_from_lines(pages: list[list[str]], line_h: int = 30) -> OcrResult:
    """Hand-built OcrResult: one word box per token, 10 px per character, 30 px per line."""
    words: list[OcrWord] = []
    for p, lines in enumerate(pages):
        for li, line in enumerate(lines):
            x = 20
            for token in line.split(" "):
                words.append(
                    OcrWord(text=token, conf=95.0, left=x, top=20 + li * line_h, width=len(token) * _CHAR_W,
                            height=20, page=p, line_key=(1, 1, li + 1))
                )
                x += (len(token) + 1) * _CHAR_W
    text = "\n\n".join("\n".join(lines) for lines in pages)
    ocr_pages = [OcrPage(page=p, width=600, height=400) for p in range(len(pages))]
    chars = sum(c.isalnum() for c in text)
    return OcrResult(words=words, text=text, pages=ocr_pages, mean_conf=95.0 if words else 0.0, char_count=chars)


PAGE0 = [
    "Statement of Marks",
    "Name: Aisha Rahman",
    "Roll No: CBSE24-104482",
    "Percentage: 76.40%",
]
PAGE1 = ["Year of Passing: 2024", "Mobile: 9876543210"]


def test_build_text_with_offsets_matches_ocr_text():
    ocr = _ocr_from_lines([PAGE0, PAGE1])
    text, offsets = build_text_with_offsets(ocr.words)
    assert text == ocr.text
    for w, (s, e) in zip(ocr.words, offsets):
        assert text[s:e] == w.text


def test_find_pii_words_returns_name_and_roll_boxes():
    ocr = _ocr_from_lines([PAGE0, PAGE1])
    hits = find_pii_words(ocr)
    got = {(w.page, w.text) for w in hits}
    assert got == {(0, "Aisha"), (0, "Rahman"), (0, "CBSE24-104482"), (1, "9876543210")}


def test_find_pii_words_rejects_text_mismatch():
    ocr = _ocr_from_lines([PAGE0])
    bad = ocr.model_copy(update={"text": ocr.text + " extra"})
    with pytest.raises(PipelineError) as ei:
        find_pii_words(bad)
    assert ei.value.error_class is ErrorClass.PII_FAILED


# ---------- image redaction ----------


def _word(left, top, width, height, text="x", page=0):
    return OcrWord(text=text, conf=90.0, left=left, top=top, width=width, height=height, page=page)


def test_redact_boxes_paints_exact_padded_region():
    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    out = redact_boxes(img, [_word(50, 40, 30, 10)], pad=6)
    assert out is not img
    assert (img == 255).all()  # input untouched
    expected = np.zeros((100, 200), dtype=bool)
    expected[34:56, 44:86] = True  # top-6 .. top+h+6, left-6 .. left+w+6
    black = (out == 0).all(axis=2)
    white = (out == 255).all(axis=2)
    assert (black == expected).all()
    assert (white == ~expected).all()


def test_redact_boxes_clips_at_edges_grayscale():
    img = np.full((50, 60), 200, dtype=np.uint8)
    out = redact_boxes(img, [_word(2, 3, 10, 5), _word(55, 45, 20, 20)], pad=6)
    expected = np.zeros((50, 60), dtype=bool)
    expected[0:14, 0:18] = True
    expected[39:50, 49:60] = True
    assert ((out == 0) == expected).all()
    assert (out[~expected] == 200).all()


def test_redact_for_vision_blacks_out_pii_per_page():
    ocr = _ocr_from_lines([PAGE0, PAGE1])
    imgs = [np.full((400, 600, 3), 255, dtype=np.uint8) for _ in range(2)]
    out = redact_for_vision(imgs, ocr)
    assert len(out) == 2
    by_text = {(w.page, w.text): w for w in ocr.words}
    for key in [(0, "Aisha"), (0, "Rahman"), (0, "CBSE24-104482"), (1, "9876543210")]:
        w = by_text[key]
        assert (out[w.page][w.top : w.top + w.height, w.left : w.left + w.width] == 0).all(), key
    for key in [(0, "Percentage:"), (0, "Statement"), (1, "2024")]:
        w = by_text[key]
        assert (out[w.page][w.top : w.top + w.height, w.left : w.left + w.width] == 255).all(), key


def test_redact_for_vision_without_boxes_is_ocr_empty():
    ocr = OcrResult(words=[], text="", pages=[OcrPage(page=0, width=10, height=10)], mean_conf=0.0, char_count=0)
    with pytest.raises(PipelineError) as ei:
        redact_for_vision([np.zeros((10, 10), dtype=np.uint8)], ocr)
    assert ei.value.error_class is ErrorClass.OCR_EMPTY


# ---------- unlabeled names found in letters and ticket tables ----------


def test_letter_addressee_and_salutation_masked():
    text = "Date: 12 October 2025\nTo,\nIndrajit Merchant\n75/853, Some Road\n\nDear Indrajit Merchant,\nThank you."
    r = mask_text(text)
    assert "Indrajit" not in r.masked_text and "Merchant" not in r.masked_text
    assert r.masked_text.count("<PERSON_1>") == 2


def test_passenger_table_first_cell_masked_but_header_kept():
    text = "Passenger Age Gender Coach/Seat Status\nAdvay Dada 60 Male D4 / 57 CNF\nTotal Fare Rs 845"
    r = mask_text(text)
    assert "Advay" not in r.masked_text
    assert "Passenger Age Gender" in r.masked_text and "Total Fare" in r.masked_text


def test_grade_codes_and_grand_total_never_masked():
    # spaCy tags "C1\nGrand Total" as a facility; neither fragment is PII.
    text = "083 COMPUTER SCIENCE 100 055 C1\nGrand Total: 382 / 500\nPercentage: 76.40%"
    assert mask_text(text).masked_text == text


def test_cisce_unique_id_masked_whole():
    r = mask_text("Unique ID: CISCE24/4137/989\nYear of Passing: 2024")
    assert "989" not in r.masked_text and "4137" not in r.masked_text
    assert r.token_map["<ROLL_NUMBER_1>"] == "CISCE24/4137/989"
