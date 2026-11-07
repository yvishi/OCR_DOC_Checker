"""Synthetic admission-document generator for OCR_DOC_Checker.

Every document produced here is SYNTHETIC: names, numbers and institutions come from Faker
(en_IN) or random digits, and every page carries a small "SPECIMEN / SYNTHETIC" marking.
LearnSphere College and Sahyadri Valley Public School are fictional. Board headers (CBSE,
MSBSHSE, CISCE) are used only as layout flavour on clearly marked specimens.

Run from the repo root:

    .venv/bin/python -m data.generate                  # regenerate dev + test sets
    .venv/bin/python -m data.generate --set test --only 1,7

Outputs:
    <out>/dev/dev_NNN_<type>.<ext>      tuning set (20 docs)
    <out>/test/test_NNN_<type>.<ext>    held-back set (exactly 20 docs)
    <csv-dir>/dev_set.csv, <csv-dir>/test_set.csv

Everything is deterministic for a given --seed: each document gets its own RNG and Faker
seed derived from (seed, set, index), so rendering a subset with --only produces the same
bytes (PNG/JPEG) as a full run, and the CSVs are always written for the whole set.

Expected-status policy (docs/ARCHITECTURE.md section 4):
  * mark_sheet with a readable percentage -> true rule result: >= 60 Eligible, < 60 Not eligible.
    Exactly 60 / 60.0 is Eligible; 59.9 is Not eligible.
  * Near-cutoff (58-62) sheets that are CLEAN, high-confidence scans with subject marks that
    add up get their true rule result (Eligible / Not eligible), not Needs review. The variant
    column says `near_cutoff` so the eval can slice them. We never degrade a near-cutoff sheet.
  * stamp_over_total (stamp corrupts the total/percentage pixels), cgpa_only (no percentage
    printed; CGPA is never converted) and subject_mismatch (subject sum disagrees with the
    printed percentage by more than 0.5) -> Needs review.
  * rotated_90 / rotated_180 / noisy sheets keep their true rule result: the value is still
    printed clearly, the pipeline is expected to fix rotation and cope with mild noise.
  * student_id_card / fee_receipt -> Not applicable.  other -> Rejected.

CSV `percentage` is the printed percentage (empty for CGPA-only and non-mark-sheets).
`pii_values` is a `|`-separated list of every PII string exactly as printed on the document.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "samples"
DEFAULT_CSV_DIR = ROOT / "data"
DEFAULT_SEED = 20240611
SETS = ("dev", "test")

CSV_COLUMNS = [
    "filename",
    "true_type",
    "student_name",
    "roll_number",
    "board_or_school",
    "passing_year",
    "percentage",
    "expected_status",
    "variant",
    "pii_values",
]

SPECIMEN = "SPECIMEN / SYNTHETIC - not a real document"

# A4 at 200 dpi; ID cards at 300 dpi (CR80 is 3.37 x 2.13 in, we render slightly larger).
A4 = (1654, 2339)
A4_DPI = 200
CARD = (1300, 820)
CARD_DPI = 385

INK = (20, 20, 30)
GREY = (120, 120, 130)
LIGHT = (200, 200, 210)

# ---------------------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------------------

_FONT_CANDIDATES: dict[str, list[str]] = {
    "regular": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "bold": [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "serif": [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ],
    "mono": [
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ],
    "mono_bold": [
        "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
        "C:/Windows/Fonts/courbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    ],
}
_FONT_FALLBACK = "/System/Library/Fonts/Helvetica.ttc"


@lru_cache(maxsize=None)
def font(style: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A TrueType font for `style`, falling back to Helvetica.ttc, then PIL's default."""
    for path in _FONT_CANDIDATES.get(style, []) + _FONT_CANDIDATES["regular"] + [_FONT_FALLBACK]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


# ---------------------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------------------


class Canvas:
    def __init__(self, size: tuple[int, int], bg: tuple[int, int, int] = (255, 255, 255)) -> None:
        self.img = Image.new("RGB", size, bg)
        self.d = ImageDraw.Draw(self.img)
        self.w, self.h = size

    def text(self, xy, s, size=34, style="regular", fill=INK, anchor="la"):
        f = font(style, size)
        self.d.text(xy, s, font=f, fill=fill, anchor=anchor)
        return self.d.textbbox(xy, s, font=f, anchor=anchor)

    def center(self, y, s, size=34, style="regular", fill=INK, x0=0, x1=None):
        x1 = self.w if x1 is None else x1
        return self.text(((x0 + x1) // 2, y), s, size, style, fill, anchor="ma")

    def width(self, s, size=34, style="regular") -> int:
        b = self.d.textbbox((0, 0), s, font=font(style, size))
        return b[2] - b[0]

    def wrap(self, s: str, max_w: int, size=34, style="regular") -> list[str]:
        lines, cur = [], ""
        for word in s.split():
            trial = f"{cur} {word}".strip()
            if cur and self.width(trial, size, style) > max_w:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines

    def label_value(self, x, y, label, value, size=34, label_w=380, value_style="bold"):
        self.text((x, y), label, size, "regular")
        return self.text((x + label_w, y), value, size, value_style)

    def line(self, xy, fill=INK, width=2):
        self.d.line(xy, fill=fill, width=width)

    def rect(self, xy, outline=INK, width=2, fill=None):
        self.d.rectangle(xy, outline=outline, width=width, fill=fill)

    def specimen(self, size=20):
        self.text((self.w - 40, self.h - 30), SPECIMEN, size, "regular", GREY, anchor="rb")


def inr(amount: float) -> str:
    """Indian digit grouping: 148500 -> 1,48,500.00"""
    whole, frac = f"{amount:.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"{whole}.{frac}"


# ---------------------------------------------------------------------------------------
# Synthetic identity helpers
# ---------------------------------------------------------------------------------------


@dataclass
class Person:
    first: str
    last: str
    gender: str
    father: str
    mother: str
    dob: date

    @property
    def name(self) -> str:
        return f"{self.first} {self.last}"


def make_person(fk: Faker, rng: random.Random, birth_year: int) -> Person:
    gender = rng.choice(["M", "F"])
    first = fk.first_name_male() if gender == "M" else fk.first_name_female()
    last = fk.last_name()
    father = f"{fk.first_name_male()} {last}"
    mother = f"{fk.first_name_female()} {last}"
    dob = date(birth_year, 1, 1) + timedelta(days=rng.randrange(365))
    return Person(first, last, gender, father, mother, dob)


def fmt_dob(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def phone(rng: random.Random) -> str:
    first = rng.choice("6789")
    digits = first + "".join(rng.choice("0123456789") for _ in range(9))
    return f"+91 {digits[:5]} {digits[5:]}"


def aadhaar(rng: random.Random) -> str:
    d = str(rng.randint(2, 9)) + "".join(rng.choice("0123456789") for _ in range(11))
    return f"{d[:4]} {d[4:8]} {d[8:]}"


def pan(rng: random.Random, last_name: str) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    initial = (last_name[:1] or "K").upper()
    if initial not in letters:
        initial = "K"
    head = "".join(rng.choice(letters) for _ in range(3)) + "P" + initial
    return head + "".join(rng.choice("0123456789") for _ in range(4)) + rng.choice(letters)


def email(rng: random.Random, p_first: str, p_last: str) -> str:
    dom = rng.choice(["example.com", "example.in", "mail.example.org"])
    sep = rng.choice([".", "_", ""])
    return f"{p_first.lower()}{sep}{p_last.lower()}{rng.randint(1, 99)}@{dom}".replace(" ", "")


def address(fk: Faker) -> str:
    return fk.address().replace("\n", ", ")


def digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


# ---------------------------------------------------------------------------------------
# Mark sheet planning
# ---------------------------------------------------------------------------------------

BOARDS: dict[str, dict[str, Any]] = {
    "CBSE": {
        "short": "CBSE",
        "header": ["CENTRAL BOARD OF SECONDARY EDUCATION"],
        "exam": "SENIOR SCHOOL CERTIFICATE EXAMINATION (CLASS XII) {year}",
        "title": "MARKS STATEMENT CUM CERTIFICATE",
        "roll_label": "Roll No.",
        "color": (20, 60, 130),
        "subjects": [
            ("301", "ENGLISH CORE"), ("041", "MATHEMATICS"), ("042", "PHYSICS"),
            ("043", "CHEMISTRY"), ("083", "COMPUTER SCIENCE"), ("044", "BIOLOGY"),
            ("048", "PHYSICAL EDUCATION"),
        ],
        "place": "New Delhi",
    },
    "MSBSHSE": {
        "short": "MSBSHSE",
        "header": ["MAHARASHTRA STATE BOARD OF SECONDARY AND", "HIGHER SECONDARY EDUCATION, PUNE (MSBSHSE)"],
        "exam": "HIGHER SECONDARY CERTIFICATE EXAMINATION - MARCH {year}",
        "title": "STATEMENT OF MARKS",
        "roll_label": "Seat No.",
        "color": (120, 30, 40),
        "subjects": [
            ("01", "ENGLISH"), ("02", "MARATHI"), ("40", "MATHEMATICS & STATISTICS"),
            ("54", "PHYSICS"), ("55", "CHEMISTRY"), ("56", "BIOLOGY"), ("98", "INFORMATION TECHNOLOGY"),
        ],
        "place": "Pune",
    },
    "CISCE": {
        "short": "CISCE",
        "header": ["COUNCIL FOR THE INDIAN SCHOOL", "CERTIFICATE EXAMINATIONS (CISCE)"],
        "exam": "INDIAN SCHOOL CERTIFICATE (ISC) EXAMINATION {year}",
        "title": "STATEMENT OF MARKS",
        "roll_label": "Unique ID",
        "color": (30, 90, 60),
        "subjects": [
            ("ENG", "ENGLISH"), ("MAT", "MATHEMATICS"), ("PHY", "PHYSICS"), ("CHE", "CHEMISTRY"),
            ("COM", "COMMERCE"), ("ECO", "ECONOMICS"), ("CSC", "COMPUTER SCIENCE"),
        ],
        "place": "New Delhi",
    },
    "SVPS": {
        "short": "Sahyadri Valley Public School",
        "header": ["SAHYADRI VALLEY PUBLIC SCHOOL, NASHIK"],
        "exam": "ANNUAL EXAMINATION (CLASS XII) {year}",
        "title": "REPORT CARD / MARK SHEET",
        "roll_label": "Roll No.",
        "color": (90, 50, 120),
        "subjects": [
            ("S1", "ENGLISH"), ("S2", "MATHEMATICS"), ("S3", "PHYSICS"), ("S4", "CHEMISTRY"),
            ("S5", "BIOLOGY"), ("S6", "HISTORY"),
        ],
        "place": "Nashik",
    },
}

SCHOOL_SUFFIXES = ["Junior College", "Higher Secondary School", "Public School", "Vidyalaya", "English Medium School"]


def roll_number(board: str, rng: random.Random, year: int) -> str:
    yy = str(year)[-2:]
    if board == "CBSE":
        return f"CBSE{yy}-{digits(rng, 6)}"
    if board == "MSBSHSE":
        return f"M{digits(rng, 6)}"
    if board == "CISCE":
        return f"CISCE{yy}/{digits(rng, 4)}/{digits(rng, 3)}"
    return f"SVPS/{yy}/{digits(rng, 4)}"


# Subject-maximum configurations tried (in order) when solving for an exact printed percentage.
_MAX_CONFIGS = [[100] * 5, [100] * 6, [200] * 5]


def solve_total(pct: float, preferred: list[int] | None = None) -> tuple[list[int], int]:
    """Find subject maxima and an integer total with round(total/max*100, 2) == pct."""
    configs = ([preferred] if preferred else []) + _MAX_CONFIGS
    for maxes in configs:
        m = sum(maxes)
        t = round(pct * m / 100)
        if round(t / m * 100, 2) == round(pct, 2):
            return maxes, t
    raise ValueError(f"no integer total gives exactly {pct}%")


def split_marks(total: int, maxes: list[int], rng: random.Random) -> list[int]:
    """Split `total` into per-subject marks within [35% of max, max], with some variety."""
    m = sum(maxes)
    marks = [int(total * mx / m) for mx in maxes]
    i = 0
    while sum(marks) < total:
        j = i % len(marks)
        if marks[j] < maxes[j]:
            marks[j] += 1
        i += 1
    lo = [math.ceil(0.35 * mx) for mx in maxes]
    for _ in range(40):
        a, b = rng.sample(range(len(marks)), 2)
        k = rng.randint(1, 7)
        if marks[a] - k >= lo[a] and marks[b] + k <= maxes[b]:
            marks[a] -= k
            marks[b] += k
    assert sum(marks) == total
    return marks


def grade_for(mark: int, mx: int) -> str:
    f = mark / mx
    for cut, g in [(0.91, "A1"), (0.81, "A2"), (0.71, "B1"), (0.61, "B2"), (0.51, "C1"), (0.41, "C2"), (0.33, "D")]:
        if f >= cut:
            return g
    return "E"


GRADE_POINTS = {10: "A1", 9: "A2", 8: "B1", 7: "B2", 6: "C1", 5: "C2", 4: "D"}


def cgpa_points(cgpa: float, n: int, rng: random.Random) -> list[int]:
    total = round(cgpa * n)
    pts = [total // n] * n
    for i in range(total - sum(pts)):
        pts[i] += 1
    for _ in range(10):
        a, b = rng.sample(range(n), 2)
        if pts[a] > 5 and pts[b] < 10:
            pts[a] -= 1
            pts[b] += 1
    assert sum(pts) == total
    return pts


# ---------------------------------------------------------------------------------------
# Document plans
# ---------------------------------------------------------------------------------------


@dataclass
class DocPlan:
    set_name: str
    index: int
    doc_type: str
    variant: str
    fmt: str  # png | jpg | pdf
    data: dict[str, Any] = field(default_factory=dict)
    row: dict[str, str] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return f"{self.set_name}_{self.index:03d}_{self.doc_type}.{self.fmt}"


# (doc_type, variant, fmt, params). Order is the document index (1-based).
SPECS: dict[str, list[tuple[str, str, str, dict[str, Any]]]] = {
    "test": [
        ("mark_sheet", "clean", "png", {"board": "CBSE", "pct": 76.4}),
        ("mark_sheet", "noisy", "jpg", {"board": "MSBSHSE", "pct": 48.2}),
        ("mark_sheet", "boundary_60", "png", {"board": "CISCE", "pct": 60.0, "pct_fmt": "{:.1f}%"}),
        ("mark_sheet", "boundary_59_9", "pdf", {"board": "SVPS", "pct": 59.9, "pages": 2}),
        ("mark_sheet", "near_cutoff", "png", {"board": "MSBSHSE", "pct": 58.6}),
        ("mark_sheet", "near_cutoff", "pdf", {"board": "CBSE", "pct": 61.4}),
        ("mark_sheet", "stamp_over_total", "png", {"board": "CBSE", "pct": 72.6, "stamp": "red"}),
        ("mark_sheet", "cgpa_only", "jpg", {"board": "CBSE", "cgpa": 8.2}),
        ("mark_sheet", "rotated_90", "png", {"board": "MSBSHSE", "pct": 81.2, "rotate": 90}),
        ("mark_sheet", "rotated_180", "jpg", {"board": "CISCE", "pct": 52.8, "rotate": 180}),
        ("mark_sheet", "subject_mismatch", "pdf", {"board": "SVPS", "pct": 66.0, "subject_pct": 63.2}),
        ("student_id_card", "clean", "png", {"issuer": "college"}),
        ("student_id_card", "clean", "jpg", {"issuer": "school"}),
        ("student_id_card", "clean", "pdf", {"issuer": "college"}),
        ("fee_receipt", "clean", "pdf", {"mode": "UPI"}),
        ("fee_receipt", "clean", "png", {"mode": "NEFT"}),
        ("fee_receipt", "clean", "jpg", {"mode": "Cash"}),
        ("other", "letter", "png", {"kind": "letter"}),
        ("other", "train_ticket", "pdf", {"kind": "train_ticket"}),
        ("other", "grocery_bill", "jpg", {"kind": "grocery_bill"}),
    ],
    "dev": [
        ("mark_sheet", "clean", "png", {"board": "MSBSHSE", "pct": 84.33, "maxes": [100] * 6}),
        ("mark_sheet", "clean", "jpg", {"board": "CBSE", "pct": 41.6}),
        ("mark_sheet", "boundary_60", "pdf", {"board": "MSBSHSE", "pct": 60.0, "pages": 2}),
        ("mark_sheet", "boundary_59_9", "png", {"board": "CISCE", "pct": 59.9}),
        ("mark_sheet", "near_cutoff", "jpg", {"board": "CBSE", "pct": 58.4}),
        ("mark_sheet", "near_cutoff", "png", {"board": "SVPS", "pct": 61.8}),
        ("mark_sheet", "stamp_over_total", "png", {"board": "MSBSHSE", "pct": 64.2, "stamp": "blue"}),
        ("mark_sheet", "cgpa_only", "pdf", {"board": "CBSE", "cgpa": 7.4}),
        ("mark_sheet", "rotated_90", "jpg", {"board": "CBSE", "pct": 55.0, "rotate": 90}),
        ("mark_sheet", "rotated_180", "png", {"board": "SVPS", "pct": 70.6, "rotate": 180}),
        ("mark_sheet", "subject_mismatch", "png", {"board": "CBSE", "pct": 71.0, "subject_pct": 68.4}),
        ("mark_sheet", "noisy", "pdf", {"board": "CISCE", "pct": 88.8}),
        ("student_id_card", "clean", "png", {"issuer": "college"}),
        ("student_id_card", "noisy", "jpg", {"issuer": "school"}),
        ("student_id_card", "clean", "png", {"issuer": "college"}),
        ("fee_receipt", "clean", "png", {"mode": "UPI"}),
        ("fee_receipt", "clean", "pdf", {"mode": "NEFT"}),
        ("fee_receipt", "clean", "jpg", {"mode": "Cash"}),
        ("other", "letter_mentions_percentage", "png", {"kind": "recommendation"}),
        ("other", "grocery_bill", "pdf", {"kind": "grocery_bill"}),
    ],
}


def _seed_int(seed: int, set_name: str, index: int) -> int:
    return seed * 1000 + (0 if set_name == "dev" else 500) + index


def _row(plan: DocPlan, **kw: Any) -> dict[str, str]:
    row = {c: "" for c in CSV_COLUMNS}
    row.update({"filename": plan.filename, "true_type": plan.doc_type, "variant": plan.variant})
    for k, v in kw.items():
        if k == "pii_values":
            row[k] = "|".join(dict.fromkeys(str(x) for x in v if x))
        else:
            row[k] = "" if v is None else str(v)
    return row


def _plan_mark_sheet(plan: DocPlan, fk: Faker, rng: random.Random, p: dict[str, Any]) -> None:
    board_key = p["board"]
    board = BOARDS[board_key]
    year = p.get("year", rng.choice([2023, 2024, 2025]))
    person = make_person(fk, rng, year - 18)
    roll = roll_number(board_key, rng, year)
    school = f"{fk.last_name()} {rng.choice(SCHOOL_SUFFIXES)}, {fk.city()}"
    show_aadhaar = rng.random() < 0.6
    aad = aadhaar(rng) if show_aadhaar else ""
    d: dict[str, Any] = {
        "board": board_key, "year": year, "person": person, "roll": roll, "school": school,
        "aadhaar": aad, "issue_date": date(year, 5, rng.randint(10, 28)),
        "pages": p.get("pages", 1), "stamp": p.get("stamp"), "rotate": p.get("rotate", 0),
        "noisy": plan.variant == "noisy", "cgpa": p.get("cgpa"),
    }
    if p.get("cgpa") is not None:
        n = 5
        subj = board["subjects"][:n]
        pts = cgpa_points(p["cgpa"], n, rng)
        d["subjects"] = [(c, s, None, None, GRADE_POINTS[g], g) for (c, s), g in zip(subj, pts)]
        pct = None
        status = "Needs review"
    else:
        pct = float(p["pct"])
        maxes, total = solve_total(pct, p.get("maxes"))
        subject_total = total
        if "subject_pct" in p:
            m2, subject_total = solve_total(p["subject_pct"], maxes)
            assert m2 == maxes
        marks = split_marks(subject_total, maxes, rng)
        subj = board["subjects"][: len(maxes)]
        d["subjects"] = [(c, s, mx, mk, grade_for(mk, mx), None) for (c, s), mx, mk in zip(subj, maxes, marks)]
        d["total"], d["max_total"] = total, sum(maxes)
        d["pct_text"] = p.get("pct_fmt", "{:.2f}%").format(pct)
        if plan.variant in ("stamp_over_total", "subject_mismatch"):
            status = "Needs review"
        else:
            status = "Eligible" if pct >= 60.0 else "Not eligible"
    plan.data = d
    plan.row = _row(
        plan,
        student_name=person.name, roll_number=roll, board_or_school=board["short"],
        passing_year=year, percentage=pct, expected_status=status,
        # names are printed in capitals on mark sheets; list both spellings
        pii_values=[person.name, person.name.upper(), person.mother, person.mother.upper(), person.father,
                    person.father.upper(), roll, fmt_dob(person.dob), aad],
    )


COURSES = ["B.Sc. Computer Science", "B.Com.", "B.A. Economics", "BBA", "B.Sc. Physics", "BCA"]
BLOOD = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
COLLEGE = "LearnSphere College"
COLLEGE_ADDR = "Survey No. 42, Model Colony, Shivajinagar, Pune 411016 (fictional)"


def _college_roll(rng: random.Random, year: int, course: str) -> str:
    dept = {"B.Sc. Computer Science": "BSC", "B.Com.": "BCOM", "B.A. Economics": "BA", "BBA": "BBA",
            "B.Sc. Physics": "BSC", "BCA": "BCA"}[course]
    return f"LSC/{year}/{dept}/{digits(rng, 4)}"


def _plan_id_card(plan: DocPlan, fk: Faker, rng: random.Random, p: dict[str, Any]) -> None:
    year = rng.choice([2024, 2025])
    college = p["issuer"] == "college"
    person = make_person(fk, rng, year - (18 if college else 17))
    course = rng.choice(COURSES)
    if college:
        roll = _college_roll(rng, year, course)
        issuer = COLLEGE
    else:
        roll = f"SVPS/{str(year)[-2:]}/{digits(rng, 4)}"
        issuer = BOARDS["SVPS"]["short"]
        course = "Class XII - Science"
    ph = phone(rng)
    addr = address(fk)
    mail = email(rng, person.first, person.last) if college else ""
    plan.data = {
        "college": college, "person": person, "roll": roll, "course": course, "phone": ph,
        "address": addr, "blood": rng.choice(BLOOD), "valid": year + (3 if college else 1),
        "email": mail, "noisy": plan.variant == "noisy",
    }
    plan.row = _row(
        plan, student_name=person.name, roll_number=roll, board_or_school=issuer,
        expected_status="Not applicable",
        pii_values=[person.name, person.father if not college else "", roll, fmt_dob(person.dob), ph, addr, mail],
    )


FEE_HEADS = [("Tuition Fee", 28000, 52000), ("Development Fee", 3000, 8000), ("Library Fee", 800, 2000),
             ("Examination Fee", 1500, 3500), ("Gymkhana / Student Activities", 500, 1500)]
BANKS = [("HDFC", "okhdfcbank"), ("SBIN", "oksbi"), ("ICIC", "okicici"), ("UTIB", "okaxis")]


def _plan_fee_receipt(plan: DocPlan, fk: Faker, rng: random.Random, p: dict[str, Any]) -> None:
    year = rng.choice([2024, 2025])
    person = make_person(fk, rng, year - 18)
    course = rng.choice(COURSES)
    roll = _college_roll(rng, year, course)
    heads = [(h, rng.randrange(lo, hi, 50)) for h, lo, hi in FEE_HEADS]
    ph = phone(rng)
    mail = email(rng, person.father.split()[0], person.last)
    bank, vpa_suffix = rng.choice(BANKS)
    mode = p["mode"]
    payer_pan = pan(rng, person.last) if rng.random() < 0.7 or mode == "NEFT" else ""
    pay: dict[str, str] = {}
    pii_extra: list[str] = []
    if mode == "UPI":
        vpa = f"{person.father.split()[0].lower()}.{person.last.lower()}@{vpa_suffix}"
        pay = {"UPI ID": vpa, "UPI Ref No.": digits(rng, 12)}
        pii_extra = [vpa]
    elif mode == "NEFT":
        acct = digits(rng, 14)
        pay = {"Payer A/c No.": acct, "IFSC": f"{bank}000{digits(rng, 4)}", "UTR No.": f"{bank}N{year}{digits(rng, 10)}"}
        pii_extra = [acct]
    else:
        pay = {"Received by": "Cashier, Accounts Section"}
    paid_on = date(year, rng.randint(6, 8), rng.randint(1, 28))
    plan.data = {
        "person": person, "roll": roll, "course": course, "heads": heads, "phone": ph, "email": mail,
        "mode": mode, "pay": pay, "pan": payer_pan, "date": paid_on,
        "receipt_no": f"LSC/FR/{str(year)[-2:]}-{str(year + 1)[-2:]}/{digits(rng, 6)}",
        "academic_year": f"{year}-{str(year + 1)[-2:]}",
    }
    plan.row = _row(
        plan, student_name=person.name, roll_number=roll, board_or_school=COLLEGE,
        expected_status="Not applicable",
        pii_values=[person.name, person.father, roll, ph, mail, payer_pan] + pii_extra,
    )


def _plan_other(plan: DocPlan, fk: Faker, rng: random.Random, p: dict[str, Any]) -> None:
    kind = p["kind"]
    person = make_person(fk, rng, rng.randint(1965, 1995))
    ph = phone(rng)
    addr = address(fk)
    d: dict[str, Any] = {"kind": kind, "person": person, "phone": ph, "address": addr,
                         "date": date(2025, rng.randint(1, 12), rng.randint(1, 28))}
    pii = [person.name, ph]
    if kind == "letter":
        d["email"] = email(rng, person.first, person.last)
        d["pan"] = pan(rng, person.last)
        d["secretary"] = f"{fk.first_name()} {fk.last_name()}"
        d["dues"] = rng.randrange(4500, 18000, 250)
        pii += [addr, d["email"], d["pan"], d["secretary"]]
    elif kind == "recommendation":
        student = make_person(fk, rng, 2006)
        d["student"] = student
        d["email"] = email(rng, person.first, person.last)
        d["score"] = rng.choice([78, 82, 86, 91])
        pii += [student.name, d["email"]]
    elif kind == "train_ticket":
        d["pnr"] = digits(rng, 10)
        d["age"] = 2025 - person.dob.year
        d["aadhaar"] = aadhaar(rng)
        pii += [d["pnr"], d["aadhaar"]]
    elif kind == "grocery_bill":
        items = [("Basmati Rice 5kg", 1, 545), ("Toor Dal 1kg", 2, 168), ("Sunflower Oil 1L", 2, 152),
                 ("Atta 10kg", 1, 465), ("Sugar 1kg", 2, 48), ("Tea 500g", 1, 280), ("Milk 1L", 4, 66),
                 ("Paneer 200g", 2, 95), ("Onion 1kg", 3, 38), ("Biscuits", 5, 30)]
        d["items"] = rng.sample(items, 7)
        d["bill_no"] = f"FB{digits(rng, 6)}"
        pii += []
    plan.data = d
    plan.row = _row(plan, expected_status="Rejected", pii_values=pii)


_PLANNERS = {
    "mark_sheet": _plan_mark_sheet,
    "student_id_card": _plan_id_card,
    "fee_receipt": _plan_fee_receipt,
    "other": _plan_other,
}


def plan_set(set_name: str, seed: int = DEFAULT_SEED) -> list[DocPlan]:
    """Plan every document of a set (pure data, no rendering). Deterministic in `seed`."""
    fk = Faker("en_IN")
    plans = []
    for i, (doc_type, variant, fmt, params) in enumerate(SPECS[set_name], start=1):
        s = _seed_int(seed, set_name, i)
        rng = random.Random(s)
        fk.seed_instance(s)
        plan = DocPlan(set_name, i, doc_type, variant, fmt)
        _PLANNERS[doc_type](plan, fk, rng, params)
        plans.append(plan)
    return plans


# ---------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------


def _emblem(c: Canvas, cx: int, cy: int, r: int, color, initials: str) -> None:
    c.d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=5)
    c.d.ellipse((cx - r + 12, cy - r + 12, cx + r - 12, cy + r - 12), outline=color, width=2)
    c.text((cx, cy), initials, int(r * 0.55), "serif", color, anchor="mm")


def render_mark_sheet(plan: DocPlan) -> list[Image.Image]:
    d = plan.data
    board = BOARDS[d["board"]]
    color = board["color"]
    p: Person = d["person"]
    c = Canvas(A4)
    W, H = A4
    c.rect((60, 60, W - 60, H - 60), outline=color, width=6)
    c.rect((76, 76, W - 76, H - 76), outline=color, width=2)
    initials = "".join(w[0] for w in board["short"].split()) if d["board"] == "SVPS" else board["short"][:4]
    _emblem(c, 230, 250, 105, color, initials)

    y = 150
    for line in board["header"]:
        c.center(y, line, 46, "serif", color, x0=330, x1=W - 100)
        y += 60
    y += 10
    c.center(y, board["exam"].format(year=d["year"]), 32, "bold", INK, x0=330, x1=W - 100)
    y += 56
    c.center(y, board["title"], 38, "bold", color, x0=330, x1=W - 100)
    y = max(y + 90, 440)
    c.line((110, y, W - 110, y), color, 3)
    y += 40

    fields = [
        ("Name of Candidate", p.name.upper()),
        ("Mother's Name", p.mother.upper()),
        ("Father's Name", p.father.upper()),
        (board["roll_label"], d["roll"]),
        ("Date of Birth", fmt_dob(p.dob)),
        ("School", d["school"]),
        ("Year of Passing", str(d["year"])),
    ]
    if d["aadhaar"]:
        fields.append(("Aadhaar No.", d["aadhaar"]))
    for label, value in fields:
        c.label_value(140, y, f"{label}:", value, 34, label_w=400)
        y += 58
    y += 30

    # Subject table
    cgpa = d["cgpa"] is not None
    if cgpa:
        cols = [("Code", 140), ("Subject", 300), ("Grade", 1050), ("Grade Point", 1260)]
    else:
        cols = [("Code", 140), ("Subject", 300), ("Max Marks", 900), ("Marks Obtained", 1110), ("Grade", 1400)]
    top = y
    c.rect((110, top, W - 110, top + 70), outline=INK, width=2, fill=(235, 238, 245))
    for name, x in cols:
        c.text((x, top + 18), name, 32, "bold")
    y = top + 70
    row_h = 66
    for code, subj, mx, mk, grade, gp in d["subjects"]:
        if cgpa:
            vals = [code, subj, grade, str(gp)]
        else:
            vals = [code, subj, str(mx), f"{mk:03d}" if d["board"] == "CBSE" else str(mk), grade]
        for (name, x), v in zip(cols, vals):
            c.text((x, y + 16), v, 32, "regular")
        y += row_h
        c.line((110, y, W - 110, y), LIGHT, 2)
    c.rect((110, top, W - 110, y), outline=INK, width=2)
    for _, x in cols[1:]:
        c.line((x - 25, top, x - 25, y), INK, 2)
    y += 60

    value_boxes = []
    if cgpa:
        c.label_value(140, y, "CGPA:", f"{d['cgpa']:.1f}", 40, label_w=400)
        y += 70
        c.text((140, y), "Result: PASS", 36, "bold")
        y += 60
    else:
        value_boxes.append(c.label_value(140, y, "Grand Total:", f"{d['total']} / {d['max_total']}", 40, label_w=400))
        y += 70
        value_boxes.append(c.label_value(140, y, "Percentage:", d["pct_text"], 40, label_w=400))
        y += 70
        c.text((140, y), "Result: PASS", 36, "bold")
        y += 60

    # Footer
    fy = H - 360
    c.text((140, fy), f"Date of Issue: {d['issue_date'].strftime('%d/%m/%Y')}", 30)
    c.text((140, fy + 50), f"Place: {board['place']}", 30)
    c.line((W - 560, fy + 60, W - 140, fy + 60), INK, 2)
    c.text((W - 350, fy + 75), "Controller of Examinations", 28, anchor="ma")
    c.text((140, H - 180), "This statement is computer generated. Marks are out of the maximum shown.", 24, fill=GREY)
    c.specimen()

    img = c.img
    if d["stamp"] and value_boxes:
        img = draw_stamp(img, value_boxes, d["stamp"], random.Random(plan.filename))
    pages = [img]
    if d["pages"] > 1:
        pages.append(_mark_sheet_back(board))
    return pages


def _mark_sheet_back(board: dict[str, Any]) -> Image.Image:
    c = Canvas(A4)
    W, _ = A4
    c.center(160, "NOTES AND GRADING SCHEME", 42, "bold", board["color"])
    y = 280
    notes = [
        "1. Minimum marks required to pass in each subject is 33 percent of the maximum.",
        "2. Grades: A1 91-100, A2 81-90, B1 71-80, B2 61-70, C1 51-60, C2 41-50, D 33-40, E below 33.",
        "3. This statement is issued subject to verification of records.",
        "4. Any alteration renders this statement invalid.",
        "5. Verify authenticity only through the issuing office.",
    ]
    for n in notes:
        for line in c.wrap(n, W - 300, 32):
            c.text((150, y), line, 32)
            y += 50
        y += 20
    c.specimen()
    return c.img


def draw_stamp(img: Image.Image, boxes: list[tuple[int, int, int, int]], color_name: str,
               rng: random.Random) -> Image.Image:
    """Semi-transparent circular office stamp over the total/percentage values."""
    rgb = (200, 30, 40) if color_name == "red" else (30, 50, 190)
    x0 = min(b[0] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y0 = min(b[1] for b in boxes)
    y1 = max(b[3] for b in boxes)
    cx, cy = (x0 + x1) // 2 + 20, (y0 + y1) // 2
    r = max(170, (x1 - x0) // 2 + 30)
    size = 2 * r + 40
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    o = size // 2
    a = 215
    ld.ellipse((o - r, o - r, o + r, o + r), outline=rgb + (a,), width=16)
    ld.ellipse((o - r + 34, o - r + 34, o + r - 34, o + r - 34), outline=rgb + (a,), width=6)
    # ring text
    ring = "* EXAMINATION OFFICE * VERIFIED COPY * EXAMINATION OFFICE * VERIFIED "
    f = font("bold", 26)
    for i, ch in enumerate(ring):
        ang = 2 * math.pi * i / len(ring)
        tx = o + (r - 18 - 4) * math.sin(ang) * 0.93
        ty = o - (r - 18 - 4) * math.cos(ang) * 0.93
        glyph = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
        ImageDraw.Draw(glyph).text((20, 20), ch, font=f, fill=rgb + (a,), anchor="mm")
        glyph = glyph.rotate(-math.degrees(ang), resample=Image.BICUBIC)
        layer.alpha_composite(glyph, (int(tx) - 20, int(ty) - 20))
    # dense centre: bands of ink with text, crossing the digits
    ld.rectangle((o - r + 40, o - 62, o + r - 40, o + 62), fill=rgb + (150,))
    ld.text((o, o - 8), "VERIFIED", font=font("bold", max(40, int(r * 0.42))), fill=rgb + (235,), anchor="mm")
    ld.text((o, o + 52), "28 MAY", font=font("bold", 44), fill=rgb + (235,), anchor="mm")
    for _ in range(14):
        yy = rng.randint(o - r + 40, o + r - 40)
        ld.line((o - r + 30, yy, o + r - 30, yy + rng.randint(-30, 30)), fill=rgb + (190,), width=rng.randint(6, 12))
    for _ in range(40):  # ink speckle
        sx, sy = rng.randint(o - r, o + r), rng.randint(o - r, o + r)
        sr = rng.randint(4, 14)
        ld.ellipse((sx - sr, sy - sr, sx + sr, sy + sr), fill=rgb + (200,))
    layer = layer.rotate(rng.uniform(-18, 18), resample=Image.BICUBIC)
    base = img.convert("RGBA")
    base.alpha_composite(layer, (cx - o, cy - o))
    return base.convert("RGB")


def render_id_card(plan: DocPlan) -> list[Image.Image]:
    d = plan.data
    p: Person = d["person"]
    W, H = CARD
    c = Canvas(CARD, (250, 250, 252))
    navy = (18, 45, 100) if d["college"] else (90, 50, 120)
    c.rect((0, 0, W, 170), outline=navy, width=1, fill=navy)
    if d["college"]:
        c.center(26, "LEARNSPHERE COLLEGE, PUNE", 50, "bold", (255, 255, 255))
        c.center(96, "Affiliated (fictional)  |  " + "www.learnsphere.example", 26, "regular", (220, 225, 240))
    else:
        c.center(26, "SAHYADRI VALLEY PUBLIC SCHOOL", 48, "bold", (255, 255, 255))
        c.center(96, "Gangapur Road, Nashik (fictional)", 26, "regular", (220, 225, 240))
    c.center(190, "STUDENT IDENTITY CARD", 36, "bold", navy)
    # photo placeholder
    c.rect((50, 250, 300, 560), outline=GREY, width=3, fill=(232, 232, 238))
    c.d.ellipse((120, 300, 230, 410), outline=GREY, width=3)
    c.d.arc((90, 420, 260, 600), 200, 340, fill=GREY, width=3)
    c.text((175, 520), "PHOTO", 26, "bold", GREY, anchor="mm")
    x, y, lw = 340, 250, 270
    rows = [
        ("Name", p.name),
        ("Enrollment No." if d["college"] else "Roll No.", d["roll"]),
        ("Course" if d["college"] else "Class", d["course"]),
        ("Date of Birth", fmt_dob(p.dob)),
        ("Blood Group", d["blood"]),
        ("Phone", d["phone"]),
    ]
    if not d["college"]:
        rows.insert(2, ("Father's Name", p.father))
    for label, value in rows:
        c.label_value(x, y, f"{label}:", value, 30, label_w=lw)
        y += 46
    c.text((x, y), "Address:", 30)
    for line in c.wrap(d["address"], W - x - lw - 40, 28)[:2]:
        c.text((x + lw, y + 2), line, 28)
        y += 40
    if d["email"]:
        c.label_value(x, y + 6, "Email:", d["email"], 28, label_w=lw, value_style="regular")
    c.rect((0, H - 90, W, H), outline=navy, width=1, fill=navy)
    c.text((50, H - 68), f"Valid till: {d['valid']}", 32, "bold", (255, 255, 255))
    c.text((W - 50, H - 66), "Principal", 28, "regular", (255, 255, 255), anchor="ra")
    c.text((W - 40, H - 100), SPECIMEN, 18, "regular", GREY, anchor="rb")
    return [c.img]


def render_fee_receipt(plan: DocPlan) -> list[Image.Image]:
    d = plan.data
    p: Person = d["person"]
    W, H = A4[0], 1900
    c = Canvas((W, H))
    navy = (18, 45, 100)
    c.rect((60, 60, W - 60, H - 60), outline=navy, width=4)
    _emblem(c, 220, 200, 90, navy, "LSC")
    c.center(130, "LEARNSPHERE COLLEGE", 56, "serif", navy, x0=300)
    c.center(205, COLLEGE_ADDR, 26, "regular", INK, x0=300)
    c.center(250, "Phone: 020-2550-0000  |  accounts@learnsphere.example", 26, "regular", INK, x0=300)
    c.center(320, "FEE RECEIPT", 44, "bold", navy)
    c.line((100, 385, W - 100, 385), navy, 3)
    y = 420
    left = [
        ("Receipt No.", d["receipt_no"]),
        ("Date", d["date"].strftime("%d/%m/%Y")),
        ("Student Name", p.name),
        ("Enrollment No.", d["roll"]),
        ("Course", d["course"]),
        ("Academic Year", d["academic_year"]),
        ("Parent / Guardian", p.father),
        ("Mobile", d["phone"]),
        ("Email", d["email"]),
    ]
    if d["pan"]:
        left.append(("Parent PAN", d["pan"]))
    for label, value in left:
        c.label_value(130, y, f"{label}:", value, 32, label_w=380)
        y += 54
    y += 30
    top = y
    c.rect((110, top, W - 110, top + 64), outline=INK, width=2, fill=(235, 238, 245))
    c.text((140, top + 16), "Sr.", 30, "bold")
    c.text((240, top + 16), "Particulars", 30, "bold")
    c.text((W - 140, top + 16), "Amount (Rs.)", 30, "bold", anchor="ra")
    y = top + 64
    for i, (h, amt) in enumerate(d["heads"], 1):
        c.text((140, y + 14), str(i), 30)
        c.text((240, y + 14), h, 30)
        c.text((W - 140, y + 14), inr(amt), 30, anchor="ra")
        y += 60
        c.line((110, y, W - 110, y), LIGHT, 2)
    total = sum(a for _, a in d["heads"])
    c.rect((110, top, W - 110, y + 70), outline=INK, width=2)
    c.text((240, y + 18), "TOTAL AMOUNT PAID", 32, "bold")
    c.text((W - 140, y + 18), f"Rs. {inr(total)}", 32, "bold", anchor="ra")
    y += 120
    c.label_value(130, y, "Payment Mode:", d["mode"], 32, label_w=380)
    y += 54
    for label, value in d["pay"].items():
        c.label_value(130, y, f"{label}:", value, 32, label_w=380)
        y += 54
    c.line((W - 560, H - 230, W - 140, H - 230), INK, 2)
    c.text((W - 350, H - 215), "Authorised Signatory (Accounts)", 26, anchor="ma")
    c.text((130, H - 160), "Fees once paid are not refundable. Subject to realisation of payment.", 24, fill=GREY)
    c.specimen()
    return [c.img]


def render_other(plan: DocPlan) -> list[Image.Image]:
    d = plan.data
    p: Person = d["person"]
    kind = d["kind"]
    if kind == "letter":
        c = Canvas(A4)
        W, _ = A4
        c.center(140, "SHREE GANESH CO-OPERATIVE HOUSING SOCIETY LTD.", 40, "bold")
        c.center(200, "Regd. No. PNA/HSG/0000 (fictional), Kothrud, Pune 411038", 28)
        c.line((120, 260, W - 120, 260), INK, 2)
        y = 310
        c.text((W - 140, y), f"Date: {d['date'].strftime('%d %B %Y')}", 30, anchor="ra")
        y += 70
        for line in ["To,", p.name] + c.wrap(d["address"], 800, 30) + [f"Mobile: {d['phone']}", f"Email: {d['email']}"]:
            c.text((140, y), line, 30)
            y += 46
        y += 40
        c.text((140, y), "Subject: Reminder for pending maintenance charges", 32, "bold")
        y += 80
        body = (
            f"Dear {p.first}, this is to inform you that maintenance charges of Rs. {inr(d['dues'])} for the "
            "current quarter are pending against your flat. Kindly clear the dues before the end of the month "
            "to avoid a late payment charge. For the income tax receipt, the society has recorded your PAN as "
            f"{d['pan']}. Please contact the society office between 10 am and 1 pm on weekdays for any queries. "
            "The annual general meeting will be held on the second Sunday of next month in the clubhouse."
        )
        for line in c.wrap(body, W - 280, 32):
            c.text((140, y), line, 32)
            y += 50
        y += 80
        for line in ["Yours sincerely,", d["secretary"], "Hon. Secretary"]:
            c.text((140, y), line, 32)
            y += 50
        c.specimen()
        return [c.img]
    if kind == "recommendation":
        c = Canvas(A4)
        W, _ = A4
        s: Person = d["student"]
        c.center(140, "TO WHOMSOEVER IT MAY CONCERN", 42, "bold")
        y = 260
        c.text((W - 140, y), f"Date: {d['date'].strftime('%d/%m/%Y')}", 30, anchor="ra")
        y += 90
        body = (
            f"I am pleased to recommend {s.name}, who has been my student in Physics for the last two years. "
            f"{s.first} is diligent and curious, and scored {d['score']}% in the school's preliminary examination. "
            "Beyond academics, they led the science club and volunteered at the community library every weekend. "
            "I am confident they will do well in any undergraduate programme they choose and I recommend them "
            "without reservation."
        )
        for line in c.wrap(body, W - 280, 34):
            c.text((140, y), line, 34)
            y += 54
        y += 80
        for line in ["Sincerely,", p.name, "Senior Teacher, Department of Physics",
                     f"Mobile: {d['phone']}", f"Email: {d['email']}"]:
            c.text((140, y), line, 32)
            y += 50
        c.specimen()
        return [c.img]
    if kind == "train_ticket":
        W, H = A4[0], 1300
        c = Canvas((W, H))
        c.rect((60, 60, W - 60, H - 60), outline=INK, width=3)
        c.center(110, "RAILYATRA E-TICKET (fictional booking service)", 40, "bold")
        c.center(170, "Electronic Reservation Slip - Passenger copy", 28)
        y = 250
        info = [
            ("PNR No.", d["pnr"]), ("Train No./Name", "12127 / INTERCITY EXP"),
            ("From", "PUNE JN (PUNE)"), ("To", "CSMT MUMBAI (CSMT)"),
            ("Date of Journey", d["date"].strftime("%d-%b-%Y")), ("Class", "CHAIR CAR (CC)"),
            ("Mobile", d["phone"]), ("ID Proof (Aadhaar)", d["aadhaar"]),
        ]
        for label, value in info:
            c.label_value(120, y, f"{label}:", value, 30, label_w=420, value_style="mono_bold")
            y += 52
        y += 30
        c.rect((100, y, W - 100, y + 60), fill=(235, 235, 235))
        for txt, x in [("Passenger", 120), ("Age", 760), ("Gender", 900), ("Coach/Seat", 1100), ("Status", 1350)]:
            c.text((x, y + 15), txt, 28, "bold")
        y += 75
        g = "Male" if p.gender == "M" else "Female"
        for txt, x in [(p.name, 120), (str(d["age"]), 760), (g, 900), ("D4 / 57", 1100), ("CNF", 1350)]:
            c.text((x, y), txt, 28, "mono_bold")
        y += 90
        c.text((120, y), "Total Fare: Rs. 225.00 (incl. reservation charge)", 30, "bold")
        y += 60
        c.text((120, y), "Carry the ID proof in original during the journey.", 26, fill=GREY)
        c.specimen()
        return [c.img]
    # grocery bill: narrow thermal-receipt look
    W, H = 900, 1700
    c = Canvas((W, H), (252, 251, 246))
    c.center(60, "FRESH BASKET KIRANA STORES", 40, "mono_bold")
    c.center(115, "Shop 7, Paud Road, Kothrud, Pune", 26, "mono")
    c.center(150, "GSTIN: 27AAAAA0000A1Z0 (fictional)", 24, "mono")
    y = 220
    c.text((50, y), f"Bill No: {d['bill_no']}", 28, "mono")
    c.text((W - 50, y), d["date"].strftime("%d/%m/%Y"), 28, "mono", anchor="ra")
    y += 45
    c.text((50, y), f"Customer: {p.name}", 28, "mono")
    y += 45
    c.text((50, y), f"Mobile: {d['phone']}", 28, "mono")
    y += 50
    c.text((50, y), "-" * 44, 28, "mono")
    y += 40
    c.text((50, y), "Item", 28, "mono_bold")
    c.text((560, y), "Qty", 28, "mono_bold")
    c.text((W - 50, y), "Amount", 28, "mono_bold", anchor="ra")
    y += 45
    total = 0
    for name, qty, rate in d["items"]:
        amt = qty * rate
        total += amt
        c.text((50, y), name, 28, "mono")
        c.text((580, y), str(qty), 28, "mono")
        c.text((W - 50, y), f"{amt:.2f}", 28, "mono", anchor="ra")
        y += 44
    c.text((50, y), "-" * 44, 28, "mono")
    y += 50
    c.text((50, y), "TOTAL (Rs.)", 34, "mono_bold")
    c.text((W - 50, y), f"{total:.2f}", 34, "mono_bold", anchor="ra")
    y += 60
    c.text((50, y), "Paid by: UPI", 28, "mono")
    y += 80
    c.center(y, "Thank you! Visit again.", 28, "mono")
    c.text((W - 20, H - 20), "SPECIMEN / SYNTHETIC", 18, "regular", GREY, anchor="rb")
    return [c.img]


_RENDERERS = {
    "mark_sheet": render_mark_sheet,
    "student_id_card": render_id_card,
    "fee_receipt": render_fee_receipt,
    "other": render_other,
}


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    """Mild scan artefacts: a few degrees of skew, slight blur, sensor noise, warm paper tint."""
    angle = rng.choice([-1, 1]) * rng.uniform(1.5, 3.0)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(245, 243, 236))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.9))
    arr = np.asarray(img).astype(np.int16)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 9, arr.shape)
    arr = arr + noise.astype(np.int16) - np.array([4, 6, 12], dtype=np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render_plan(plan: DocPlan) -> list[Image.Image]:
    """Render a planned document to page images (scan effects and rotation applied)."""
    pages = _RENDERERS[plan.doc_type](plan)
    rng = random.Random(f"fx:{plan.filename}")
    out = []
    for i, img in enumerate(pages):
        if plan.data.get("noisy"):
            img = degrade(img, rng)
        rot = plan.data.get("rotate", 0)
        if rot and i == 0:
            img = img.rotate(rot, expand=True)  # counter-clockwise, like a sideways scan
        out.append(img)
    return out


def _dpi(plan: DocPlan) -> int:
    return CARD_DPI if plan.doc_type == "student_id_card" else A4_DPI


def write_plan(plan: DocPlan, set_dir: Path) -> Path:
    pages = render_plan(plan)
    path = set_dir / plan.filename
    dpi = _dpi(plan)
    if plan.fmt == "png":
        pages[0].save(path, "PNG", dpi=(dpi, dpi), optimize=False)
    elif plan.fmt == "jpg":
        pages[0].save(path, "JPEG", quality=85 if plan.data.get("noisy") else 92, dpi=(dpi, dpi))
    else:
        _write_pdf(pages, path, dpi)
    return path


def _write_pdf(pages: list[Image.Image], path: Path, dpi: int) -> None:
    import pymupdf

    doc = pymupdf.open()
    for img in pages:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        page = doc.new_page(width=img.width * 72 / dpi, height=img.height * 72 / dpi)
        page.insert_image(page.rect, stream=buf.getvalue())
    doc.set_metadata({
        "title": "SPECIMEN / SYNTHETIC", "author": "OCR_DOC_Checker synthetic generator",
        "creator": "data/generate.py", "producer": "data/generate.py",
        "creationDate": "D:20250101000000Z", "modDate": "D:20250101000000Z",
    })
    doc.save(path, garbage=3, deflate=True, no_new_id=True)
    doc.close()


def write_csv(plans: Iterable[DocPlan], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        for p in plans:
            w.writerow(p.row)


def generate(
    out_dir: Path = DEFAULT_OUT,
    csv_dir: Path = DEFAULT_CSV_DIR,
    seed: int = DEFAULT_SEED,
    sets: Iterable[str] = SETS,
    only: Iterable[int] | None = None,
) -> dict[str, list[DocPlan]]:
    """Plan + render the given sets. `only` restricts rendering to those 1-based indices
    (the CSV is still written for the whole set). Without `only`, stale files are removed first."""
    out_dir, csv_dir = Path(out_dir), Path(csv_dir)
    only_set = set(only) if only is not None else None
    result = {}
    for set_name in sets:
        plans = plan_set(set_name, seed)
        set_dir = out_dir / set_name
        set_dir.mkdir(parents=True, exist_ok=True)
        if only_set is None:
            for stale in set_dir.glob(f"{set_name}_*"):
                stale.unlink()
        for plan in plans:
            if only_set is None or plan.index in only_set:
                write_plan(plan, set_dir)
        write_csv(plans, csv_dir / f"{set_name}_set.csv")
        result[set_name] = plans
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic OCR_DOC_Checker documents.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="samples root (default data/samples)")
    ap.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR, help="where *_set.csv go (default data/)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--set", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--only", type=str, default=None, help="comma-separated 1-based indices to render")
    args = ap.parse_args(argv)
    sets = SETS if args.set == "all" else (args.set,)
    only = [int(x) for x in args.only.split(",")] if args.only else None
    result = generate(args.out, args.csv_dir, args.seed, sets, only)
    for s, plans in result.items():
        print(f"{s}: {len(plans)} documents planned -> {args.out / s}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
