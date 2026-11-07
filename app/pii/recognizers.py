"""Custom Presidio recognizers for Indian admission documents, plus the protected-term lists.

Two kinds of recognizer live here:

* ``PatternRecognizer`` instances (pattern + context words). Presidio's context enhancer adds
  0.35 to the score when a context word appears among the few words before the match.
* ``LabelAnchoredRecognizer``: a regex with a named group ``value``. The label ("Roll No",
  "Father's Name", "DOB", "Address", ...) anchors the match, and only the ``value`` group is
  reported, so the label itself survives masking. These get a high score because a label on the
  same line is the strongest signal we have.

Nothing here logs or prints document text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from presidio_analyzer import (
    AnalysisExplanation,
    EntityRecognizer,
    Pattern,
    PatternRecognizer,
    RecognizerResult,
)

# ---------- entity names ----------

IN_AADHAAR = "IN_AADHAAR"
IN_PAN = "IN_PAN"
IN_PHONE = "IN_PHONE"
BANK_ACCOUNT = "BANK_ACCOUNT"
POLICY_NUMBER = "POLICY_NUMBER"
ROLL_NUMBER = "ROLL_NUMBER"
DATE_OF_BIRTH = "DATE_OF_BIRTH"
PARENT_NAME = "PARENT_NAME"
ADDRESS = "ADDRESS"
PERSON = "PERSON"
LOCATION = "LOCATION"
EMAIL_ADDRESS = "EMAIL_ADDRESS"

CUSTOM_ENTITIES = (
    IN_AADHAAR,
    IN_PAN,
    IN_PHONE,
    BANK_ACCOUNT,
    POLICY_NUMBER,
    ROLL_NUMBER,
    DATE_OF_BIRTH,
    PARENT_NAME,
    ADDRESS,
)
# Built-ins we keep. Everything else (DATE_TIME, NRP, ORGANIZATION, US_*, credit cards, URLs,
# phonenumbers-based PHONE_NUMBER, ...) is never registered.
BUILTIN_ENTITIES = (PERSON, LOCATION, EMAIL_ADDRESS)
ALL_ENTITIES = CUSTOM_ENTITIES + BUILTIN_ENTITIES

LABEL_SCORE = 0.95
LANG = "en"

_CASE_SENSITIVE = re.MULTILINE | re.DOTALL  # for PAN / IFSC: upper case is part of the format

# ---------- shared regex fragments ----------

_SEP = r"[ \t]*[:\-–.#]?[ \t]*"  # between a label and its value: "Roll No: X", "DOB - X", "Seat No. X"
_NO = r"(?:no\b\.?|num\b\.?|number|#)?"

# Words that start another field on the same line. A captured name / address stops before them.
_STOP_LABELS = (
    r"roll|seat|enrol(?:l)?ment|enrolment|registration|reg|dob|d\.o\.b|date|birth|born|"
    r"father|mother|parent|guardian|name|gender|sex|class|section|school|college|board|"
    r"address|phone|mobile|mob|tel|contact|email|e-mail|aadhaar|aadhar|pan|"
    r"account|a/c|ifsc|policy|year|percentage|total|result|signature|category|"
    r"student|candidate|id|age|blood|nationality|religion|division|grade|exam|examination|"
    r"centre|center|medium|stream|subject"
)
_NAME_WORD = rf"(?!(?:{_STOP_LABELS})\b)[A-Za-z][A-Za-z.'’\-]*"
_NAME_VALUE = rf"(?P<value>{_NAME_WORD}(?:[ \t]+{_NAME_WORD}){{0,5}})"

_MONTHS = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_DATE = (
    r"(?:\d{1,2}[/\-.]\d{1,2}[/\-.](?:\d{4}|\d{2})"  # 12/03/2007, 12-03-07
    rf"|\d{{1,2}}(?:st|nd|rd|th)?[ \t\-]+{_MONTHS}\.?,?[ \t\-]+\d{{4}}"  # 12 Mar 2007, 12-March-2007
    rf"|{_MONTHS}\.?[ \t]+\d{{1,2}}(?:st|nd|rd|th)?,?[ \t]+\d{{4}}"  # March 12, 2007
    r"|\d{4}-\d{2}-\d{2})"  # 2007-03-12
)


class LabelAnchoredRecognizer(EntityRecognizer):
    """Regex recognizer that reports only the ``value`` named group of each match.

    ``exclude_before`` is an optional regex tested against the 25 characters before the match;
    when it matches, the hit is skipped (used so a bare "Name:" label does not fire inside
    "Father's Name:").  ``value_must`` is an optional regex the value has to contain.
    """

    def __init__(
        self,
        supported_entity: str,
        name: str,
        regex: str,
        score: float = LABEL_SCORE,
        flags: int = re.IGNORECASE | re.MULTILINE,
        exclude_before: str | None = None,
        value_must: str | None = None,
    ) -> None:
        self._regex = re.compile(regex, flags)
        self._score = score
        self._exclude_before = re.compile(exclude_before, re.IGNORECASE) if exclude_before else None
        self._value_must = re.compile(value_must) if value_must else None
        super().__init__(supported_entities=[supported_entity], name=name, supported_language=LANG)

    def load(self) -> None:  # nothing to load
        pass

    def analyze(self, text: str, entities: list[str], nlp_artifacts=None) -> list[RecognizerResult]:
        entity = self.supported_entities[0]
        if entities and entity not in entities:
            return []
        results: list[RecognizerResult] = []
        for m in self._regex.finditer(text):
            start, end = m.span("value")
            if start < 0 or end <= start:
                continue
            value = text[start:end]
            # trim trailing separators / punctuation the greedy value may have grabbed
            stripped = value.rstrip(" \t,;:.-–/")
            end = start + len(stripped)
            if end <= start:
                continue
            if self._value_must and not self._value_must.search(stripped):
                continue
            if self._exclude_before and self._exclude_before.search(text[max(0, m.start() - 25) : m.start()]):
                continue
            explanation = AnalysisExplanation(
                recognizer=self.name,
                original_score=self._score,
                textual_explanation=f"Label-anchored {entity}",
            )
            results.append(
                RecognizerResult(
                    entity_type=entity,
                    start=start,
                    end=end,
                    score=self._score,
                    analysis_explanation=explanation,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                        RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                        # the label already is the context; don't boost again
                        RecognizerResult.IS_SCORE_ENHANCED_BY_CONTEXT_KEY: True,
                    },
                )
            )
        return results


# ---------- pattern recognizers (pattern + context) ----------


def _aadhaar() -> PatternRecognizer:
    # No Verhoeff check: synthetic numbers would fail it. Pattern + context only.
    return PatternRecognizer(
        supported_entity=IN_AADHAAR,
        name="InAadhaarPattern",
        patterns=[
            Pattern("aadhaar_grouped", r"(?<![\d\w])[2-9]\d{3}([ \-])\d{4}\1\d{4}(?![\d\w])", 0.6),
            Pattern("aadhaar_plain", r"(?<![\d\w])[2-9]\d{11}(?![\d\w])", 0.4),
        ],
        context=["aadhaar", "aadhar", "adhaar", "uid", "uidai", "enrolment"],
    )


def _pan() -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity=IN_PAN,
        name="InPanPattern",
        patterns=[Pattern("pan", r"(?<![A-Za-z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Za-z0-9])", 0.6)],
        context=["pan", "permanent", "income", "tax"],
        global_regex_flags=_CASE_SENSITIVE,
    )


def _phone() -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity=IN_PHONE,
        name="InPhonePattern",
        patterns=[
            Pattern("in_mobile_cc", r"(?<![\w+])(?:\+91|91)[ \-]?[6-9]\d{4}[ \-]?\d{5}(?!\d)", 0.7),
            Pattern("in_mobile_0", r"(?<![\w+])0[6-9]\d{4}[ \-]?\d{5}(?!\d)", 0.5),
            Pattern("in_mobile", r"(?<![\w+])[6-9]\d{4}[ \-]?\d{5}(?!\d)", 0.45),
            # landline with STD code, e.g. 080-23456789; needs context
            Pattern("in_landline", r"(?<![\w+])0\d{2,4}[ \-]\d{6,8}(?!\d)", 0.2),
        ],
        context=["phone", "mobile", "mob", "tel", "telephone", "contact", "cell", "whatsapp", "ph"],
    )


def _bank_account() -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity=BANK_ACCOUNT,
        name="BankAccountPattern",
        patterns=[
            Pattern("account_digits", r"(?<![\w])\d{9,18}(?![\w])", 0.1),  # needs context
        ],
        context=["account", "acct", "acc", "ifsc", "bank", "savings", "a/c"],
    )


def _ifsc() -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity=BANK_ACCOUNT,
        name="IfscPattern",
        patterns=[Pattern("ifsc", r"(?<![A-Za-z0-9])[A-Z]{4}0[A-Z0-9]{6}(?![A-Za-z0-9])", 0.6)],
        context=["ifsc", "bank", "branch"],
        global_regex_flags=_CASE_SENSITIVE,
    )


def _upi() -> PatternRecognizer:
    # name@bank, but not an email address (no ".tld" after the handle)
    return PatternRecognizer(
        supported_entity=BANK_ACCOUNT,
        name="UpiPattern",
        patterns=[
            Pattern("upi", r"(?<![\w.\-@])[\w.\-]{2,64}@[A-Za-z][A-Za-z0-9]{1,30}(?![\w@]|\.[A-Za-z])", 0.5)
        ],
        context=["upi", "vpa", "gpay", "paytm", "phonepe", "bhim"],
    )


def _policy() -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity=POLICY_NUMBER,
        name="PolicyNumberPattern",
        patterns=[Pattern("policy_digits", r"(?<![\w])[A-Z0-9][A-Z0-9/\-]{6,19}\d(?![\w])", 0.05)],
        context=["policy", "insurance", "insured", "mediclaim"],
    )


def _roll_patterns() -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity=ROLL_NUMBER,
        name="RollNumberPattern",
        patterns=[
            # CBSE24-104482, MH2024/123456: a board-ish prefix with year and serial
            Pattern("board_roll", r"(?<![\w\-])[A-Z]{2,6}\d{2,4}[\-/]\d{4,8}(?![\w\-])", 0.5),
            Pattern("letter_roll", r"(?<![\w])[A-Z]\d{6,9}(?![\w])", 0.2),  # M123456; needs context
            Pattern("digit_roll", r"(?<![\w.,])\d{6,10}(?![\w.,%])", 0.1),  # needs context
        ],
        context=["roll", "seat", "enrollment", "enrolment", "registration", "reg", "candidate", "admit"],
        global_regex_flags=re.MULTILINE | re.DOTALL,
    )


# ---------- label-anchored recognizers ----------


def _roll_label() -> LabelAnchoredRecognizer:
    strong = r"roll|seat|enrol(?:l)?ment|enrolment|exam(?:ination)?[ \t]+roll|hall[ \t]+ticket|admit[ \t]+card"
    weak = r"registration|regn?\.?|reg|application|admission|index|candidate|student|exam(?:ination)?|unique|uid"
    regex = (
        rf"(?:\b(?:{strong})[ \t]*{_NO}|\b(?:{weak})[ \t]*(?:no\b\.?|num\b\.?|number|#|id\b|code\b)){_SEP}"
        r"(?P<value>[A-Z0-9](?:[A-Z0-9]|[\-/](?=[A-Z0-9])){3,24})(?![A-Za-z0-9])"
    )
    return LabelAnchoredRecognizer(ROLL_NUMBER, "RollNumberLabel", regex, value_must=r"\d")


def _dob_label() -> LabelAnchoredRecognizer:
    label = r"(?:\bd[ \t]*\.?[ \t]*o[ \t]*\.?[ \t]*b\b\.?|\bdate[ \t]+of[ \t]+birth\b|\bbirth[ \t]*date\b|\bborn[ \t]+on\b)"
    regex = rf"{label}[ \t]*(?:\([^)\n]{{0,20}}\))?{_SEP}(?P<value>{_DATE})(?![\d])"
    return LabelAnchoredRecognizer(DATE_OF_BIRTH, "DateOfBirthLabel", regex)


def _parent_label() -> LabelAnchoredRecognizer:
    who = r"(?:father|mother|guardian|parent)(?:['’]?s)?"
    label = (
        rf"(?:\b{who}(?:[ \t]*/[ \t]*{who})?"
        r"(?:[ \t]+(?:full[ \t]+)?name[ \t]*[:\-–.]?|[ \t]*[:\-–])"
        rf"|\bname[ \t]+of[ \t]+(?:the[ \t]+)?{who}[ \t]*[:\-–.]?"
        r"|\b[sdw][ \t]*/[ \t]*o\b[ \t]*[:\-–.]?)"
    )
    regex = rf"{label}[ \t]*(?:(?:mr|mrs|ms|shri|smt|dr)\.?[ \t]+)?{_NAME_VALUE}"
    return LabelAnchoredRecognizer(PARENT_NAME, "ParentNameLabel", regex)


def _student_name_label() -> LabelAnchoredRecognizer:
    label = (
        r"(?:\b(?:student|candidate|learner|applicant)(?:['’]?s)?[ \t]+(?:full[ \t]+)?name"
        r"|\bname[ \t]+of[ \t]+(?:the[ \t]+)?(?:student|candidate|learner|applicant)"
        r"|\bfull[ \t]+name|\bname)"
    )
    regex = rf"{label}[ \t]*[:\-–][ \t]*(?:(?:mr|mrs|ms|miss|kumari|km)\.?[ \t]+)?{_NAME_VALUE}"
    # skip "Father's Name:", "Mother Name:", "Guardian Name:", "School Name:"
    exclude = r"(?:father|mother|guardian|parent|school|college|board|institute|bank|branch)(?:['’]?s)?[ \t]*\Z"
    return LabelAnchoredRecognizer(PERSON, "StudentNameLabel", regex, exclude_before=exclude)


def _address_label() -> LabelAnchoredRecognizer:
    stop = r"phone|mobile|mob|tel|contact|email|e-mail|dob|d\.o\.b|date[ \t]+of[ \t]+birth|roll|aadhaar|aadhar|pan"
    regex = (
        r"\b(?:permanent[ \t]+|residential[ \t]+|postal[ \t]+|correspondence[ \t]+|home[ \t]+)?address\b"
        rf"(?:[ \t]+for[ \t]+correspondence)?{_SEP}"
        rf"(?P<value>(?:(?!\b(?:{stop})\b)[^\n])+)"
    )
    return LabelAnchoredRecognizer(ADDRESS, "AddressLabel", regex, value_must=r"[A-Za-z0-9]{2}")


def _bank_label() -> LabelAnchoredRecognizer:
    regex = (
        rf"(?:\b(?:bank[ \t]+)?(?:account|acct|acc)\b|\ba/c\b)[ \t]*{_NO}{_SEP}"
        r"(?P<value>\d(?:[ \-]?\d){7,21})(?!\d)"
    )
    return LabelAnchoredRecognizer(BANK_ACCOUNT, "BankAccountLabel", regex)


def _policy_label() -> LabelAnchoredRecognizer:
    regex = (
        rf"\b(?:insurance[ \t]+)?policy[ \t]*{_NO}{_SEP}"
        r"(?P<value>[A-Z0-9](?:[A-Z0-9]|[\-/](?=[A-Z0-9])){4,24})(?![A-Za-z0-9])"
    )
    return LabelAnchoredRecognizer(POLICY_NUMBER, "PolicyNumberLabel", regex, value_must=r"\d")


def _addressee_label() -> LabelAnchoredRecognizer:
    """Letter addressee/salutation: "To,\\n<Name>" and "Dear <Name>" (spaCy often calls these ORG)."""
    honorific = r"(?:(?:mr|mrs|ms|miss|shri|smt|dr|prof)\.?[ \t]+)?"
    regex = rf"(?:^[ \t]*to[ \t]*,?[ \t]*\n[ \t]*|\bdear[ \t]+){honorific}{_NAME_VALUE}"
    return LabelAnchoredRecognizer(PERSON, "AddresseeLabel", regex)


def _name_table_row() -> LabelAnchoredRecognizer:
    """First cell of the row under a table header that starts with Passenger/Name (tickets, lists).

    Case-sensitive: 1-3 capitalised words at line start, so headers like "Total Fare" are not names.
    """
    header = r"^[ \t]*(?:(?i:passenger(?:[ \t]+name)?|name(?:[ \t]+of[ \t]+passenger)?))\b[^\n]*\n[ \t]*"
    word = r"[A-Z][a-z][A-Za-z'’\-]*"
    regex = rf"{header}(?P<value>{word}(?:[ \t]+{word}){{1,2}})(?=[ \t]|$)"
    return LabelAnchoredRecognizer(PERSON, "NameTableRow", regex, flags=re.MULTILINE)


def custom_recognizers() -> list[EntityRecognizer]:
    """All custom recognizers, freshly built."""
    return [
        _aadhaar(),
        _pan(),
        _phone(),
        _bank_account(),
        _ifsc(),
        _upi(),
        _policy(),
        _roll_patterns(),
        _roll_label(),
        _dob_label(),
        _parent_label(),
        _student_name_label(),
        _address_label(),
        _bank_label(),
        _policy_label(),
        _addressee_label(),
        _name_table_row(),
    ]


# ---------- protected terms (never PERSON / LOCATION) ----------
#
# spaCy happily tags board names, subjects and headings as PERSON / LOCATION / ORG. ORG is never
# registered at all. For PERSON / LOCATION from the NER model we cut out any part that overlaps a
# protected phrase and drop what is left if it is only deny-listed words.

PROTECTED_PHRASES: tuple[str, ...] = (
    # boards and institutions
    r"central[ \t]+board[ \t]+of[ \t]+secondary[ \t]+education",
    r"council[ \t]+for[ \t]+the[ \t]+indian[ \t]+school[ \t]+certificate[ \t]+examinations?",
    r"national[ \t]+institute[ \t]+of[ \t]+open[ \t]+schooling",
    r"(?:[A-Z][a-z]+[ \t]+){0,3}state[ \t]+board(?:[ \t]+of(?:[ \t]+(?:and|&|[A-Z][a-z]+))+)?",
    r"(?:[A-Z][a-z]+[ \t]+){0,3}board[ \t]+of(?:[ \t]+(?:and|&|[A-Z][a-z]+))+",
    r"(?:[A-Z][A-Za-z]+[ \t]+){0,3}(?:college|school|university|institute|vidyalaya|academy)\b",
    r"\b(?:cbse|icse|isc|cisce|nios|hsc|ssc|igcse|learnsphere)\b",
    # document titles and headings
    r"statement[ \t]+of[ \t]+marks", r"marks?[ \t]*sheet", r"grade[ \t]+sheet", r"mark[ \t]+statement",
    r"(?:provisional[ \t]+|migration[ \t]+|passing[ \t]+|school[ \t]+leaving[ \t]+)?certificate",
    r"(?:senior[ \t]+)?(?:secondary|higher[ \t]+secondary)[ \t]+(?:school[ \t]+)?(?:certificate[ \t]+)?examination",
    r"all[ \t]+india", r"fee[ \t]+receipt", r"(?:student[ \t]+)?(?:identity|id)[ \t]+card",
    r"controller[ \t]+of[ \t]+examinations?", r"year[ \t]+of[ \t]+passing",
    # subjects (multi-word first)
    r"social[ \t]+science", r"computer[ \t]+science", r"political[ \t]+science", r"business[ \t]+studies",
    r"informatics[ \t]+practices", r"physical[ \t]+education", r"environmental[ \t]+(?:studies|science)",
    r"information[ \t]+technology", r"home[ \t]+science", r"fine[ \t]+arts?",
)

DENY_WORDS: frozenset[str] = frozenset(
    w.lower()
    for w in """
    english hindi marathi sanskrit tamil telugu kannada bengali gujarati urdu punjabi malayalam
    french german odia assamese mathematics maths math physics chemistry biology science history
    geography civics economics accountancy accounts commerce psychology sociology philosophy
    statistics music art arts painting drawing
    cbse icse isc cisce nios hsc ssc board state central council secondary higher senior education
    examination examinations exam certificate statement marks mark sheet marksheet grade grades
    total percentage result pass passed fail division distinction first second third class year
    passing obtained maximum max min minimum theory practical internal external subject subjects
    code roll seat number no name date birth principal registrar controller signature seal
    learnsphere college school university institute academy vidyalaya admission department
    father mother guardian parent parents student candidate learner applicant address permanent
    residential mobile phone contact email dob aadhaar aadhar pan enrollment enrolment
    registration reg seat id card identity receipt fee fees amount paid gender male female
    signature valid till issue issued blood group section stream medium category nationality
    india maharashtra karnataka kerala gujarat rajasthan punjab haryana bihar odisha assam
    telangana andhra pradesh uttar madhya tamil nadu west bengal goa delhi
    grand aggregate overall cgpa sgpa gpa percent remarks marks-obtained
    """.split()
)

_PROTECTED_RE = re.compile("|".join(f"(?:{p})" for p in PROTECTED_PHRASES), re.IGNORECASE)


def protected_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _PROTECTED_RE.finditer(text) if m.end() > m.start()]


# Grade codes printed in mark-sheet tables (A1, B2, C1, D, E2): spaCy sometimes tags them as places.
_GRADE_CODE = re.compile(r"[a-e][12]?", re.IGNORECASE)


def only_deny_words(fragment: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+", fragment)
    return not words or all(
        w.lower() in DENY_WORDS or w.isdigit() or _GRADE_CODE.fullmatch(w) for w in words
    )


def subtract_spans(start: int, end: int, holes: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Parts of [start, end) not covered by any hole."""
    parts = [(start, end)]
    for hs, he in holes:
        nxt = []
        for s, e in parts:
            if he <= s or hs >= e:
                nxt.append((s, e))
                continue
            if hs > s:
                nxt.append((s, hs))
            if he < e:
                nxt.append((he, e))
        parts = nxt
    return parts
