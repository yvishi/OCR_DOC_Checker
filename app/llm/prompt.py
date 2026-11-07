"""Prompt and wire schema for the single classify-and-extract call.

The model only ever sees masked text (or a redacted image). Ranges and lengths are enforced
locally by Pydantic (app.schemas.LLMOutput); the wire schema stays plain because providers'
strict JSON modes reject keywords such as minimum/maxLength.
"""

from __future__ import annotations

PROMPT_VERSION = "extract_v1"

SYSTEM_PROMPT = """\
You read admission documents for LearnSphere College, an Indian college. You receive the OCR text \
of ONE document (or an image of it). Personal data has been replaced with placeholders such as \
<PERSON_1>, <ROLL_NUMBER_1> or <PARENT_NAME_1>; black boxes on images hide the same data.

1. Classify the document:
   - "mark_sheet": a board or school statement of marks / marksheet / grade sheet for a student.
   - "student_id_card": a student identity card.
   - "fee_receipt": a receipt for fees paid.
   - "other": anything else (letters, tickets, bills, certificates that list no marks, ...),
     even if it mentions marks or a percentage.
2. Extract these fields, or null when the document does not show them:
   - student_name: the student's name. Copy the placeholder exactly (e.g. "<PERSON_1>"). Never a parent's name.
   - roll_number: the roll / seat / enrollment / ID number. Copy the placeholder exactly (e.g. "<ROLL_NUMBER_1>").
   - board_or_school: the examining board (e.g. "CBSE", "MSBSHSE", "CISCE") or the issuing school/college name.
   - passing_year: the 4-digit year of passing. For ID cards and fee receipts use null.
   - percentage: the overall percentage exactly as printed, as a number (76.40% -> 76.4).
3. percentage_as_printed: copy the percentage characters exactly as they appear in the input
   (e.g. "76.40%"), or null.

Rules:
- Never compute, estimate or convert. If the document shows only CGPA or grade points, set
  percentage and percentage_as_printed to null and cgpa_only to true. Never turn CGPA into a percentage.
- If the percentage is missing, unreadable, covered by a stamp or garbled, use null. Do not
  recompute it from subject marks or totals.
- Do not guess any field. Prefer null over a guess.
- Return only the JSON object described by the schema.
"""

_NULLABLE_STRING = {"type": ["string", "null"]}

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["document_type", "fields", "percentage_as_printed", "cgpa_only"],
    "properties": {
        "document_type": {"type": "string", "enum": ["mark_sheet", "student_id_card", "fee_receipt", "other"]},
        "fields": {
            "type": "object",
            "additionalProperties": False,
            "required": ["student_name", "roll_number", "board_or_school", "passing_year", "percentage"],
            "properties": {
                "student_name": _NULLABLE_STRING,
                "roll_number": _NULLABLE_STRING,
                "board_or_school": _NULLABLE_STRING,
                "passing_year": {"type": ["integer", "null"]},
                "percentage": {"type": ["number", "null"]},
            },
        },
        "percentage_as_printed": _NULLABLE_STRING,
        "cgpa_only": {"type": "boolean"},
    },
}


def text_messages(masked_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "OCR text of the document (personal data masked):\n\n" + masked_text},
    ]


def image_messages(png_data_urls: list[str]) -> list[dict]:
    content: list[dict] = [
        {"type": "text", "text": "Image(s) of the document; personal data is blacked out. OCR was too weak to use."}
    ]
    content += [{"type": "image_url", "image_url": {"url": url}} for url in png_data_urls]
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
