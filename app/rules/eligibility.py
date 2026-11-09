"""The eligibility decision (docs/ARCHITECTURE.md §4). Python only; the LLM never decides.

Reasons are fixed strings: they never contain names, IDs or the student's figures.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.rules import subjects
from app.rules.guards import non_percentage_metric_shown, percentage_evidence
from app.schemas import Check, CheckStatus, DocumentType, LLMOutput, OcrResult

REASONS = {
    "other": "This is not a mark sheet, student ID card or fee receipt.",
    "not_mark_sheet": "The cutoff check applies to mark sheets only.",
    "vision": "Read from the image only because the scan text was too weak; confirm the percentage manually.",
    "cgpa": "Only CGPA is shown. CGPA is never converted to a percentage.",
    "other_metric": "The result is not given as a percentage (such as percentile, GPA or grades) and is never converted.",
    "unlabeled": "The extracted figure is not labeled as a percentage on the document.",
    "no_text": "No readable text was found, so this image was not sent to the AI model. Check it manually.",
    "missing": "The percentage is missing or unreadable.",
    "ungrounded": "The extracted percentage could not be matched to the document text.",
    "low_conf": "The percentage was read with low OCR confidence (possibly covered by a stamp).",
    "stamp": "A stamp overlaps the percentage or total; confirm the figures manually.",
    "near_cutoff": "The percentage is close to the cutoff and could not be confirmed against the total.",
    "total_mismatch": "The printed total does not match the printed percentage.",
    "subjects": "The subject marks do not add up to the printed percentage.",
    "eligible": "Percentage is at least 60.",
    "not_eligible": "Percentage is below 60.",
}


def decide(llm: LLMOutput, ocr: OcrResult | None, settings: Settings | None = None) -> Check:
    """`ocr` is None on the vision route (no trustworthy word evidence)."""
    settings = settings or get_settings()
    doc = llm.document_type
    if doc is DocumentType.OTHER:
        return Check(status=CheckStatus.REJECTED, reason=REASONS["other"])
    if doc is not DocumentType.MARK_SHEET:
        return Check(status=CheckStatus.NOT_APPLICABLE, reason=REASONS["not_mark_sheet"])

    review = lambda key: Check(status=CheckStatus.NEEDS_REVIEW, reason=REASONS[key])  # noqa: E731
    if llm.cgpa_only:
        return review("cgpa")
    pct = llm.fields.percentage
    if pct is None:
        return review("other_metric" if ocr is not None and non_percentage_metric_shown(ocr.text) else "missing")
    if ocr is None:
        return review("vision")

    ev = percentage_evidence(llm, ocr)
    if not ev.grounded:
        return review("ungrounded")
    if ev.label == "other":
        return review("other_metric")
    if ev.label is None:
        return review("unlabeled")
    if ev.min_conf is None or ev.min_conf < settings.field_min_conf:
        return review("low_conf")
    if ev.stamp_over_figures:
        return review("stamp")
    # Checked before the total and near-cutoff paths so the officer always gets both figures.
    mismatch = subjects.disagrees(ocr.text, pct)
    if mismatch is not None:
        return Check(
            status=CheckStatus.NEEDS_REVIEW,
            reason=REASONS["subjects"],
            printed_percentage=pct,
            recomputed_percentage=mismatch.percentage,
        )
    if ev.total_matches is False:
        return review("total_mismatch")
    if abs(pct - settings.cutoff_percentage) <= settings.near_cutoff_band:
        # Near the cutoff a misread digit flips the result: require the printed total to agree.
        if ev.total_matches is not True:
            return review("near_cutoff")

    # Compare at 2 decimals so 60, 60.0 and 60.00 are all eligible and 59.9 / 59.99 are not.
    if round(pct, 2) >= settings.cutoff_percentage:
        return Check(status=CheckStatus.ELIGIBLE, reason=REASONS["eligible"])
    return Check(status=CheckStatus.NOT_ELIGIBLE, reason=REASONS["not_eligible"])
