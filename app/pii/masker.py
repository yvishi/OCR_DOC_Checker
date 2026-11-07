"""Local PII masking with Presidio. No network calls: spaCy runs locally, and the built-in
EmailRecognizer is replaced by a plain regex because it uses tldextract, which may fetch the
public-suffix list over HTTP.

Nothing in this module logs or prints document text, values or the token map.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry, RecognizerResult
from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import SpacyRecognizer

from app.errors import ErrorClass, PipelineError
from app.pii import recognizers as rec
from app.schemas import MaskResult, OcrResult, OcrWord

# Presidio logs raw document text at DEBUG/INFO (e.g. "Context list is: <text>"). Pin its loggers
# above that, independent of the app's root level, and stop propagation so no handler sees it.
for _name in ("presidio-analyzer", "presidio-anonymizer", "decision_process"):
    _logger = logging.getLogger(_name)
    _logger.setLevel(logging.WARNING)
    _logger.propagate = False

SPACY_MODEL = "en_core_web_lg"
SCORE_THRESHOLD = 0.4

# NER labels we keep from spaCy, and their Presidio names. ORG, NORP, DATE, CARDINAL, PERCENT ...
# are ignored outright: they would eat board names, years, marks and percentages.
_NER_MAPPING = {"PERSON": "PERSON", "PER": "PERSON", "GPE": "LOCATION", "LOC": "LOCATION", "FAC": "LOCATION"}
_NER_IGNORE = [
    "ORG", "ORGANIZATION", "NORP", "DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY",
    "QUANTITY", "EVENT", "LANGUAGE", "LAW", "PRODUCT", "WORK_OF_ART", "MISC",
]
_NER_RECOGNIZERS = {"SpacyRecognizer"}  # results from these go through the protected-term filter

# Entities whose values are identifiers: spaces / hyphens are dropped when normalising, so
# "2345 6789 0123" and "2345-6789-0123" share a token.
_ID_ENTITIES = {rec.IN_AADHAAR, rec.IN_PAN, rec.IN_PHONE, rec.BANK_ACCOUNT, rec.POLICY_NUMBER, rec.ROLL_NUMBER}
# Values of these are searched for again across the text, so a name found once by its label is
# also masked where it appears without one (e.g. an upper-case name in a signature line).
_PROPAGATE = {rec.PERSON, rec.PARENT_NAME, rec.ROLL_NUMBER, rec.IN_AADHAAR, rec.IN_PAN, rec.IN_PHONE, rec.BANK_ACCOUNT}

_EDGE_PUNCT = " \t\r\n.,;:!?\"'’‘“”()[]{}<>-–—/\\|*_"

_analyzer: AnalyzerEngine | None = None
_analyzer_lock = threading.Lock()


@dataclass(frozen=True)
class PiiSpan:
    start: int
    end: int
    entity_type: str
    score: float
    custom: bool  # from one of our recognizers rather than a Presidio built-in


# ---------- analyzer ----------


def _email_recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity=rec.EMAIL_ADDRESS,
        name="EmailPattern",
        patterns=[
            Pattern(
                "email",
                r"(?<![\w.+\-])[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9\-]{1,63}(?:\.[A-Za-z0-9\-]{1,63})*\.[A-Za-z]{2,24}(?![\w])",
                0.9,
            )
        ],
        context=["email", "mail", "e-mail"],
    )


def build_analyzer() -> AnalyzerEngine:
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": rec.LANG, "model_name": SPACY_MODEL}],
            "ner_model_configuration": {
                "model_to_presidio_entity_mapping": _NER_MAPPING,
                "labels_to_ignore": _NER_IGNORE,
                "low_score_entity_names": [],
            },
        }
    )
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry(supported_languages=[rec.LANG])
    registry.add_recognizer(SpacyRecognizer(supported_language=rec.LANG, supported_entities=[rec.PERSON, rec.LOCATION]))
    registry.add_recognizer(_email_recognizer())
    for r in rec.custom_recognizers():
        registry.add_recognizer(r)
    enhancer = LemmaContextAwareEnhancer(
        context_prefix_count=6, context_suffix_count=0, context_matching_mode="whole_word"
    )
    return AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=[rec.LANG],
        context_aware_enhancer=enhancer,
        default_score_threshold=SCORE_THRESHOLD,
    )


def get_analyzer() -> AnalyzerEngine:
    """Module-level lazy singleton: loading spaCy lg takes seconds, so do it once."""
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                _analyzer = build_analyzer()
    return _analyzer


# ---------- detection ----------


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in _EDGE_PUNCT:
        start += 1
    while end > start and text[end - 1] in _EDGE_PUNCT:
        end -= 1
    return start, end


def _alnum_count(s: str) -> int:
    return sum(c.isalnum() for c in s)


def _keep_piece(text: str, s: int, e: int, ner: bool) -> tuple[int, int] | None:
    s, e = _trim(text, s, e)
    if e <= s or _alnum_count(text[s:e]) < 2:
        return None
    if ner and rec.only_deny_words(text[s:e]):
        return None
    return s, e


def _from_presidio(text: str, results: list[RecognizerResult]) -> list[PiiSpan]:
    protected = rec.protected_spans(text)
    spans: list[PiiSpan] = []
    for r in results:
        name = (r.recognition_metadata or {}).get(RecognizerResult.RECOGNIZER_NAME_KEY, "")
        ner = name in _NER_RECOGNIZERS
        custom = not ner and name != "EmailPattern"
        if ner:
            # NER spans never legitimately cross a line break, and never cover protected terms
            breaks = [(m.start(), m.end()) for m in re.finditer(r"\n", text[r.start : r.end])]
            holes = protected + [(r.start + a, r.start + b) for a, b in breaks]
            pieces = rec.subtract_spans(r.start, r.end, holes)
        else:
            pieces = [(r.start, r.end)]
        for s, e in pieces:
            kept = _keep_piece(text, s, e, ner)
            if kept:
                spans.append(PiiSpan(kept[0], kept[1], r.entity_type, float(r.score), custom))
    return spans


def _normalize(entity_type: str, value: str) -> str:
    v = re.sub(r"\s+", " ", value).strip(_EDGE_PUNCT).casefold()
    if entity_type in _ID_ENTITIES:
        v = re.sub(r"[\s\-]", "", v)
    return v


def _propagate(text: str, spans: list[PiiSpan]) -> list[PiiSpan]:
    """Find further occurrences of already-detected values (case / spacing insensitive)."""
    extra: list[PiiSpan] = []
    seen: set[tuple[str, str]] = set()
    for sp in spans:
        if sp.entity_type not in _PROPAGATE:
            continue
        value = text[sp.start : sp.end]
        key = (sp.entity_type, _normalize(sp.entity_type, value))
        if key in seen or _alnum_count(value) < 3 or rec.only_deny_words(value) and sp.entity_type in (rec.PERSON, rec.PARENT_NAME):
            continue
        seen.add(key)
        if sp.entity_type in _ID_ENTITIES:
            chars = [re.escape(c) for c in re.sub(r"[\s\-]", "", value)]
            body = r"[ \-]?".join(chars)
        else:
            body = r"\s+".join(re.escape(w) for w in value.split())
        for m in re.finditer(rf"(?<![\w]){body}(?![\w])", text, re.IGNORECASE):
            extra.append(PiiSpan(m.start(), m.end(), sp.entity_type, sp.score - 0.01, sp.custom))
    return extra


def _resolve(text: str, spans: list[PiiSpan]) -> list[PiiSpan]:
    """Greedy overlap resolution: higher score, then longer, then custom over built-in.

    A losing span that only partly overlaps keeps its uncovered remainder (masking more is the
    safe direction), unless the remainder is trivial or only deny-listed words.
    """
    order = sorted(spans, key=lambda s: (-round(s.score, 6), -(s.end - s.start), not s.custom, s.start))
    accepted: list[PiiSpan] = []
    for sp in order:
        pieces = rec.subtract_spans(sp.start, sp.end, [(a.start, a.end) for a in accepted])
        if pieces == [(sp.start, sp.end)]:
            accepted.append(sp)
            continue
        for s, e in pieces:
            kept = _keep_piece(text, s, e, ner=True)
            if kept:
                accepted.append(PiiSpan(kept[0], kept[1], sp.entity_type, sp.score, sp.custom))
    return sorted(accepted, key=lambda s: s.start)


def detect_pii(text: str) -> list[PiiSpan]:
    """Non-overlapping PII spans in ``text``, sorted by start offset."""
    if not text or not text.strip():
        return []
    try:
        results = get_analyzer().analyze(
            text=text, language=rec.LANG, entities=list(rec.ALL_ENTITIES), score_threshold=SCORE_THRESHOLD
        )
        spans = _from_presidio(text, results)
        spans = _resolve(text, spans)
        spans = _resolve(text, spans + _propagate(text, spans))
        return spans
    except PipelineError:
        raise
    except Exception:  # never let a message that may quote text escape
        raise PipelineError(ErrorClass.PII_FAILED) from None


# ---------- masking ----------


def mask_text(text: str) -> MaskResult:
    """Replace each PII span with a stable ``<ENTITY_N>`` placeholder.

    Same entity type + same normalised value (case, whitespace, edge punctuation; also spaces and
    hyphens for ID-like types) -> same placeholder. Numbering is per type, in order of first
    appearance, and starts at 1 on every call. ``token_map`` maps placeholder -> first surface form.
    """
    spans = detect_pii(text)
    try:
        token_for: dict[tuple[str, str], str] = {}
        counters: dict[str, int] = {}
        token_map: dict[str, str] = {}
        out: list[str] = []
        pos = 0
        for sp in spans:
            value = text[sp.start : sp.end]
            key = (sp.entity_type, _normalize(sp.entity_type, value))
            token = token_for.get(key)
            if token is None:
                counters[sp.entity_type] = counters.get(sp.entity_type, 0) + 1
                token = f"<{sp.entity_type}_{counters[sp.entity_type]}>"
                token_for[key] = token
                token_map[token] = value
            out.append(text[pos : sp.start])
            out.append(token)
            pos = sp.end
        out.append(text[pos:])
        return MaskResult(masked_text="".join(out), entity_counts=dict(counters), token_map=token_map)
    except Exception:
        raise PipelineError(ErrorClass.PII_FAILED) from None


# ---------- OCR words ----------


def build_text_with_offsets(words: list[OcrWord]) -> tuple[str, list[tuple[int, int]]]:
    """Rebuild ``OcrResult.text`` exactly as the OCR stage does, recording each word's offsets.

    Spaces within a line, ``\\n`` between lines (a new line starts whenever ``line_key`` changes
    within a page), ``\\n\\n`` between pages (pages in sorted order). Returns the text and a list
    of ``(start, end)`` aligned with ``words``.
    """
    pages: dict[int, list[list[int]]] = {}
    last_key: dict[int, tuple[int, int, int]] = {}
    for i, w in enumerate(words):
        lines = pages.setdefault(w.page, [])
        if not lines or last_key.get(w.page) != w.line_key:
            lines.append([])
            last_key[w.page] = w.line_key
        lines[-1].append(i)

    offsets: list[tuple[int, int]] = [(0, 0)] * len(words)
    parts: list[str] = []
    pos = 0
    for pi, p in enumerate(sorted(pages)):
        if pi:
            parts.append("\n\n")
            pos += 2
        for li, line in enumerate(pages[p]):
            if li:
                parts.append("\n")
                pos += 1
            for wi, idx in enumerate(line):
                if wi:
                    parts.append(" ")
                    pos += 1
                t = words[idx].text
                offsets[idx] = (pos, pos + len(t))
                parts.append(t)
                pos += len(t)
    return "".join(parts), offsets


def find_pii_words(ocr: OcrResult) -> list[OcrWord]:
    """OCR words that overlap any detected PII span in ``ocr.text`` (in ``ocr.words`` order)."""
    text, offsets = build_text_with_offsets(ocr.words)
    if text != ocr.text:
        # offsets would be wrong; refuse rather than redact the wrong boxes
        raise PipelineError(ErrorClass.PII_FAILED)
    spans = detect_pii(text)
    if not spans:
        return []
    hits: list[OcrWord] = []
    for w, (s, e) in zip(ocr.words, offsets):
        if e > s and any(sp.start < e and s < sp.end for sp in spans):
            hits.append(w)
    return hits
