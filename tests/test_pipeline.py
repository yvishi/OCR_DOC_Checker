"""End-to-end tests for increment 1 (upload -> OCR -> route -> mask) over the synthetic samples."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import telemetry as tel
from app.config import Settings
from app.errors import GENERIC_MESSAGE, ErrorClass, PipelineError
from app.pipeline import Route, prepare
from tests.telemetry_helpers import assert_no_leak, read_jsonl

DATA = Path(__file__).resolve().parent.parent / "data"


def _rows() -> list[dict[str, str]]:
    rows = []
    for name, folder in (("dev_set.csv", "dev"), ("test_set.csv", "test")):
        csv_path = DATA / name
        if not csv_path.exists():
            continue
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row["path"] = str(DATA / "samples" / folder / row["filename"])
                rows.append(row)
    return rows


ROWS = _rows()
pytestmark = pytest.mark.skipif(not ROWS, reason="run `python -m data.generate` first")


def _secrets(row: dict[str, str]) -> list[str]:
    values = [v for v in row.get("pii_values", "").split("|") if v.strip()]
    values += [row[k] for k in ("student_name", "roll_number") if row.get(k)]
    return sorted(set(values))


@pytest.fixture
def otel(tmp_path):
    spans, path = InMemorySpanExporter(), tmp_path / "telemetry.jsonl"
    tel.reset_telemetry()
    tel.setup_telemetry(exporter=spans, jsonl_path=path, metric_reader=InMemoryMetricReader())
    yield spans, path
    tel.reset_telemetry()


@pytest.fixture
def settings(tmp_path):
    return Settings(upload_dir=tmp_path / "uploads")


def _run(path: str, settings: Settings):
    try:
        return prepare(Path(path).read_bytes(), settings)
    except PipelineError as exc:
        assert str(exc) == GENERIC_MESSAGE
        return exc


def test_telemetry_never_leaks_pii(otel, settings, caplog):
    """Every synthetic sample: no name, roll number or other PII value in spans or logs."""
    spans, jsonl = otel
    caplog.set_level(logging.DEBUG)
    all_secrets: set[str] = set()
    for row in ROWS:
        _run(row["path"], settings)
        all_secrets.update(_secrets(row))
    tel.force_flush()
    assert spans.get_finished_spans(), "no spans exported"
    assert_no_leak(spans.get_finished_spans(), read_jsonl(jsonl) + caplog.messages, sorted(all_secrets))


def test_uploads_are_deleted(otel, settings):
    for row in ROWS[:4]:
        _run(row["path"], settings)
    assert not any(settings.upload_dir.glob("*")), "an upload was left on disk"


def test_clean_documents_take_text_route_and_mask_identity(otel, settings):
    """Clean, upright docs: text route, and the student's name and roll number never survive masking."""
    clean = [r for r in ROWS if r.get("variant") == "clean"]
    assert clean, "dataset has no clean variants"
    for row in clean:
        result = _run(row["path"], settings)
        assert not isinstance(result, PipelineError), f"{row['filename']}: {result.error_class}"
        assert result.route is Route.TEXT, row["filename"]
        assert result.masked_text
        lowered = " ".join(result.masked_text.lower().split())
        for value in (row.get("student_name"), row.get("roll_number")):
            if value:
                assert " ".join(value.lower().split()) not in lowered, f"{row['filename']}: identity leaked to LLM text"


def test_masked_text_has_no_pii_values_on_text_route(otel, settings):
    """Across all samples that take the text route, report any PII value that survives masking."""
    leaks: list[str] = []
    for row in ROWS:
        result = _run(row["path"], settings)
        if isinstance(result, PipelineError) or result.route is not Route.TEXT:
            continue
        lowered = " ".join(result.masked_text.lower().split())
        for value in _secrets(row):
            if len(value) >= 4 and " ".join(value.lower().split()) in lowered:
                leaks.append(f"{row['filename']}:{row.get('variant')}")
                break
    assert not leaks, f"PII survived masking in: {leaks}"


def test_garbage_upload_gets_generic_error(otel, settings):
    with pytest.raises(PipelineError) as info:
        prepare(b"not a document at all", settings)
    assert info.value.error_class is ErrorClass.UPLOAD_BAD_TYPE
    assert str(info.value) == GENERIC_MESSAGE


def test_percentage_survives_masking_on_readable_mark_sheets(otel, settings):
    """The rule stage needs the figures: masking must never eat the printed percentage line."""
    missing: list[str] = []
    for row in ROWS:
        if row["true_type"] != "mark_sheet" or not row["percentage"] or row["variant"] == "stamp_over_total":
            continue
        result = _run(row["path"], settings)
        assert not isinstance(result, PipelineError), row["filename"]
        line = next((ln for ln in result.masked_text.splitlines() if "percentage" in ln.lower()), "")
        whole = row["percentage"].split(".")[0]
        if "<" in line or whole not in line:
            missing.append(f"{row['filename']}:{row['variant']}")
    assert not missing, f"percentage lost in: {missing}"


# ---------- LLM boundary (mocked transport: no network) ----------


def test_llm_request_never_contains_pii(otel, settings):
    """The single outbound call: across every sample, no PII value appears in the request body."""
    import base64
    import json

    import httpx
    from pydantic import SecretStr

    from app.pipeline import process

    canned = {
        "document_type": "other",
        "fields": {"student_name": None, "roll_number": None, "board_or_school": None, "passing_year": None, "percentage": None},
        "percentage_as_printed": None,
        "cgpa_only": False,
    }
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(canned)}}], "usage": {}})

    llm_settings = settings.model_copy(update={"llm_api_key": SecretStr("sk-or-test"), "llm_model": "test-model"})
    leaks: list[str] = []
    for row in ROWS:
        before = len(bodies)
        try:
            process(Path(row["path"]).read_bytes(), llm_settings, httpx.MockTransport(handler))
        except PipelineError:
            continue
        assert len(bodies) == before + 1, f"{row['filename']}: expected exactly one LLM call"
        sent = " ".join(json.loads(bodies[-1]).__str__().lower().split())
        for value in _secrets(row):
            if len(value) >= 4 and " ".join(value.lower().split()) in sent:
                leaks.append(f"{row['filename']}: {value[:3]}…")
    assert bodies, "no LLM calls were made"
    assert not leaks, f"PII sent to the LLM: {leaks}"


def test_process_returns_validated_output(otel, settings):
    import json

    import httpx
    from pydantic import SecretStr

    from app.pipeline import process
    from app.schemas import DocumentType

    canned = {
        "document_type": "mark_sheet",
        "fields": {"student_name": "<PERSON_1>", "roll_number": "<ROLL_NUMBER_1>", "board_or_school": "CBSE", "passing_year": 2023, "percentage": 76.4},
        "percentage_as_printed": "76.40%",
        "cgpa_only": False,
    }
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(canned)}}], "usage": {"prompt_tokens": 9, "completion_tokens": 3}})
    )
    llm_settings = settings.model_copy(update={"llm_api_key": SecretStr("k"), "llm_model": "test-model"})
    row = next(r for r in ROWS if r["true_type"] == "mark_sheet" and r["variant"] == "clean")
    out = process(Path(row["path"]).read_bytes(), llm_settings, transport)
    assert out.llm.document_type is DocumentType.MARK_SHEET
    assert out.prompt_tokens == 9 and "llm" in out.prepared.timings_ms
    assert not any(llm_settings.upload_dir.glob("*"))


def test_extract_restores_identity_and_decides(otel, settings):
    """Full request with a canned LLM reply: real name/roll restored locally, rule applied."""
    import json

    import httpx
    from pydantic import SecretStr

    from app.pipeline import extract
    from app.schemas import CheckStatus

    row = next(r for r in ROWS if r["true_type"] == "mark_sheet" and r["variant"] == "clean" and float(r["percentage"]) >= 62)
    printed = f"{float(row['percentage']):.2f}%"
    canned = {
        "document_type": "mark_sheet",
        "fields": {"student_name": "<PERSON_1>", "roll_number": "<ROLL_NUMBER_1>", "board_or_school": row["board_or_school"], "passing_year": int(row["passing_year"]), "percentage": float(row["percentage"])},
        "percentage_as_printed": printed,
        "cgpa_only": False,
    }
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(canned)}}], "usage": {}})
    )
    llm_settings = settings.model_copy(update={"llm_api_key": SecretStr("k"), "llm_model": "test-model"})
    resp = extract(Path(row["path"]).read_bytes(), llm_settings, transport)
    assert resp.fields.student_name.lower() == row["student_name"].lower()
    assert resp.fields.roll_number == row["roll_number"]
    assert resp.check.status is CheckStatus.ELIGIBLE
    assert set(resp.pipeline.timings_ms) == {"ocr", "pii", "llm", "rule"}
    assert resp.pii.masked_before_llm and resp.pii.entity_counts.get("PERSON")
    body = resp.model_dump_json()
    assert "<PERSON_" not in body and "<ROLL_NUMBER_" not in body  # placeholders never reach the client


# ---------- no text / weak text routes (mocked transport: no network) ----------


def _png_bytes(im) -> bytes:
    import io

    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _blank_png() -> bytes:
    from PIL import Image

    return _png_bytes(Image.new("RGB", (1600, 1000), "white"))


def _blank_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    return doc.tobytes()


def _weak_scan() -> bytes:
    """A readable name and nothing else: under the 40-character text gate, with word boxes."""
    from PIL import Image, ImageDraw

    from tests.test_ocr import _font

    im = Image.new("RGB", (1600, 1000), "white")
    ImageDraw.Draw(im).text((100, 100), "Name: Ravi Menon", fill="black", font=_font(56))
    return _png_bytes(im)


def _weak_then_blank_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "Name: Ravi Menon", fontsize=22)
    doc.new_page()
    return doc.tobytes()


def _canned_llm(doc_type: str = "other", percentage: float | None = None):
    """Settings plus a mock transport that records every outbound request body."""
    import json

    import httpx

    printed = f"{percentage:.2f}%" if percentage is not None else None
    canned = {
        "document_type": doc_type,
        "fields": {"student_name": None, "roll_number": None, "board_or_school": None, "passing_year": None, "percentage": percentage},
        "percentage_as_printed": printed,
        "cgpa_only": False,
    }
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(canned)}}], "usage": {}})

    return httpx.MockTransport(handler), bodies


def _llm_settings(settings: Settings) -> Settings:
    from pydantic import SecretStr

    return settings.model_copy(update={"llm_api_key": SecretStr("k"), "llm_model": "test-model"})


def _sent_images(body: dict) -> list[str]:
    content = body["messages"][-1]["content"]
    return [part["image_url"]["url"] for part in content if part.get("type") == "image_url"]


def _decode(data_url: str):
    import base64
    import io

    import numpy as np
    from PIL import Image

    return np.array(Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))))


def _assert_manual_review(resp, bodies):
    assert bodies == [], "an image without readable text reached the hosted model"
    assert resp.check.status.value == "Needs review" and resp.document_type.value == "other"
    assert not any(resp.fields.model_dump().values()) and not resp.pipeline.used_vision_fallback


@pytest.mark.parametrize("make", [_blank_png, _blank_pdf], ids=["blank_png", "blank_pdf"])
def test_blank_upload_needs_review_without_llm_call(otel, settings, make):
    from app.pipeline import extract

    transport, bodies = _canned_llm()
    llm_settings = _llm_settings(settings)
    _assert_manual_review(extract(make(), llm_settings, transport), bodies)
    assert not any(llm_settings.upload_dir.glob("*")), "an upload was left on disk"
    summary = tel.last_request_summary()
    assert summary["check_status"] == "Needs review" and summary["outcome"] == "success"


def test_photo_with_only_junk_ocr_words_never_reaches_the_llm(otel, settings):
    """Real OCR on the holiday-photo sample, which Tesseract reads as a pile of low-confidence junk."""
    from app.ocr.tesseract import run_ocr_with_images
    from app.pipeline import extract
    from app.upload import store_upload

    data = (DATA / "samples" / "dev" / "dev_029_other.jpg").read_bytes()
    ocr, _ = run_ocr_with_images(store_upload(data, settings), settings)
    assert ocr.has_boxes, "fixture must be a photo where OCR invents words"
    transport, bodies = _canned_llm()
    _assert_manual_review(extract(data, _llm_settings(settings), transport), bodies)


def test_no_word_photo_never_reaches_the_llm(otel, settings, monkeypatch):
    import numpy as np

    from app import pipeline
    from app.schemas import OcrPage, OcrResult

    empty = OcrResult(words=[], text="", pages=[OcrPage(page=0, width=800, height=600)], mean_conf=0.0, char_count=0)
    photo = np.random.default_rng(1).integers(0, 255, (600, 800, 3), dtype=np.uint8)
    monkeypatch.setattr(pipeline, "run_ocr_with_images", lambda *_: (empty, [photo]))
    transport, bodies = _canned_llm()
    _assert_manual_review(pipeline.extract(_blank_png(), _llm_settings(settings), transport), bodies)


def test_vision_route_omits_a_photo_page_with_only_junk_words(otel, settings, monkeypatch):
    import numpy as np

    from app import pipeline
    from app.pii.masker import build_text_with_offsets
    from app.schemas import OcrPage, OcrResult, OcrWord

    def word(text, conf, page):
        return OcrWord(text=text, conf=conf, left=10, top=10, width=40, height=20, page=page)

    words = [word(t, 90.0, 0) for t in ("Marks", "Board", "Exam")] + [word("eee", 30.0, 1), word("ot", 95.0, 1)]
    pages = [OcrPage(page=i, width=800, height=600) for i in (0, 1)]
    text, _ = build_text_with_offsets(words)
    weak = OcrResult(words=words, text=text, pages=pages, mean_conf=50.0, char_count=14)
    images = [np.full((600, 800, 3), 255, dtype=np.uint8), np.zeros((600, 800, 3), dtype=np.uint8)]
    monkeypatch.setattr(pipeline, "run_ocr_with_images", lambda *_: (weak, images))
    transport, bodies = _canned_llm()
    resp = pipeline.extract(_blank_png(), _llm_settings(settings), transport)
    assert resp.pipeline.used_vision_fallback and len(_sent_images(bodies[0])) == 1


@pytest.mark.parametrize(
    "doc_type,percentage,status",
    [
        ("mark_sheet", 76.4, "Needs review"),
        ("student_id_card", None, "Not applicable"),
        ("fee_receipt", None, "Not applicable"),
        ("other", None, "Rejected"),
    ],
)
def test_weak_scan_takes_redacted_vision_route(otel, settings, doc_type, percentage, status):
    from app.ocr.tesseract import passes_text_gate, run_ocr_with_images
    from app.pipeline import extract
    from app.upload import store_upload

    data = _weak_scan()
    ocr, _ = run_ocr_with_images(store_upload(data, settings), settings)
    assert ocr.has_boxes and not passes_text_gate(ocr, settings), "fixture must be a weak scan"

    transport, bodies = _canned_llm(doc_type, percentage)
    resp = extract(data, _llm_settings(settings), transport)
    assert len(bodies) == 1, "expected exactly one LLM call"
    assert resp.pipeline.used_vision_fallback and resp.check.status.value == status
    assert resp.document_type.value == doc_type

    images = _sent_images(bodies[0])
    assert len(images) == 1
    assert "ravi" not in str(bodies[0]).lower() and "menon" not in str(bodies[0]).lower()
    sent = _decode(images[0])
    for word in (w for w in ocr.words if w.text in ("Ravi", "Menon")):
        assert (sent[word.top : word.top + word.height, word.left : word.left + word.width] == 0).all(), word.text


def test_vision_route_omits_pages_without_boxes(otel, settings):
    """A page OCR found nothing on cannot be redacted, so it must not leave the machine."""
    from app.ocr.tesseract import run_ocr_with_images
    from app.pipeline import extract
    from app.upload import store_upload

    data = _weak_then_blank_pdf()
    ocr, images = run_ocr_with_images(store_upload(data, settings), settings)
    assert len(images) == 2 and {w.page for w in ocr.words} == {0}, "fixture must have one page without boxes"

    transport, bodies = _canned_llm()
    resp = extract(data, _llm_settings(settings), transport)
    assert resp.pipeline.used_vision_fallback
    assert len(_sent_images(bodies[0])) == 1
