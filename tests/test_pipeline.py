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
