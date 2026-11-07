from __future__ import annotations

import json
import logging
import uuid

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from app import telemetry as tel
from app.errors import ErrorClass, PipelineError
from tests.telemetry_helpers import assert_no_leak, dump_telemetry, read_jsonl

SECRETS = ["Aisha Rahman", "CBSE24-104482", "2345 6789 0123", "aisha.rahman@example.com"]


@pytest.fixture
def otel(tmp_path):
    spans = InMemorySpanExporter()
    reader = InMemoryMetricReader()
    jsonl = tmp_path / "telemetry.jsonl"
    tel.reset_telemetry()
    tel.setup_telemetry(exporter=spans, jsonl_path=jsonl, metric_reader=reader)
    yield spans, reader, jsonl
    tel.reset_telemetry()


def metric_points(reader: InMemoryMetricReader) -> dict[str, list]:
    out: dict[str, list] = {}
    data = reader.get_metrics_data()
    if data is None:
        return out
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                out.setdefault(m.name, []).extend(m.data.data_points)
    return out


def rejected_total(reader) -> int:
    return sum(p.value for p in metric_points(reader).get("telemetry.rejected_attributes", []))


# ---------- safe_attributes ----------


def test_allowed_attributes_pass(otel):
    rid = uuid.uuid4()
    attrs = {
        "request_id": rid.hex,
        "document_type": "mark_sheet",
        "page_count": 2,
        "file_extension": "pdf",
        "ocr_engine": "tesseract",
        "char_count": 812,
        "mean_conf": 91.5,
        "used_vision_fallback": False,
        "entity_counts.PERSON": 1,
        "entity_counts.PHONE_NUMBER": 2,
        "placeholder_count": 3,
        "model_name": "gpt-4o-mini",
        "prompt_tokens": 512,
        "completion_tokens": 88,
        "prompt_version": "v1",
        "duration_ms": 12.5,
        "stage": "ocr",
        "check_status": "Needs review",
        "error_class": "ocr_empty",
        "outcome": "success",
        "rotation_applied": 90,
        "stamp_pixels_removed": 1200,
        "low_conf_word_count": 4,
    }
    assert tel.safe_attributes(attrs) == attrs
    assert tel.safe_attributes({"request_id": str(rid)}) == {"request_id": str(rid)}
    assert rejected_total(otel[1]) == 0


def test_entity_counts_dict_is_flattened(otel):
    assert tel.safe_attributes({"entity_counts": {"PERSON": 1, "AADHAAR": 2}}) == {
        "entity_counts.PERSON": 1,
        "entity_counts.AADHAAR": 2,
    }


@pytest.mark.parametrize("secret", SECRETS)
@pytest.mark.parametrize(
    "key",
    ["student_name", "text", "check_status", "document_type", "stage", "error_class", "outcome",
     "model_name", "prompt_version", "request_id", "ocr_engine", "file_extension", "page_count"],
)
def test_pii_looking_values_dropped_in_any_key(otel, key, secret):
    assert tel.safe_attributes({key: secret}) == {}


@pytest.mark.parametrize("secret", SECRETS)
def test_pii_as_key_dropped(otel, secret, caplog):
    caplog.set_level(logging.DEBUG, logger="ocr_doc_checker.telemetry")
    assert tel.safe_attributes({secret: 1, "entity_counts": {secret: 1}}) == {}
    assert secret not in caplog.text
    assert "<unsafe-key>" in caplog.text


@pytest.mark.parametrize(
    "attrs",
    [
        {"request_id": "not-a-uuid"},
        {"request_id": "CBSE24-104482"},
        {"request_id": 12345},
        {"check_status": "eligible"},
        {"check_status": "Pass"},
        {"error_class": "ValueError"},
        {"document_type": "passport"},
        {"page_count": -1},
        {"page_count": True},
        {"page_count": "2"},
        {"mean_conf": 150.0},
        {"mean_conf": float("nan")},
        {"used_vision_fallback": "yes"},
        {"entity_counts.person": 1},
        {"entity_counts.PERSON-1": 1},
        {"entity_counts.Aisha": 1},
        {"entity_counts.PERSON": "1"},
        {"entity_counts": {"person": 1, "PHONE NUMBER": 2}},
        {"model_name": "aisha"},
        {"prompt_version": "Aisha Rahman"},
        {"rotation_applied": 360},
        {"filename": "aisha_marksheet.pdf"},
    ],
)
def test_bad_values_dropped_and_counted(otel, attrs):
    assert tel.safe_attributes(attrs) == {}
    assert rejected_total(otel[1]) >= 1


def test_rejection_log_names_only_safe_keys(otel, caplog):
    caplog.set_level(logging.DEBUG, logger="ocr_doc_checker.telemetry")
    tel.safe_attributes({"check_status": "Aisha Rahman", "student_name": "x"})
    assert "check_status" in caplog.text
    assert "student_name" not in caplog.text  # unknown keys are never named
    assert "Aisha Rahman" not in caplog.text


# ---------- spans ----------


def test_stage_span_records_duration_and_allowed_attrs(otel):
    spans, _, _ = otel
    with tel.stage_span("ocr", char_count=120, student_name="Aisha Rahman") as s:
        s.set(mean_conf=88.0, roll="CBSE24-104482")
    (span,) = spans.get_finished_spans()
    assert span.name == "ocr"
    assert span.attributes["stage"] == "ocr"
    assert span.attributes["char_count"] == 120
    assert span.attributes["mean_conf"] == 88.0
    assert span.attributes["duration_ms"] >= 0
    assert "student_name" not in span.attributes and "roll" not in span.attributes
    assert span.status.status_code is not StatusCode.ERROR
    assert_no_leak(spans.get_finished_spans(), [], SECRETS)


def test_unknown_span_name_rejected(otel):
    with pytest.raises(ValueError):
        with tel.stage_span("Aisha Rahman"):
            pass


@pytest.mark.parametrize("sanitize", [True, False])
def test_exception_in_span_never_leaks(tmp_path, sanitize):
    spans = InMemorySpanExporter()
    tel.reset_telemetry()
    tel.setup_telemetry(exporter=spans, jsonl_path=tmp_path / "t.jsonl",
                        metric_reader=InMemoryMetricReader(), sanitize=sanitize)
    try:
        with pytest.raises(ValueError, match="Aisha Rahman"):
            with tel.stage_span("request"):
                with tel.stage_span("pii"):
                    raise ValueError("Aisha Rahman CBSE24-104482")
        finished = spans.get_finished_spans()
        assert {s.name for s in finished} == {"request", "pii"}
        for s in finished:
            assert s.status.status_code is StatusCode.ERROR
            assert s.status.description == tel.GENERIC_ERROR_DESCRIPTION
            assert len(s.events) == 0  # record_exception=False
            assert s.attributes["error_class"] == "internal"
        dump = dump_telemetry(finished)
        assert "Aisha" not in dump and "CBSE24-104482" not in dump
        assert "ValueError" not in dump
        assert_no_leak(finished, read_jsonl(tmp_path / "t.jsonl"), SECRETS)
    finally:
        tel.reset_telemetry()


def test_pipeline_error_sets_error_class(otel):
    spans, _, _ = otel
    with pytest.raises(PipelineError):
        with tel.stage_span("llm", model_name="gpt-4o-mini"):
            raise PipelineError(ErrorClass.SCHEMA_INVALID)
    (span,) = spans.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error_class"] == "schema_invalid"
    assert span.attributes["model_name"] == "gpt-4o-mini"
    assert not span.events


def test_sanitizing_exporter_filters_foreign_spans(otel):
    """Spans created directly on the provider (e.g. by the FastAPI instrumentor) are still allowlisted."""
    spans, _, _ = otel
    tracer = tel.get_tracer_provider().get_tracer("foreign")
    with tracer.start_as_current_span("POST /v1/extract") as s:
        s.set_attribute("http.route", "/v1/extract")
        s.set_attribute("http.status_code", 200)
        s.set_attribute("http.url", "http://x/v1/extract?name=Aisha%20Rahman")
        s.add_event("note", {"student": "Aisha Rahman"})
    with tracer.start_as_current_span("Aisha Rahman"):
        pass
    a, b = spans.get_finished_spans()
    assert a.name == "POST /v1/extract"
    assert dict(a.attributes) == {"http.route": "/v1/extract", "http.status_code": 200}
    assert not a.events
    assert b.name == "redacted"
    assert_no_leak(spans.get_finished_spans(), [], SECRETS)


# ---------- log_event ----------


def test_log_event_writes_only_allowed_keys(otel):
    spans, _, jsonl = otel
    rid = uuid.uuid4().hex
    with tel.stage_span("request", request_id=rid):
        tel.log_event("stage_completed", stage="ocr", duration_ms=5.0, char_count=40,
                      ocr_text="Aisha Rahman CBSE24-104482", email="aisha.rahman@example.com")
    tel.log_event("request_completed", request_id=rid, outcome="success", check_status="Eligible",
                  entity_counts={"PERSON": 1, "aadhaar": 1}, document_type="2345 6789 0123")
    assert tel.log_event("Aisha Rahman", outcome="success") is None

    lines = read_jsonl(jsonl)
    assert len(lines) == 2
    allowed = set(tel._VALIDATORS) | {"ts", "event", "trace_id", "span_id"}
    for line in lines:
        rec = json.loads(line)
        for key in rec:
            assert key in allowed or (key.startswith("entity_counts.") and key[14:].isupper()), key
    first, second = (json.loads(l) for l in lines)
    assert first["event"] == "stage_completed" and first["char_count"] == 40 and "trace_id" in first
    assert second["entity_counts.PERSON"] == 1 and "document_type" not in second
    assert_no_leak(spans.get_finished_spans(), lines, SECRETS)


# ---------- metrics ----------


def test_metrics_recorded_with_expected_attributes(otel):
    _, reader, _ = otel
    tel.record_request("success")
    tel.record_request("Aisha Rahman")
    tel.record_stage_latency("ocr", 42.0)
    tel.record_entities({"PERSON": 2, "IN_AADHAAR": 1, "Aisha Rahman": 3, "PHONE_NUMBER": -1})
    tel.record_check("Eligible")
    tel.record_check("Needs review")
    tel.record_fallback()
    pts = metric_points(reader)

    req = {tuple(sorted(p.attributes.items())): p.value for p in pts["requests_total"]}
    assert req[(("outcome", "success"),)] == 1
    assert req[()] == 1  # the bad outcome is counted without a label
    lat = [p for p in pts["stage_latency_ms"] if dict(p.attributes) == {"stage": "ocr"}]
    assert lat and lat[0].sum == 42.0 and lat[0].count == 1
    ents = {p.attributes["entity_type"]: p.value for p in pts["pii_entities_total"]}
    assert ents == {"PERSON": 2, "IN_AADHAAR": 1}
    checks = {p.attributes["check_status"]: p.value for p in pts["check_results_total"]}
    assert checks == {"Eligible": 1, "Needs review": 1}
    assert sum(p.value for p in pts["vision_fallback_total"]) == 1
    assert rejected_total(reader) >= 3

    dumped = json.dumps({k: [dict(p.attributes) for p in v] for k, v in pts.items()})
    assert "Aisha" not in dumped


def test_stage_spans_feed_latency_histogram(otel):
    _, reader, _ = otel
    with tel.stage_span("rule"):
        pass
    pts = metric_points(reader)["stage_latency_ms"]
    assert any(dict(p.attributes) == {"stage": "rule"} and p.count == 1 for p in pts)


# ---------- last_request_summary ----------


def test_last_request_summary(otel):
    assert tel.last_request_summary() is None
    rid = uuid.uuid4().hex
    with tel.request_span(request_id=rid, document_type="mark_sheet", student="Aisha Rahman") as req:
        for stage in ("ocr", "pii", "llm", "rule"):
            with tel.stage_span(stage):
                pass
        tel.record_entities({"PERSON": 1, "PHONE_NUMBER": 1})
        tel.record_check("Eligible")
        req.set(outcome="success")
    summary = tel.last_request_summary()
    assert summary["request_id"] == rid
    assert summary["document_type"] == "mark_sheet"
    assert set(summary["stages"]) == {"ocr", "pii", "llm", "rule", "request"}
    assert all(v >= 0 for v in summary["stages"].values())
    assert summary["check_status"] == "Eligible"
    assert summary["outcome"] == "success"
    assert summary["entity_counts"] == {"PERSON": 1, "PHONE_NUMBER": 1}
    assert "Aisha" not in json.dumps(summary)


def test_last_request_summary_on_failure(otel):
    with pytest.raises(PipelineError):
        with tel.request_span(request_id=str(uuid.uuid4())):
            with tel.stage_span("ocr"):
                raise PipelineError(ErrorClass.OCR_EMPTY)
    summary = tel.last_request_summary()
    assert summary["error_class"] == "ocr_empty"
    assert summary["outcome"] == "error"
    assert "ocr" in summary["stages"]


# ---------- setup ----------


def test_setup_is_idempotent(otel):
    provider = tel.get_tracer_provider()
    tel.setup_telemetry(exporter="console")
    assert tel.get_tracer_provider() is provider


def test_jsonl_mode_writes_spans(tmp_path):
    path = tmp_path / "spans.jsonl"
    tel.reset_telemetry()
    tel.setup_telemetry(exporter="jsonl", jsonl_path=path, metric_reader=InMemoryMetricReader())
    try:
        with tel.stage_span("ocr", char_count=5, text="Aisha Rahman"):
            pass
        tel.log_event("stage_completed", stage="ocr")
        lines = read_jsonl(path)
        span_rec = json.loads(lines[0])
        assert span_rec["type"] == "span" and span_rec["name"] == "ocr"
        assert span_rec["attributes"]["char_count"] == 5
        assert json.loads(lines[1])["event"] == "stage_completed"
        assert_no_leak([], lines, SECRETS)
    finally:
        tel.reset_telemetry()


def test_assert_no_leak_detects_secret_case_and_whitespace_insensitive():
    with pytest.raises(AssertionError):
        assert_no_leak([], ['{"x": "aisha   RAHMAN"}'], ["Aisha Rahman"])
    with pytest.raises(AssertionError):
        assert_no_leak([], [{"x": "2345\n6789 0123"}], ["2345 6789 0123"])
    assert_no_leak([], ['{"event": "request_completed"}'], ["Aisha Rahman"])
