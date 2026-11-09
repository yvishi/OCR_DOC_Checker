"""API tests: the HTTP layer only. `pipeline.extract` is replaced by a fake, so no OCR or LLM runs."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import get_settings
from app.errors import GENERIC_MESSAGE, ErrorClass, PipelineError
from app.schemas import (
    Check,
    CheckStatus,
    DocumentType,
    ExtractResponse,
    LearnSphereFields,
    PiiSummary,
    PipelineSummary,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
SECRET_FILENAME = "Aisha_Rahman_CBSE24-104482_marksheet.png"

CANNED = ExtractResponse(
    request_id="7c1e0a4e3d2b4f0a9c112a9b0e6d4f18",
    document_type=DocumentType.MARK_SHEET,
    fields=LearnSphereFields(
        student_name="Aisha Rahman",
        roll_number="CBSE24-104482",
        board_or_school="CBSE",
        passing_year=2024,
        percentage=76.4,
    ),
    check=Check(status=CheckStatus.ELIGIBLE, reason="Percentage is at least 60."),
    pii=PiiSummary(masked_before_llm=True, entity_counts={"PERSON": 1, "IN_PHONE": 1}),
    pipeline=PipelineSummary(
        ocr_engine="tesseract",
        used_vision_fallback=False,
        timings_ms={"ocr": 640.0, "pii": 38.0, "llm": 1102.0, "rule": 3.0},
    ),
)

EXPECTED_JSON = {
    "request_id": "7c1e0a4e3d2b4f0a9c112a9b0e6d4f18",
    "document_type": "mark_sheet",
    "fields": {
        "student_name": "Aisha Rahman",
        "roll_number": "CBSE24-104482",
        "board_or_school": "CBSE",
        "passing_year": 2024,
        "percentage": 76.4,
    },
    "check": {"status": "Eligible", "reason": "Percentage is at least 60."},
    "pii": {"masked_before_llm": True, "entity_counts": {"PERSON": 1, "IN_PHONE": 1}},
    "pipeline": {
        "ocr_engine": "tesseract",
        "used_vision_fallback": False,
        "timings_ms": {"ocr": 640.0, "pii": 38.0, "llm": 1102.0, "rule": 3.0},
    },
}


class FakeExtract:
    def __init__(self, result=CANNED, exc: BaseException | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.exc is not None:
            raise self.exc
        return self.result


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def fake(monkeypatch) -> FakeExtract:
    f = FakeExtract()
    monkeypatch.setattr(main.pipeline, "extract", f)
    return f


def _post(client: TestClient, data: bytes = PNG_BYTES, filename: str = SECRET_FILENAME):
    return client.post("/v1/extract", files={"file": (filename, data, "image/png")})


def _assert_generic(resp) -> None:
    assert resp.json() == {"error": GENERIC_MESSAGE}
    body = resp.text.lower()
    for ec in ErrorClass:
        assert ec.value not in body
    assert "traceback" not in body and "exception" not in body


# ---------- happy path ----------


def test_extract_returns_exact_contract_json(client, fake):
    resp = _post(client)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == EXPECTED_JSON


def test_fake_receives_only_the_bytes(client, fake, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    resp = _post(client)
    assert resp.status_code == 200
    assert len(fake.calls) == 1
    args, kwargs = fake.calls[0]
    assert args == (PNG_BYTES,)
    assert kwargs == {}
    # The filename appears nowhere: not in the response, logs or console output.
    captured = capsys.readouterr()
    for text in (resp.text, caplog.text, captured.out, captured.err):
        assert SECRET_FILENAME not in text
        assert "Aisha_Rahman" not in text


# ---------- size cap ----------


def test_oversize_upload_is_413_and_pipeline_not_called(client, fake):
    cap = get_settings().max_upload_bytes
    resp = _post(client, data=b"\x89PNG\r\n\x1a\n" + b"\x00" * (cap - 7))  # cap + 1 bytes
    assert resp.status_code == 413
    _assert_generic(resp)
    assert fake.calls == []


def test_declared_oversize_body_is_rejected_before_parsing(client, fake):
    cap = get_settings().max_upload_bytes
    resp = _post(client, data=b"\x00" * (cap + main.MULTIPART_OVERHEAD_BYTES + 1))
    assert resp.status_code == 413
    _assert_generic(resp)
    assert fake.calls == []


def test_upload_exactly_at_cap_is_accepted(client, fake):
    cap = get_settings().max_upload_bytes
    data = b"\x89PNG\r\n\x1a\n" + b"\x00" * (cap - 8)
    resp = _post(client, data=data)
    assert resp.status_code == 200
    assert fake.calls[0][0] == (data,)


# ---------- errors ----------


@pytest.mark.parametrize("error_class,status", list(main.STATUS_BY_ERROR_CLASS.items()))
def test_pipeline_error_maps_to_status_with_generic_body(client, monkeypatch, error_class, status):
    monkeypatch.setattr(main.pipeline, "extract", FakeExtract(exc=PipelineError(error_class)))
    resp = _post(client)
    assert resp.status_code == status
    _assert_generic(resp)


def test_every_error_class_has_a_status():
    assert set(main.STATUS_BY_ERROR_CLASS) == set(ErrorClass)
    assert main.STATUS_BY_ERROR_CLASS[ErrorClass.UPLOAD_TOO_LARGE] == 413
    assert main.STATUS_BY_ERROR_CLASS[ErrorClass.UPLOAD_BAD_TYPE] == 415
    assert main.STATUS_BY_ERROR_CLASS[ErrorClass.UPLOAD_EMPTY] == 400
    assert main.STATUS_BY_ERROR_CLASS[ErrorClass.OCR_EMPTY] == 422
    assert main.STATUS_BY_ERROR_CLASS[ErrorClass.LLM_FAILED] == 502
    assert main.STATUS_BY_ERROR_CLASS[ErrorClass.SCHEMA_INVALID] == 502


def test_blank_page_is_422_generic_through_the_real_pipeline(client, monkeypatch):
    import io

    from PIL import Image

    def no_llm(*_args, **_kwargs):
        raise AssertionError("the hosted model must not be called without word boxes")

    monkeypatch.setattr(main.pipeline, "_call_llm", no_llm)
    buf = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(buf, "PNG")
    resp = _post(client, buf.getvalue())
    assert resp.status_code == 422
    _assert_generic(resp)


def test_unexpected_exception_is_500_generic(client, monkeypatch):
    leaky = RuntimeError("Aisha Rahman CBSE24-104482 ocr_failed")
    monkeypatch.setattr(main.pipeline, "extract", FakeExtract(exc=leaky))
    resp = _post(client)
    assert resp.status_code == 500
    _assert_generic(resp)
    assert "Aisha" not in resp.text and "CBSE24" not in resp.text


def test_missing_file_field_is_422_generic(client, fake):
    resp = client.post("/v1/extract", data={"not_file": "Aisha Rahman"})
    assert resp.status_code == 422
    _assert_generic(resp)
    assert "Aisha" not in resp.text
    assert fake.calls == []


def test_unknown_route_has_no_detail(client):
    resp = client.get("/v1/nope")
    assert resp.status_code == 404
    assert resp.json() == {"error": GENERIC_MESSAGE}


# ---------- other endpoints ----------


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_telemetry_last_returns_summary(client, monkeypatch):
    summary = {"request_id": "abc", "stages": {"ocr": 1.0}, "outcome": "success"}
    monkeypatch.setattr(main.telemetry, "last_request_summary", lambda: summary)
    resp = client.get("/v1/telemetry/last")
    assert resp.status_code == 200
    assert resp.json() == summary


# ---------- CORS ----------


def test_cors_allows_streamlit_origin(client):
    resp = client.options(
        "/v1/extract",
        headers={"Origin": "http://localhost:8501", "Access-Control-Request-Method": "POST"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:8501"
    resp = client.get("/healthz", headers={"Origin": "http://localhost:8501"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:8501"


def test_cors_rejects_unknown_origin(client):
    resp = client.get("/healthz", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in resp.headers
    resp = client.options(
        "/v1/extract",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in resp.headers


def test_allowed_origins_parsing():
    assert main.allowed_origins("") == ["http://localhost:8501"]
    assert main.allowed_origins("https://a.example, https://b.example/ ,*") == [
        "http://localhost:8501",
        "https://a.example",
        "https://b.example",
    ]
