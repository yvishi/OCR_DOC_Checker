"""Headless tests of the Streamlit demo page with the HTTP calls mocked (no API, OCR or LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

from tests.test_api import EXPECTED_JSON

SCRIPT = str(Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py")
SAMPLE_LABEL = "Mark sheet 001 (PNG)"


class FakeHttp:
    def __init__(self, status: int = 200, body=EXPECTED_JSON, exc: Exception | None = None) -> None:
        self.status = status
        self.body = body
        self.exc = exc
        self.posts: list[dict] = []

    def post(self, url, files=None, timeout=None, **kwargs):
        self.posts.append({"url": url, "files": files, "timeout": timeout})
        if self.exc is not None:
            raise self.exc
        return httpx.Response(self.status, json=self.body, request=httpx.Request("POST", url))

    def get(self, url, timeout=None, **kwargs):
        summary = {"request_id": "abc", "stages": {"ocr": 640.0}, "outcome": "success", "check_status": "Eligible"}
        return httpx.Response(200, json=summary, request=httpx.Request("GET", url))


def _run_with(monkeypatch, fake: FakeHttp) -> AppTest:
    monkeypatch.setattr(httpx, "post", fake.post)
    monkeypatch.setattr(httpx, "get", fake.get)
    at = AppTest.from_file(SCRIPT, default_timeout=30)
    at.run()
    assert not at.exception
    at.sidebar.selectbox(key="sample_choice").select(SAMPLE_LABEL).run()
    at.button[0].click().run()
    assert not at.exception
    return at


def _all_markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_initial_page_renders(monkeypatch):
    at = AppTest.from_file(SCRIPT, default_timeout=30)
    at.run()
    assert not at.exception
    text = _all_markdown(at)
    assert "LearnSphere College · Admission Document Checker" in text
    assert "Synthetic demo data only" in text
    assert at.button[0].disabled  # nothing chosen yet
    options = at.sidebar.selectbox(key="sample_choice").options
    assert SAMPLE_LABEL in options


def test_sample_check_shows_status_fields_and_json(monkeypatch):
    fake = FakeHttp()
    at = _run_with(monkeypatch, fake)

    assert len(fake.posts) == 1
    post = fake.posts[0]
    assert post["url"].endswith("/v1/extract")
    assert post["timeout"] == 90.0
    sent_name, sent_bytes, _mime = post["files"]["file"]
    assert sent_name == "document.png"  # the sample's own filename is never sent
    assert sent_bytes.startswith(b"\x89PNG")

    text = _all_markdown(at)
    assert ">Eligible<" in text
    assert "Percentage is at least 60." in text
    assert "Mark sheet" in text
    assert "Masked before AI" in text

    assert len(at.json) == 1
    assert json.loads(at.json[0].value) == EXPECTED_JSON
    assert not at.error

    table = at.dataframe[0].value
    assert list(table["Value"]) == ["Aisha Rahman", "CBSE24-104482", "CBSE", "2024", "76.4%"]


def test_null_fields_show_dash(monkeypatch):
    body = json.loads(json.dumps(EXPECTED_JSON))
    body["fields"] = {k: None for k in body["fields"]}
    body["check"] = {"status": "Needs review", "reason": "Percentage could not be read."}
    at = _run_with(monkeypatch, FakeHttp(body=body))
    assert list(at.dataframe[0].value["Value"]) == ["—"] * 5
    assert ">Needs review<" in _all_markdown(at)


def test_subject_mismatch_shows_both_figures(monkeypatch):
    body = json.loads(json.dumps(EXPECTED_JSON))
    body["check"] = {
        "status": "Needs review",
        "reason": "The subject marks do not add up to the printed percentage.",
        "printed_percentage": 66.0,
        "recomputed_percentage": 63.2,
    }
    at = _run_with(monkeypatch, FakeHttp(body=body))
    shown = {m.label: m.value for m in at.metric}
    assert shown["Printed percentage"] == "66%"
    assert shown["Recomputed from subjects"] == "63.2%"


def test_figures_hidden_without_a_mismatch(monkeypatch):
    at = _run_with(monkeypatch, FakeHttp())
    assert "Printed percentage" not in {m.label for m in at.metric}


def test_api_error_shows_generic_message(monkeypatch):
    at = _run_with(monkeypatch, FakeHttp(status=422, body={"error": "Could not read this document"}))
    assert [e.value for e in at.error] == ["Could not read this document"]
    assert not at.json


def test_unreachable_api_shows_start_hint(monkeypatch):
    at = _run_with(monkeypatch, FakeHttp(exc=httpx.ConnectError("refused")))
    assert len(at.error) == 1
    assert "uvicorn app.main:app --port 8000" in at.error[0].value


def test_pdf_and_image_samples_show_a_preview():
    for label in ("Mark sheet 004 (PDF)", "Student ID card 013 (JPG)"):
        at = AppTest.from_file(SCRIPT, default_timeout=30)
        at.run()
        at.sidebar.selectbox(key="sample_choice").select(label).run()
        assert not at.exception
        assert len(at.get("image")) == 1, label
        assert not at.button[0].disabled


def test_sample_labels_are_readable():
    at = AppTest.from_file(SCRIPT, default_timeout=30)
    at.run()
    options = at.sidebar.selectbox(key="sample_choice").options
    for label in ("Mark sheet 004 (PDF)", "Student ID card 013 (JPG)", "Fee receipt 015 (PDF)", "Other document 018 (PNG)"):
        assert label in options
