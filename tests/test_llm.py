"""LLM client tests with a mocked HTTP transport: no network, no key needed."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.errors import GENERIC_MESSAGE, ErrorClass, PipelineError
from app.llm import client
from app.schemas import DocumentType

GOOD = {
    "document_type": "mark_sheet",
    "fields": {
        "student_name": "<PERSON_1>",
        "roll_number": "<ROLL_NUMBER_1>",
        "board_or_school": "CBSE",
        "passing_year": 2024,
        "percentage": 76.4,
    },
    "percentage_as_printed": "76.40%",
    "cgpa_only": False,
}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(client.time, "sleep", lambda s: None)


@pytest.fixture
def settings():
    return Settings(llm_api_key=SecretStr("sk-or-test"), llm_model="anthropic/claude-haiku-5.5")


def _reply(content, status=200, usage=None):
    body = {"model": "anthropic/claude-haiku-5.5", "choices": [{"message": {"content": content}}]}
    body["usage"] = usage or {"prompt_tokens": 120, "completion_tokens": 40}
    return httpx.Response(status, json=body)


class Recorder:
    """Mock transport that records requests and replays canned responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    @property
    def transport(self):
        return httpx.MockTransport(self)

    def body(self, i=0) -> dict:
        return json.loads(self.requests[i].content)


def test_valid_response_parsed(settings):
    rec = Recorder(_reply(json.dumps(GOOD)))
    result = client.extract_from_text("Name: <PERSON_1>\nPercentage: 76.40%", settings, rec.transport)
    assert result.output.document_type is DocumentType.MARK_SHEET
    assert result.output.fields.percentage == 76.4
    assert result.output.fields.student_name == "<PERSON_1>"
    assert (result.prompt_tokens, result.completion_tokens) == (120, 40)
    assert result.prompt_version == "extract_v1"


def test_request_shape_privacy_and_schema(settings):
    rec = Recorder(_reply(json.dumps(GOOD)))
    client.extract_from_text("masked text only", settings, rec.transport)
    req = rec.requests[0]
    assert str(req.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer sk-or-test"
    body = rec.body()
    assert body["provider"] == {"data_collection": "deny", "require_parameters": True}
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "temperature" not in body  # unsupported by the configured model; would break routing
    assert body["model"] == "anthropic/claude-haiku-5.5"
    assert body["messages"][-1]["content"].endswith("masked text only")


def test_fenced_json_tolerated(settings):
    rec = Recorder(_reply("```json\n" + json.dumps(GOOD) + "\n```"))
    assert client.extract_from_text("x", settings, rec.transport).output.fields.passing_year == 2024


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({**GOOD, "document_type": "passport"}),
        json.dumps({**GOOD, "fields": {**GOOD["fields"], "percentage": 140}}),
        json.dumps({**GOOD, "fields": {**GOOD["fields"], "passing_year": 24}}),
        json.dumps({**GOOD, "extra": "field"}),
        json.dumps({"document_type": "mark_sheet"}),
    ],
)
def test_invalid_output_is_schema_invalid(settings, content):
    rec = Recorder(_reply(content))
    with pytest.raises(PipelineError) as info:
        client.extract_from_text("x", settings, rec.transport)
    assert info.value.error_class is ErrorClass.SCHEMA_INVALID
    assert str(info.value) == GENERIC_MESSAGE


def test_retry_once_on_429_then_succeed(settings):
    rec = Recorder(httpx.Response(429, json={"error": {}}), _reply(json.dumps(GOOD)))
    assert client.extract_from_text("x", settings, rec.transport).output.document_type is DocumentType.MARK_SHEET
    assert len(rec.requests) == 2


@pytest.mark.parametrize(
    "responses",
    [
        (httpx.Response(401, json={"error": {"message": "bad key"}}),),
        (httpx.Response(404, json={"error": {"message": "No endpoints found"}}),),
        (httpx.Response(503, json={}), httpx.Response(503, json={})),
        (httpx.ConnectError("boom"), httpx.ConnectError("boom")),
        (httpx.Response(200, text="<html>"),),
        (httpx.Response(200, json={"choices": []}),),
    ],
)
def test_transport_and_http_failures_are_llm_failed(settings, responses):
    rec = Recorder(*responses)
    with pytest.raises(PipelineError) as info:
        client.extract_from_text("x", settings, rec.transport)
    assert info.value.error_class is ErrorClass.LLM_FAILED
    assert str(info.value) == GENERIC_MESSAGE
    assert info.value.__cause__ is None  # nothing from the response body is chained


def test_missing_key_or_model_fails_without_network(settings):
    rec = Recorder()
    for s in (Settings(llm_api_key=SecretStr(""), llm_model="m"), Settings(llm_api_key=SecretStr("k"), llm_model="")):
        with pytest.raises(PipelineError) as info:
            client.extract_from_text("x", s, rec.transport)
        assert info.value.error_class is ErrorClass.LLM_FAILED
    assert rec.requests == []


def test_images_sent_as_downscaled_png_data_urls(settings):
    rec = Recorder(_reply(json.dumps(GOOD)))
    big = np.full((3200, 2400), 255, dtype=np.uint8)
    client.extract_from_images([big, big], settings, rec.transport)
    parts = rec.body()["messages"][-1]["content"]
    urls = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
    assert len(urls) == 2 and all(u.startswith("data:image/png;base64,") for u in urls)
    import base64

    import cv2

    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(urls[0].split(",", 1)[1]), np.uint8), cv2.IMREAD_UNCHANGED)
    assert max(decoded.shape[:2]) == client.MAX_IMAGE_SIDE


def test_no_images_never_calls_model(settings):
    rec = Recorder()
    with pytest.raises(PipelineError) as info:
        client.extract_from_images([], settings, rec.transport)
    assert info.value.error_class is ErrorClass.OCR_EMPTY
    assert rec.requests == []
