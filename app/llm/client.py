"""The only outbound network call: one OpenRouter chat completion on masked input.

Every request sets provider.data_collection="deny" (only providers that don't store or train on
prompts) and require_parameters=true (only providers that honour the JSON schema). Errors never
carry response bodies or document text: callers see PipelineError(LLM_FAILED | SCHEMA_INVALID).
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass

import cv2
import httpx
import numpy as np
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.errors import ErrorClass, PipelineError
from app.llm import prompt
from app.schemas import LLMOutput

TIMEOUT_S = 45.0
MAX_ATTEMPTS = 2  # one retry on 429 / 5xx / transport errors
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
MAX_IMAGE_SIDE = 1600  # keep vision payloads small; redaction already happened at full size


@dataclass(frozen=True)
class LLMResult:
    output: LLMOutput
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    prompt_version: str = prompt.PROMPT_VERSION


def _png_data_url(img: np.ndarray) -> str:
    h, w = img.shape[:2]
    scale = min(1.0, MAX_IMAGE_SIDE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise PipelineError(ErrorClass.LLM_FAILED)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def build_payload(messages: list[dict], settings: Settings) -> dict:
    return {
        "model": settings.llm_model,
        "messages": messages,
        # No temperature: newer models (e.g. claude-haiku-5.5) don't accept sampling parameters, and
        # require_parameters=true would then match no provider at all (HTTP 404).
        # Reasoning tokens count against max_tokens; at 400 a sheet whose subject marks disagreed
        # with its percentage made the model think until the JSON was cut off (finish_reason=length).
        "max_tokens": 2000,
        "reasoning": {"effort": "low"},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "learnsphere_extract", "strict": True, "schema": prompt.RESPONSE_SCHEMA},
        },
        "provider": {"data_collection": "deny", "require_parameters": True},
    }


def _post(payload: dict, settings: Settings, transport: httpx.BaseTransport | None) -> dict:
    key = settings.llm_api_key.get_secret_value()
    if not key or not settings.llm_model:
        raise PipelineError(ErrorClass.LLM_FAILED)
    headers = {"Authorization": f"Bearer {key}", "X-Title": "OCR_DOC_Checker"}
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    with httpx.Client(timeout=TIMEOUT_S, transport=transport) as client:
        for attempt in range(MAX_ATTEMPTS):
            try:
                resp = client.post(url, json=payload, headers=headers)
            except httpx.HTTPError:
                if attempt + 1 < MAX_ATTEMPTS:
                    time.sleep(1.0)
                    continue
                raise PipelineError(ErrorClass.LLM_FAILED) from None
            if resp.status_code in RETRY_STATUS and attempt + 1 < MAX_ATTEMPTS:
                time.sleep(1.0)
                continue
            if resp.status_code != 200:
                raise PipelineError(ErrorClass.LLM_FAILED)
            try:
                return resp.json()
            except ValueError:
                raise PipelineError(ErrorClass.LLM_FAILED) from None
    raise PipelineError(ErrorClass.LLM_FAILED)


def _parse(body: dict) -> LLMOutput:
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise PipelineError(ErrorClass.LLM_FAILED) from None
    if not isinstance(content, str):
        raise PipelineError(ErrorClass.SCHEMA_INVALID)
    text = content.strip()
    if text.startswith("```"):  # tolerate a fenced block from models that add one
        text = text.strip("`").removeprefix("json").strip()
    try:
        return LLMOutput.model_validate(json.loads(text))
    except (ValueError, ValidationError):
        raise PipelineError(ErrorClass.SCHEMA_INVALID) from None


def _call(messages: list[dict], settings: Settings | None, transport: httpx.BaseTransport | None) -> LLMResult:
    settings = settings or get_settings()
    body = _post(build_payload(messages, settings), settings, transport)
    output = _parse(body)
    usage = body.get("usage") or {}
    return LLMResult(
        output=output,
        model_name=str(body.get("model") or settings.llm_model),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )


def extract_from_text(
    masked_text: str, settings: Settings | None = None, transport: httpx.BaseTransport | None = None
) -> LLMResult:
    return _call(prompt.text_messages(masked_text), settings, transport)


def extract_from_images(
    redacted_images: list[np.ndarray], settings: Settings | None = None, transport: httpx.BaseTransport | None = None
) -> LLMResult:
    if not redacted_images:
        raise PipelineError(ErrorClass.OCR_EMPTY)
    return _call(prompt.image_messages([_png_data_url(img) for img in redacted_images]), settings, transport)
