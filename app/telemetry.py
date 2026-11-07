"""Tracing, metrics and structured logs with a privacy allowlist (docs/ARCHITECTURE.md §7).

Every attribute that reaches a span, a metric or a log line goes through `safe_attributes`, which
accepts only keys from a fixed allowlist and values that match a per-key rule. Anything else is
dropped and counted in `telemetry.rejected_attributes`; its value is never logged.

The module keeps its own TracerProvider / MeterProvider objects instead of setting the OTel
globals, so tests can rebuild them freely (`reset_telemetry()` + `setup_telemetry(...)`). For the
FastAPI instrumentor pass `tracer_provider=get_tracer_provider()`; spans from it are still passed
through the sanitizing exporter, which re-applies the allowlist at export time.

Environment:
    TELEMETRY_EXPORTER   console (default) | none | jsonl
    TELEMETRY_JSONL_PATH path of the JSON-lines file (default <project root>/telemetry.jsonl);
                         setting it also enables the JSONL writer in console mode.
"""

from __future__ import annotations

import contextvars
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Status, StatusCode

from app.config import PROJECT_ROOT
from app.errors import ErrorClass, PipelineError
from app.schemas import DocumentType

logger = logging.getLogger("ocr_doc_checker.telemetry")

SERVICE_NAME = "ocr_doc_checker"
DEFAULT_JSONL_PATH = PROJECT_ROOT / "telemetry.jsonl"
GENERIC_ERROR_DESCRIPTION = "stage failed"
UNSAFE_KEY = "<unsafe-key>"

# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

STAGES = frozenset({"request", "upload", "ocr", "pii", "llm", "rule", "vision"})
SPAN_NAMES = frozenset({"request", "ocr", "pii", "llm", "rule", "vision"})
EVENTS = frozenset({"request_completed", "request_failed", "stage_completed"})
CHECK_STATUSES = frozenset({"Eligible", "Not eligible", "Needs review", "Not applicable", "Rejected"})
OUTCOMES = frozenset({"success", "error", "needs_review", "rejected"})
DOCUMENT_TYPES = frozenset(d.value for d in DocumentType)
ERROR_CLASSES = frozenset(e.value for e in ErrorClass)
FILE_EXTENSIONS = frozenset({"pdf", "png", "jpg", "jpeg"})
OCR_ENGINES = frozenset({"tesseract", "vision", "none"})
MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4", "claude-", "gemini-", "llama", "mistral", "mixtral", "qwen", "deepseek")
HTTP_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})

_MAX_STR = 64
_MAX_INT = 10**9
_UUID_RE = re.compile(r"^(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$")
_ENTITY_TYPE_RE = re.compile(r"^[A-Z][A-Z_]{0,39}$")
_ENTITY_KEY_PREFIX = "entity_counts."
_MODEL_RE = re.compile(r"^[a-z0-9][a-z0-9._:/-]{1,63}$")
_PROMPT_VERSION_RE = re.compile(r"^(?:[a-z][a-z_]{0,23}[-_.])?v\d{1,3}(?:\.\d{1,3}){0,2}$")
_HTTP_ROUTE_RE = re.compile(r"^/[A-Za-z0-9_/{}.\-]{0,63}$")
_HTTP_SPAN_NAME_RE = re.compile(
    r"^(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS) /[A-Za-z0-9_/{}.\-]{0,63}(?: http (?:send|receive))?$"
)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _count(v: Any) -> bool:
    return _is_int(v) and 0 <= v <= _MAX_INT


def _non_neg_number(v: Any) -> bool:
    if _is_int(v):
        return 0 <= v <= _MAX_INT
    return isinstance(v, float) and math.isfinite(v) and 0.0 <= v <= _MAX_INT


def _in(values: frozenset[str]):
    return lambda v: isinstance(v, str) and v in values


def _model_name(v: Any) -> bool:
    if not isinstance(v, str) or not _MODEL_RE.match(v):
        return False
    if v.startswith(MODEL_PREFIXES):
        return True
    try:  # the configured model is always acceptable
        from app.config import get_settings

        return bool(get_settings().llm_model) and v == get_settings().llm_model
    except Exception:  # noqa: BLE001 - settings problems must never break telemetry
        return False


_VALIDATORS: dict[str, Any] = {
    "request_id": lambda v: isinstance(v, str) and bool(_UUID_RE.match(v)),
    "document_type": _in(DOCUMENT_TYPES),
    "page_count": _count,
    "file_extension": _in(FILE_EXTENSIONS),
    "ocr_engine": _in(OCR_ENGINES),
    "char_count": _count,
    "mean_conf": lambda v: _non_neg_number(v) and v <= 100,
    "used_vision_fallback": lambda v: isinstance(v, bool),
    "placeholder_count": _count,
    "model_name": _model_name,
    "prompt_tokens": _count,
    "completion_tokens": _count,
    "prompt_version": lambda v: isinstance(v, str) and bool(_PROMPT_VERSION_RE.match(v)),
    "duration_ms": _non_neg_number,
    "stage": _in(STAGES),
    "check_status": _in(CHECK_STATUSES),
    "error_class": _in(ERROR_CLASSES),
    "outcome": _in(OUTCOMES),
    "rotation_applied": lambda v: _is_int(v) and 0 <= v < 360,
    "stamp_pixels_removed": _count,
    "low_conf_word_count": _count,
    # metric label for pii_entities_total
    "entity_type": lambda v: isinstance(v, str) and bool(_ENTITY_TYPE_RE.match(v)),
}

# Extra keys accepted only at export time, for spans made by the FastAPI instrumentation.
_HTTP_VALIDATORS: dict[str, Any] = {
    "http.method": _in(HTTP_METHODS),
    "http.request.method": _in(HTTP_METHODS),
    "http.route": lambda v: isinstance(v, str) and bool(_HTTP_ROUTE_RE.match(v)),
    "http.status_code": lambda v: _is_int(v) and 100 <= v <= 599,
    "http.response.status_code": lambda v: _is_int(v) and 100 <= v <= 599,
}

ALLOWED_KEYS = frozenset(_VALIDATORS) | frozenset({_ENTITY_KEY_PREFIX + "<TYPE>"})


def _validator_for(key: Any, extra: Mapping[str, Any] | None = None):
    if not isinstance(key, str):
        return None
    if key in _VALIDATORS:
        return _VALIDATORS[key]
    if extra and key in extra:
        return extra[key]
    if key.startswith(_ENTITY_KEY_PREFIX) and _ENTITY_TYPE_RE.match(key[len(_ENTITY_KEY_PREFIX) :]):
        return _count
    return None


def _reject(key: Any, reason: str) -> None:
    safe_key = key if _validator_for(key, _HTTP_VALIDATORS) is not None else UNSAFE_KEY
    logger.debug("telemetry: rejected attribute %s (%s)", safe_key, reason)
    counter = _STATE.rejected_counter
    if counter is not None:
        try:
            counter.add(1, {"reason": reason})
        except Exception:  # noqa: BLE001
            pass


def _filter(attrs: Mapping[Any, Any] | None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not attrs:
        return out
    for key, value in attrs.items():
        # convenience: entity_counts={"PERSON": 1} -> entity_counts.PERSON=1
        if key == "entity_counts" and isinstance(value, Mapping):
            nested = {f"{_ENTITY_KEY_PREFIX}{t}" if isinstance(t, str) else UNSAFE_KEY: n for t, n in value.items()}
            out.update(_filter(nested, extra))
            continue
        validator = _validator_for(key, extra)
        if validator is None:
            _reject(key, "unknown_key")
            continue
        if isinstance(value, str) and len(value) > _MAX_STR:
            _reject(key, "bad_value")
            continue
        try:
            ok = bool(validator(value))
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            _reject(key, "bad_value")
            continue
        out[key] = value
    return out


def safe_attributes(attrs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return only the allowlisted attributes with valid values; count and drop the rest."""
    return _filter(attrs)


# ---------------------------------------------------------------------------
# Export-time sanitizer (defence in depth, also covers FastAPI instrumentation spans)
# ---------------------------------------------------------------------------


def _safe_span_name(name: str) -> str:
    if name in SPAN_NAMES or _HTTP_SPAN_NAME_RE.match(name or ""):
        return name
    return "redacted"


def _sanitize_span(span: ReadableSpan) -> ReadableSpan:
    status = span.status
    if status.status_code is StatusCode.ERROR:
        status = Status(StatusCode.ERROR, GENERIC_ERROR_DESCRIPTION)
    elif status.description:
        status = Status(status.status_code)
    return ReadableSpan(
        name=_safe_span_name(span.name),
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=_filter(dict(span.attributes or {}), _HTTP_VALIDATORS),
        events=(),  # events (incl. exception events) are never exported
        links=(),
        kind=span.kind,
        status=status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class SanitizingSpanExporter(SpanExporter):
    """Wraps another exporter and re-applies the allowlist to every span before export."""

    def __init__(self, inner: SpanExporter) -> None:
        self.inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self.inner.export([_sanitize_span(s) for s in spans])

    def shutdown(self) -> None:
        self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.inner.force_flush(timeout_millis)


class JsonlSpanExporter(SpanExporter):
    """Writes spans as JSON lines (the no-collector fallback)."""

    def __init__(self, writer: _JsonlWriter) -> None:
        self.writer = writer

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for s in spans:
            duration = (s.end_time - s.start_time) / 1e6 if s.end_time and s.start_time else None
            self.writer.write(
                {
                    "type": "span",
                    "name": s.name,
                    "trace_id": format(s.context.trace_id, "032x") if s.context else None,
                    "span_id": format(s.context.span_id, "016x") if s.context else None,
                    "status": s.status.status_code.name,
                    "span_duration_ms": duration,
                    "attributes": dict(s.attributes or {}),
                }
            )
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


# ---------------------------------------------------------------------------
# State and setup
# ---------------------------------------------------------------------------


@dataclass
class _State:
    configured: bool = False
    tracer_provider: TracerProvider | None = None
    meter_provider: MeterProvider | None = None
    tracer: trace.Tracer | None = None
    jsonl: _JsonlWriter | None = None
    rejected_counter: Any = None
    requests_total: Any = None
    stage_latency_ms: Any = None
    vision_fallback_total: Any = None
    pii_entities_total: Any = None
    check_results_total: Any = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_STATE = _State()


def setup_telemetry(
    exporter: str | SpanExporter | None = None,
    jsonl_path: str | Path | None = None,
    *,
    metric_reader: MetricReader | None = None,
    sanitize: bool = True,
    force: bool = False,
) -> None:
    """Configure tracer + meter providers once. Later calls are no-ops unless `force=True`.

    exporter: "console" | "none" | "jsonl", or a SpanExporter instance (e.g. InMemorySpanExporter).
              Default from TELEMETRY_EXPORTER, else "console".
    jsonl_path: enables the JSON-lines writer for log_event (and for spans in "jsonl" mode).
    metric_reader: e.g. InMemoryMetricReader for tests. Default: console reader in console mode.
    sanitize: wrap the span exporter in SanitizingSpanExporter (only tests should turn this off).
    """
    with _STATE.lock:
        if _STATE.configured and not force:
            return
        if _STATE.configured:
            _shutdown_locked()

        mode = exporter if exporter is not None else os.environ.get("TELEMETRY_EXPORTER", "console")
        if isinstance(mode, str):
            mode = mode.strip().lower() or "console"
            if mode not in {"console", "none", "jsonl"}:
                mode = "console"

        env_path = os.environ.get("TELEMETRY_JSONL_PATH")
        path = jsonl_path or env_path or (DEFAULT_JSONL_PATH if mode == "jsonl" else None)
        writer = _JsonlWriter(Path(path)) if path else None

        resource = Resource({"service.name": SERVICE_NAME})
        tp = TracerProvider(resource=resource)
        span_exporter: SpanExporter | None
        if isinstance(mode, SpanExporter):
            span_exporter = mode
        elif mode == "console":
            span_exporter = ConsoleSpanExporter()
        elif mode == "jsonl":
            span_exporter = JsonlSpanExporter(writer)  # type: ignore[arg-type]
        else:
            span_exporter = None
        if span_exporter is not None:
            if sanitize:
                span_exporter = SanitizingSpanExporter(span_exporter)
            tp.add_span_processor(SimpleSpanProcessor(span_exporter))

        readers: list[MetricReader] = []
        if metric_reader is not None:
            readers.append(metric_reader)
        elif mode == "console":
            readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=60_000))
        mp = MeterProvider(resource=resource, metric_readers=readers)
        meter = mp.get_meter(SERVICE_NAME)

        _STATE.tracer_provider = tp
        _STATE.meter_provider = mp
        _STATE.tracer = tp.get_tracer(SERVICE_NAME)
        _STATE.jsonl = writer
        _STATE.rejected_counter = meter.create_counter(
            "telemetry.rejected_attributes", description="Attributes dropped by the allowlist"
        )
        _STATE.requests_total = meter.create_counter("requests_total", description="Requests by outcome")
        _STATE.stage_latency_ms = meter.create_histogram(
            "stage_latency_ms", unit="ms", description="Stage latency by stage"
        )
        _STATE.vision_fallback_total = meter.create_counter(
            "vision_fallback_total", description="Requests that took the vision fallback"
        )
        _STATE.pii_entities_total = meter.create_counter(
            "pii_entities_total", description="PII entities detected, by type"
        )
        _STATE.check_results_total = meter.create_counter(
            "check_results_total", description="Eligibility check results by status"
        )
        _STATE.configured = True


def _shutdown_locked() -> None:
    for provider in (_STATE.tracer_provider, _STATE.meter_provider):
        if provider is not None:
            try:
                provider.shutdown()
            except Exception:  # noqa: BLE001
                pass
    _STATE.configured = False
    _STATE.tracer_provider = _STATE.meter_provider = _STATE.tracer = None
    _STATE.jsonl = None
    _STATE.rejected_counter = _STATE.requests_total = _STATE.stage_latency_ms = None
    _STATE.vision_fallback_total = _STATE.pii_entities_total = _STATE.check_results_total = None


def reset_telemetry() -> None:
    """Shut down and forget the providers and the last-request summary (for tests)."""
    with _STATE.lock:
        _shutdown_locked()
    with _SUMMARY_LOCK:
        global _LAST_SUMMARY
        _LAST_SUMMARY = None


def _ensure() -> None:
    if not _STATE.configured:
        setup_telemetry()


def get_tracer_provider() -> TracerProvider:
    """The module's TracerProvider, e.g. for FastAPIInstrumentor.instrument_app(app, tracer_provider=...)."""
    _ensure()
    assert _STATE.tracer_provider is not None
    return _STATE.tracer_provider


def get_meter_provider() -> MeterProvider:
    _ensure()
    assert _STATE.meter_provider is not None
    return _STATE.meter_provider


def force_flush() -> None:
    if _STATE.tracer_provider is not None:
        _STATE.tracer_provider.force_flush()
    if _STATE.meter_provider is not None:
        _STATE.meter_provider.force_flush()


# ---------------------------------------------------------------------------
# Last-request summary (for the UI panel)
# ---------------------------------------------------------------------------


@dataclass
class _Summary:
    request_id: str | None = None
    stages: dict[str, float] = field(default_factory=dict)
    outcome: str | None = None
    check_status: str | None = None
    error_class: str | None = None
    document_type: str | None = None
    used_vision_fallback: bool | None = None
    entity_counts: dict[str, int] = field(default_factory=dict)

    def absorb(self, attrs: Mapping[str, Any]) -> None:
        """Take summary-relevant values from already-allowlisted attributes."""
        for key in ("request_id", "outcome", "check_status", "error_class", "document_type", "used_vision_fallback"):
            if key in attrs:
                setattr(self, key, attrs[key])
        for key, value in attrs.items():
            if key.startswith(_ENTITY_KEY_PREFIX):
                self.entity_counts[key[len(_ENTITY_KEY_PREFIX) :]] = value

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "stages": dict(self.stages),
            "outcome": self.outcome,
            "check_status": self.check_status,
            "error_class": self.error_class,
            "document_type": self.document_type,
            "used_vision_fallback": self.used_vision_fallback,
            "entity_counts": dict(self.entity_counts),
        }


_CURRENT: contextvars.ContextVar[_Summary | None] = contextvars.ContextVar("telemetry_summary", default=None)
_SUMMARY_LOCK = threading.Lock()
_LAST_SUMMARY: dict[str, Any] | None = None


def _absorb(attrs: Mapping[str, Any]) -> None:
    current = _CURRENT.get()
    if current is not None and attrs:
        current.absorb(attrs)


def last_request_summary() -> dict[str, Any] | None:
    """Latest finished request: per-stage durations (ms), outcome, check status, entity counts."""
    with _SUMMARY_LOCK:
        if _LAST_SUMMARY is None:
            return None
        return json.loads(json.dumps(_LAST_SUMMARY))  # deep copy


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


class StageSpan:
    """Handle yielded by stage_span; every attribute set through it is allowlisted."""

    def __init__(self, span: trace.Span) -> None:
        self._span = span

    def set(self, **attrs: Any) -> None:
        self.set_attributes(attrs)

    def set_attributes(self, attrs: Mapping[str, Any]) -> None:
        clean = safe_attributes(attrs)
        if clean:
            self._span.set_attributes(clean)
            _absorb(clean)

    @property
    def span(self) -> trace.Span:
        return self._span


@contextmanager
def stage_span(name: str, **attrs: Any) -> Iterator[StageSpan]:
    """Span for one stage (`request`, `ocr`, `pii`, `llm`, `rule`, `vision`).

    Applies the allowlist, records duration_ms (span attribute + stage_latency_ms histogram) and,
    on exception, sets status ERROR with a generic description and only `error_class`. The
    exception is re-raised; its message is never recorded.
    """
    if name not in SPAN_NAMES:
        raise ValueError("unknown telemetry span name")
    _ensure()
    assert _STATE.tracer is not None

    token = None
    summary: _Summary | None = None
    if name == "request":
        summary = _Summary()
        token = _CURRENT.set(summary)

    clean = safe_attributes({**attrs, "stage": name})
    _absorb(clean)
    start = time.perf_counter()
    error_class: str | None = None
    try:
        with _STATE.tracer.start_as_current_span(
            name, attributes=clean, record_exception=False, set_status_on_exception=False
        ) as span:
            handle = StageSpan(span)
            try:
                yield handle
            except BaseException as exc:
                if isinstance(exc, PipelineError):
                    error_class = ErrorClass(exc.error_class).value
                elif isinstance(exc, Exception):
                    error_class = ErrorClass.INTERNAL.value
                if error_class is not None:
                    span.set_attribute("error_class", error_class)
                    span.set_status(Status(StatusCode.ERROR, GENERIC_ERROR_DESCRIPTION))
                raise
            finally:
                duration = round((time.perf_counter() - start) * 1000.0, 3)
                span.set_attribute("duration_ms", duration)
                record_stage_latency(name, duration)
                current = _CURRENT.get()
                if current is not None:
                    current.stages[name] = duration
                    if error_class is not None and current.error_class is None:
                        current.error_class = error_class
    finally:
        if summary is not None and token is not None:
            if error_class is not None and summary.outcome is None:
                summary.outcome = "error"
            _CURRENT.reset(token)
            global _LAST_SUMMARY
            with _SUMMARY_LOCK:
                _LAST_SUMMARY = summary.as_dict()


def request_span(**attrs: Any):
    """Shorthand for stage_span("request", ...)."""
    return stage_span("request", **attrs)


# ---------------------------------------------------------------------------
# Structured log / JSONL fallback
# ---------------------------------------------------------------------------


def log_event(event: str, **attrs: Any) -> dict[str, Any] | None:
    """Emit one structured, allowlisted log record. Returns the record (None if the event is unknown)."""
    _ensure()
    if event not in EVENTS:
        _reject(None, "unknown_event")
        return None
    clean = safe_attributes(attrs)
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": event,
    }
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        record["trace_id"] = format(ctx.trace_id, "032x")
        record["span_id"] = format(ctx.span_id, "016x")
    record.update(clean)
    _absorb(clean)
    line = json.dumps(record, separators=(",", ":"))
    logger.info(line)
    if _STATE.jsonl is not None:
        try:
            _STATE.jsonl.write(record)
        except OSError:
            logger.warning("telemetry: could not write JSONL record")
    return record


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def record_request(outcome: str) -> None:
    _ensure()
    clean = safe_attributes({"outcome": outcome})
    _STATE.requests_total.add(1, clean)
    _absorb(clean)


def record_stage_latency(stage: str, ms: float) -> None:
    _ensure()
    if not _non_neg_number(ms):
        _reject("duration_ms", "bad_value")
        return
    _STATE.stage_latency_ms.record(ms, safe_attributes({"stage": stage}))


def record_entities(entity_counts: Mapping[str, int]) -> None:
    """Add per-type entity counts to pii_entities_total (and to the request summary)."""
    _ensure()
    accepted: dict[str, int] = {}
    for etype, n in (entity_counts or {}).items():
        attrs = safe_attributes({"entity_type": etype})
        if "entity_type" not in attrs or not _count(n):
            if "entity_type" in attrs:
                _reject("entity_type", "bad_value")
            continue
        if n:
            _STATE.pii_entities_total.add(n, attrs)
        accepted[f"{_ENTITY_KEY_PREFIX}{etype}"] = n
    _absorb(accepted)


def record_check(status: str) -> None:
    _ensure()
    clean = safe_attributes({"check_status": status})
    _STATE.check_results_total.add(1, clean)
    _absorb(clean)


def record_fallback() -> None:
    _ensure()
    _STATE.vision_fallback_total.add(1)
    _absorb({"used_vision_fallback": True})


__all__ = [
    "ALLOWED_KEYS",
    "CHECK_STATUSES",
    "EVENTS",
    "OUTCOMES",
    "SPAN_NAMES",
    "STAGES",
    "SanitizingSpanExporter",
    "StageSpan",
    "force_flush",
    "get_meter_provider",
    "get_tracer_provider",
    "last_request_summary",
    "log_event",
    "record_check",
    "record_entities",
    "record_fallback",
    "record_request",
    "record_stage_latency",
    "request_span",
    "reset_telemetry",
    "safe_attributes",
    "setup_telemetry",
    "stage_span",
]
