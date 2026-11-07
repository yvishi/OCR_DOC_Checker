"""Reusable leak-check helpers for telemetry tests (docs/ARCHITECTURE.md §7 "Leak test")."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def serialize_span(span: Any) -> dict[str, Any]:
    """Everything a span could carry to an exporter: name, attributes, events, status, resource, links."""
    return {
        "name": span.name,
        "attributes": dict(span.attributes or {}),
        "events": [
            {"name": e.name, "attributes": dict(e.attributes or {})} for e in (getattr(span, "events", None) or ())
        ],
        "status": {
            "code": span.status.status_code.name,
            "description": span.status.description,
        },
        "resource": dict(span.resource.attributes) if span.resource is not None else {},
        "links": [dict(link.attributes or {}) for link in (getattr(span, "links", None) or ())],
    }


def _leaves(obj: Any, out: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            _leaves(v, out)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _leaves(v, out)
    elif obj is not None:
        out.append(str(obj))


def read_jsonl(path: str | Path) -> list[str]:
    """Lines of a telemetry JSONL file ([] if it does not exist)."""
    p = Path(path)
    if not p.exists():
        return []
    return [line for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_telemetry(exported_spans: Iterable[Any], log_lines: Iterable[Any] = ()) -> str:
    spans = [serialize_span(s) for s in exported_spans]
    logs = [line if isinstance(line, str) else json.dumps(line, default=str) for line in log_lines]
    return json.dumps({"spans": spans, "logs": logs}, default=str, ensure_ascii=False)


def assert_no_leak(exported_spans: Iterable[Any], log_lines: Iterable[Any], secrets: Sequence[str]) -> None:
    """Fail if any secret appears in the spans or log lines (case-insensitive, whitespace-normalised).

    exported_spans: e.g. InMemorySpanExporter.get_finished_spans()
    log_lines: JSONL lines (str) or dict records; e.g. read_jsonl(path) or caplog messages
    secrets: synthetic names, roll numbers, Aadhaar numbers, emails, ...
    """
    spans = list(exported_spans)
    logs = list(log_lines)
    # check the JSON dump, plus every raw leaf string, so escaping (\\n, \\u00e9) cannot hide a value
    leaves: list[str] = []
    _leaves([serialize_span(s) for s in spans], leaves)
    for line in logs:
        if isinstance(line, str):
            leaves.append(line)
            try:
                _leaves(json.loads(line), leaves)
            except ValueError:
                pass
        else:
            _leaves(line, leaves)
    haystacks = [_norm(dump_telemetry(spans, logs))] + [_norm(x) for x in leaves]
    leaked = []
    for secret in secrets:
        needle = _norm(str(secret))
        if needle and any(needle in h for h in haystacks):
            leaked.append(secret)
    assert not leaked, f"telemetry leaked {len(leaked)} secret value(s): {leaked!r}"
