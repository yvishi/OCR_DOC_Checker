"""Score the real pipeline against a labeled CSV: field match, document type and status accuracy.

Run from the repo root (each document makes one live LLM call):

    python -m eval.evaluate                  # the held-back test set
    python -m eval.evaluate --set all --limit 5

The labels hold synthetic names and IDs, so nothing here prints or writes a field value: output
carries only filenames, variants, document types, statuses, error classes and field names.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from app import telemetry as tel
from app.config import Settings
from app.errors import PipelineError
from app.pipeline import extract

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
# `*.jsonl` is gitignored, so a run never leaves a report to commit by accident.
DEFAULT_REPORT = ROOT / "eval" / "last_report.jsonl"

SETS = {"dev": ("dev_set.csv", "dev"), "test": ("test_set.csv", "test")}
REQUIRED_COLUMNS = ("filename", "true_type", "expected_status")
TEXT_FIELDS = ("student_name", "roll_number", "board_or_school")
NUMERIC_FIELDS = ("passing_year", "percentage")
FIELDS = TEXT_FIELDS + NUMERIC_FIELDS
NUMERIC_TOLERANCE = 0.005
NO_VARIANT = "(none)"


@dataclass
class DocResult:
    filename: str
    variant: str
    true_type: str
    expected_status: str
    actual_type: str | None = None
    actual_status: str | None = None
    error: str | None = None  # error class only, never an exception message
    fields_scored: int = 0
    field_misses: list[str] = field(default_factory=list)

    @property
    def type_ok(self) -> bool:
        return self.actual_type == self.true_type

    @property
    def status_ok(self) -> bool:
        return self.actual_status == self.expected_status

    @property
    def fields_ok(self) -> int:
        return self.fields_scored - len(self.field_misses)

    @property
    def is_miss(self) -> bool:
        return bool(self.error or self.field_misses) or not (self.type_ok and self.status_ok)


class DatasetError(Exception):
    """The labeled CSV is missing or unusable. The message holds no label values."""


def load_rows(set_name: str, data_dir: Path = DATA_DIR) -> list[dict[str, str]]:
    """Rows of the chosen set, each with a `path` to its sample. `all` is dev, then test."""
    rows: list[dict[str, str]] = []
    for name in SETS if set_name == "all" else (set_name,):
        csv_name, folder = SETS[name]
        csv_path = data_dir / csv_name
        if not csv_path.exists():
            raise DatasetError(f"{csv_name} not found; run `python -m data.generate` first")
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise DatasetError(f"{csv_name} is missing columns: {', '.join(missing)}")
            for row in reader:
                row["path"] = str(data_dir / "samples" / folder / row["filename"])
                rows.append(row)
    return rows


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def field_matches(name: str, expected: str | None, actual: Any) -> bool:
    """Case and whitespace never matter; numbers compare by value; an empty label means null."""
    want = (expected or "").strip()
    got = "" if actual is None else str(actual).strip()
    if not want or not got:
        return want == got
    if name in NUMERIC_FIELDS:
        try:
            return abs(float(want) - float(got)) < NUMERIC_TOLERANCE
        except ValueError:
            return False
    return _norm(want) == _norm(got)


def evaluate_row(
    row: dict[str, str], settings: Settings | None = None, transport: httpx.BaseTransport | None = None
) -> DocResult:
    """Run one document. Any failure becomes a scored miss, so the run always continues."""
    scored = [name for name in FIELDS if name in row]
    result = DocResult(
        filename=row["filename"],
        variant=(row.get("variant") or "").strip() or NO_VARIANT,
        true_type=row["true_type"],
        expected_status=row["expected_status"],
        fields_scored=len(scored),
    )
    try:
        response = extract(Path(row["path"]).read_bytes(), settings, transport)
    except PipelineError as exc:
        result.error = exc.error_class.value
    except Exception as exc:  # the class name only: exception text can carry document content
        result.error = type(exc).__name__
    if result.error:
        result.field_misses = scored  # no answer is a miss even where the label is null
        return result

    result.actual_type = response.document_type.value
    result.actual_status = response.check.status.value
    actual = response.fields.model_dump()
    result.field_misses = [name for name in scored if not field_matches(name, row[name], actual.get(name))]
    return result


def _score(correct: int, total: int) -> dict[str, Any]:
    return {"correct": correct, "total": total, "accuracy": round(correct / total, 4) if total else None}


def _metrics(results: Sequence[DocResult]) -> dict[str, dict[str, Any]]:
    return {
        "field_match": _score(sum(r.fields_ok for r in results), sum(r.fields_scored for r in results)),
        "document_type": _score(sum(r.type_ok for r in results), len(results)),
        "status": _score(sum(r.status_ok for r in results), len(results)),
    }


def _group(results: Iterable[DocResult], attr: str) -> dict[str, list[DocResult]]:
    groups: dict[str, list[DocResult]] = {}
    for r in results:
        groups.setdefault(getattr(r, attr), []).append(r)
    return groups


def summarize(results: Sequence[DocResult], set_name: str) -> dict[str, Any]:
    return {
        "set": set_name,
        "documents": len(results),
        "errors": sum(1 for r in results if r.error),
        "metrics": _metrics(results),
        "by_type": {k: _metrics(v) for k, v in _group(results, "true_type").items()},
        "by_variant": {
            k: _score(sum(r.status_ok for r in v), len(v)) for k, v in _group(results, "variant").items()
        },
    }


def _fmt(score: dict[str, Any]) -> str:
    pct = "n/a" if score["accuracy"] is None else f"{score['accuracy'] * 100:.1f}%"
    return f"{score['correct']}/{score['total']} ({pct})"


def _miss_line(r: DocResult) -> str:
    parts: list[str] = []
    if r.error:
        parts.append(f"error: {r.error}")
    if not r.type_ok:
        parts.append(f"type: {r.true_type} -> {r.actual_type or '-'}")
    if not r.status_ok:
        parts.append(f"status: {r.expected_status} -> {r.actual_status or '-'}")
    if r.field_misses:
        parts.append(f"fields: {', '.join(r.field_misses)}")
    return f"  {r.filename} [{r.variant}]  " + "; ".join(parts)


def format_report(summary: dict[str, Any], results: Sequence[DocResult]) -> str:
    m = summary["metrics"]
    lines = [
        f"Eval on {summary['set']}: {summary['documents']} documents, {summary['errors']} pipeline errors",
        "",
        "By document type (true type):",
        f"  {'type':<18}{'fields':<22}{'document type':<20}status",
    ]
    for name, t in summary["by_type"].items():
        lines.append(
            f"  {name:<18}{_fmt(t['field_match']):<22}{_fmt(t['document_type']):<20}{_fmt(t['status'])}"
        )
    lines += ["", "Status by variant:"]
    lines += [f"  {name:<28}{_fmt(score)}" for name, score in summary["by_variant"].items()]
    misses = [r for r in results if r.is_miss]
    lines += ["", f"Misses ({len(misses)}):"]
    lines += [_miss_line(r) for r in misses] or ["  none"]
    lines += [
        "",
        f"Field match {_fmt(m['field_match'])} | "
        f"Document type {_fmt(m['document_type'])} | "
        f"Status {_fmt(m['status'])}",
    ]
    return "\n".join(lines)


def write_report(path: Path, summary: dict[str, Any], results: Sequence[DocResult]) -> None:
    """JSON lines: one `document` record per file, then one `summary` record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps({"kind": "document", **asdict(r)}) + "\n")
        fh.write(json.dumps({"kind": "summary", **summary}) + "\n")


def _fraction(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be a fraction between 0 and 1")
    return value


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(
    argv: Sequence[str] | None = None,
    settings: Settings | None = None,
    transport: httpx.BaseTransport | None = None,
    data_dir: Path = DATA_DIR,
) -> int:
    """Exit 0 normally, 1 below --min-status-accuracy, 2 when the labeled set can't be loaded."""
    parser = argparse.ArgumentParser(prog="python -m eval.evaluate", description="Score the pipeline on a labeled set.")
    parser.add_argument("--set", dest="set_name", choices=[*SETS, "all"], default="test")
    parser.add_argument("--limit", type=_positive, help="only the first N documents (a smoke run)")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT, help="JSON-lines report path")
    parser.add_argument(
        "--min-status-accuracy", type=_fraction, help="exit 1 when status accuracy is below this fraction, e.g. 0.9"
    )
    args = parser.parse_args(argv)

    try:
        rows = load_rows(args.set_name, data_dir)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rows = rows[: args.limit] if args.limit else rows
    if not rows:
        print("error: the labeled set has no documents", file=sys.stderr)
        return 2

    tel.setup_telemetry(exporter="none")  # keep span output out of the report; a no-op if already set up
    results = [evaluate_row(row, settings, transport) for row in rows]
    summary = summarize(results, args.set_name)
    write_report(args.out, summary, results)
    print(format_report(summary, results))

    accuracy = summary["metrics"]["status"]["accuracy"]
    if args.min_status_accuracy is not None and accuracy < args.min_status_accuracy:
        print(f"Status accuracy is below the required {args.min_status_accuracy:.2f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
