"""Eval script tests: temporary labeled sets and fake responses, never the network."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import httpx
import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from PIL import Image, ImageDraw, ImageFont
from pydantic import SecretStr

from app import telemetry as tel
from app.config import Settings
from app.errors import ErrorClass, PipelineError
from app.schemas import Check, CheckStatus, DocumentType, ExtractResponse, LearnSphereFields, PiiSummary, PipelineSummary
from eval import evaluate as ev

NAME, ROLL = "Zebulon Quixote", "ZQ/99/1234"
COLUMNS = [
    "filename", "true_type", "student_name", "roll_number", "board_or_school",
    "passing_year", "percentage", "expected_status", "variant", "pii_values",
]  # fmt: skip


def _row(filename="a.png", **overrides) -> dict[str, str]:
    row = {
        "filename": filename,
        "true_type": "mark_sheet",
        "student_name": NAME,
        "roll_number": ROLL,
        "board_or_school": "Central   Board",
        "passing_year": "2024",
        "percentage": "76.4",
        "expected_status": "Eligible",
        "variant": "clean",
        "pii_values": f"{NAME}|secret-aadhaar",
    }
    return row | overrides


def _response(doc=DocumentType.MARK_SHEET, status=CheckStatus.ELIGIBLE, **fields) -> ExtractResponse:
    values = {
        "student_name": NAME.upper(),
        "roll_number": ROLL,
        "board_or_school": "central board",
        "passing_year": 2024,
        "percentage": 76.40,
    }
    return ExtractResponse(
        request_id="r",
        document_type=doc,
        fields=LearnSphereFields(**(values | fields)),
        check=Check(status=status, reason="x"),
        pii=PiiSummary(masked_before_llm=True),
        pipeline=PipelineSummary(),
    )


def _dataset(tmp_path: Path, rows: list[dict[str, str]], name="test") -> Path:
    data = tmp_path / "data"
    folder = data / "samples" / name
    folder.mkdir(parents=True, exist_ok=True)
    with (data / f"{name}_set.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        (folder / row["filename"]).write_bytes(b"sample")
    return data


@pytest.fixture
def fake_extract(monkeypatch):
    """Replaces the pipeline. `answers` maps a sample's bytes to a reply or an exception to raise."""
    answers: dict[bytes, object] = {}

    def extract(data, settings=None, transport=None):
        answer = answers.get(data, _response())
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(ev, "extract", extract)
    return answers


@pytest.mark.parametrize(
    "name, expected, actual, ok",
    [
        ("student_name", "Asha  Rao", "ASHA RAO", True),
        ("student_name", "Asha Rao", "Asha Roy", False),
        ("percentage", "60", 60.0, True),
        ("percentage", "76.4", "76.40", True),
        ("percentage", "59.9", 60.0, False),
        ("passing_year", "2024", 2024, True),
        ("percentage", "", None, True),
        ("percentage", "", 60.0, False),
        ("student_name", "Asha", None, False),
        ("percentage", "n/a", 60.0, False),
    ],
)
def test_field_matches(name, expected, actual, ok):
    assert ev.field_matches(name, expected, actual) is ok


def test_all_correct_document_scores_every_field(tmp_path, fake_extract):
    data = _dataset(tmp_path, [_row()])
    row = ev.load_rows("test", data)[0]
    result = ev.evaluate_row(row)
    assert result.type_ok and result.status_ok and not result.is_miss
    assert result.fields_scored == 5 and result.fields_ok == 5


def test_only_labeled_columns_are_scored(tmp_path, fake_extract):
    (tmp_path / "s.png").write_bytes(b"sample")
    row = {
        "filename": "s.png", "true_type": "mark_sheet", "expected_status": "Eligible",
        "percentage": "76.4", "path": str(tmp_path / "s.png"),
    }  # fmt: skip
    result = ev.evaluate_row(row)
    assert result.fields_scored == 1 and not result.field_misses and result.variant == ev.NO_VARIANT


def test_an_empty_label_requires_null(tmp_path, fake_extract):
    data = _dataset(tmp_path, [_row(percentage="")])
    result = ev.evaluate_row(ev.load_rows("test", data)[0])
    assert result.field_misses == ["percentage"]


def test_pipeline_error_is_a_miss_and_the_run_continues(tmp_path, fake_extract, capsys):
    rows = [_row("bad.png"), _row("good.png")]
    data = _dataset(tmp_path, rows)
    (data / "samples" / "test" / "bad.png").write_bytes(b"bad")
    fake_extract[b"bad"] = PipelineError(ErrorClass.LLM_FAILED)
    code = ev.main(["--out", str(tmp_path / "r.jsonl")], data_dir=data)
    out = capsys.readouterr().out
    assert code == 0
    assert "error: llm_failed" in out and "1 pipeline errors" in out
    assert "Status 1/2 (50.0%)" in out and "Field match 5/10 (50.0%)" in out


def test_unexpected_exception_records_only_its_class(tmp_path, fake_extract):
    data = _dataset(tmp_path, [_row()])
    fake_extract[b"sample"] = ValueError(f"document said {NAME}")
    result = ev.evaluate_row(ev.load_rows("test", data)[0])
    assert result.error == "ValueError" and result.is_miss


def test_missing_sample_file_is_a_miss(tmp_path, fake_extract):
    data = _dataset(tmp_path, [_row()])
    row = ev.load_rows("test", data)[0]
    Path(row["path"]).unlink()
    assert ev.evaluate_row(row).error == "FileNotFoundError"


def test_breakdowns_group_by_type_and_any_variant(tmp_path, fake_extract, capsys):
    rows = [
        _row("m.png"),
        _row("i.png", true_type="student_id_card", expected_status="Not applicable", variant="brand_new_variant"),
    ]
    data = _dataset(tmp_path, rows)
    (data / "samples" / "test" / "i.png").write_bytes(b"id")
    fake_extract[b"id"] = _response(DocumentType.FEE_RECEIPT, CheckStatus.NOT_APPLICABLE)
    report = tmp_path / "r.jsonl"
    assert ev.main(["--out", str(report)], data_dir=data) == 0
    out = capsys.readouterr().out
    assert "type: student_id_card -> fee_receipt" in out and "brand_new_variant" in out
    summary = json.loads(report.read_text().splitlines()[-1])
    assert summary["kind"] == "summary"
    assert summary["by_type"]["student_id_card"]["document_type"]["correct"] == 0
    assert summary["by_type"]["student_id_card"]["status"]["correct"] == 1
    assert summary["by_type"]["mark_sheet"]["document_type"]["accuracy"] == 1.0
    assert summary["by_variant"]["brand_new_variant"]["correct"] == 1


def test_output_and_report_never_contain_field_values(tmp_path, fake_extract, capsys):
    data = _dataset(tmp_path, [_row(percentage="77.77")])
    fake_extract[b"sample"] = _response(
        student_name="Wrong Person", roll_number="WRONG/1", board_or_school="Wrong Board", passing_year=1999, percentage=12.34
    )
    report = tmp_path / "r.jsonl"
    ev.main(["--out", str(report)], data_dir=data)
    captured = capsys.readouterr()
    text = captured.out + captured.err + report.read_text()
    assert "student_name" in text and "percentage" in text  # the miss is named
    for secret in (NAME, ROLL, "Central", "secret-aadhaar", "Wrong Person", "WRONG/1", "Wrong Board", "77.77", "12.34", "1999"):
        assert secret.lower() not in text.lower(), secret


def test_set_all_is_dev_then_test_and_limit_applies(tmp_path, fake_extract, capsys):
    _dataset(tmp_path, [_row("d.png")], "dev")
    data = _dataset(tmp_path, [_row("t.png")], "test")
    assert [r["filename"] for r in ev.load_rows("all", data)] == ["d.png", "t.png"]
    ev.main(["--set", "all", "--limit", "1", "--out", str(tmp_path / "r.jsonl")], data_dir=data)
    assert "1 documents" in capsys.readouterr().out


@pytest.mark.parametrize("threshold, code", [("1.0", 1), ("0.5", 0)])
def test_min_status_accuracy_sets_the_exit_code(tmp_path, fake_extract, threshold, code):
    rows = [_row("a.png"), _row("b.png", expected_status="Rejected")]
    data = _dataset(tmp_path, rows)
    argv = ["--out", str(tmp_path / "r.jsonl"), "--min-status-accuracy", threshold]
    assert ev.main(argv, data_dir=data) == code


def test_missing_csv_exits_2(tmp_path, capsys):
    assert ev.main(["--out", str(tmp_path / "r.jsonl")], data_dir=tmp_path) == 2
    assert "data.generate" in capsys.readouterr().err


def test_csv_without_required_columns_exits_2(tmp_path, capsys):
    (tmp_path / "test_set.csv").write_text("filename,true_type\na.png,other\n", encoding="utf-8")
    assert ev.main(["--out", str(tmp_path / "r.jsonl")], data_dir=tmp_path) == 2
    assert "expected_status" in capsys.readouterr().err


@pytest.fixture
def otel(tmp_path):
    tel.reset_telemetry()
    tel.setup_telemetry(
        exporter=InMemorySpanExporter(), jsonl_path=tmp_path / "t.jsonl", metric_reader=InMemoryMetricReader()
    )
    yield
    tel.reset_telemetry()


def test_real_pipeline_with_injected_transport(tmp_path, otel, capsys):
    """One rendered page through real OCR and masking; only the LLM reply is canned."""
    img = Image.new("RGB", (1400, 500), "white")
    font = ImageFont.load_default(size=44)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(("Riverside College Fee Receipt", "Tuition fee paid in full", "Amount received 25000")):
        draw.text((60, 60 + i * 120), line, fill="black", font=font)

    row = _row(
        "r.png", true_type="fee_receipt", expected_status="Not applicable", variant="clean",
        student_name="", roll_number="", board_or_school="Riverside College", passing_year="", percentage="",
    )  # fmt: skip
    data = _dataset(tmp_path, [row])
    img.save(data / "samples" / "test" / "r.png")

    canned = {
        "document_type": "fee_receipt",
        "fields": {
            "student_name": None, "roll_number": None, "board_or_school": "Riverside College",
            "passing_year": None, "percentage": None,
        },
        "percentage_as_printed": None,
        "cgpa_only": False,
    }  # fmt: skip
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(canned)}}], "usage": {}})

    settings = Settings(
        upload_dir=tmp_path / "uploads", llm_api_key=SecretStr("k"), llm_model="test-model"
    )
    argv = ["--out", str(tmp_path / "r.jsonl")]
    assert ev.main(argv, settings=settings, transport=httpx.MockTransport(handler), data_dir=data) == 0
    assert len(calls) == 1
    out = capsys.readouterr().out
    assert "Status 1/1 (100.0%)" in out and "Document type 1/1" in out and "Field match 5/5" in out


def test_empty_ocr_scores_as_could_not_read(tmp_path, monkeypatch):
    from app.errors import ErrorClass, PipelineError
    from eval import evaluate as ev

    def no_boxes(*_args, **_kwargs):
        raise PipelineError(ErrorClass.OCR_EMPTY)

    monkeypatch.setattr(ev, "extract", no_boxes)
    sample = tmp_path / "blank.png"
    sample.write_bytes(b"x")
    row = {
        "filename": "blank.png", "path": str(sample), "true_type": "other", "variant": "blank_page",
        "expected_status": "Could not read", "student_name": "", "percentage": "",
    }
    result = ev.evaluate_row(row)
    assert result.status_ok and result.type_ok and not result.is_miss
    row["expected_status"] = "Eligible"
    assert not ev.evaluate_row(row).status_ok
