"""Tests for the synthetic document generator (data/generate.py)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.schemas import DocumentType
from data import generate as gen

STATUSES = {"Eligible", "Not eligible", "Needs review", "Not applicable", "Rejected"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("gen1")
    gen.generate(out_dir=root / "samples", csv_dir=root)
    return root


def test_deterministic_across_runs(full_run, tmp_path):
    gen.generate(out_dir=tmp_path / "samples", csv_dir=tmp_path)
    for s in gen.SETS:
        assert (full_run / f"{s}_set.csv").read_text() == (tmp_path / f"{s}_set.csv").read_text()
        for f in sorted((full_run / "samples" / s).iterdir()):
            assert f.read_bytes() == (tmp_path / "samples" / s / f.name).read_bytes(), f.name


def test_only_renders_subset_identically(full_run, tmp_path):
    gen.generate(out_dir=tmp_path / "samples", csv_dir=tmp_path, sets=["test"], only=[1, 7, 15])
    files = sorted(p.name for p in (tmp_path / "samples" / "test").iterdir())
    assert files == ["test_001_mark_sheet.png", "test_007_mark_sheet.png", "test_015_fee_receipt.pdf"]
    for name in files:
        assert (tmp_path / "samples" / "test" / name).read_bytes() == (full_run / "samples" / "test" / name).read_bytes()
    # CSV is always written for the whole set
    assert len(_read_csv(tmp_path / "test_set.csv")) == 20


def test_csv_rows_match_files(full_run):
    for s in gen.SETS:
        rows = _read_csv(full_run / f"{s}_set.csv")
        assert list(rows[0].keys()) == gen.CSV_COLUMNS
        on_disk = {p.name for p in (full_run / "samples" / s).iterdir()}
        assert {r["filename"] for r in rows} == on_disk


def test_repo_csv_is_current():
    """The committed CSVs equal what the generator plans, and every row has a file on disk."""
    for s in gen.SETS:
        path = gen.DEFAULT_CSV_DIR / f"{s}_set.csv"
        if not path.exists():
            pytest.skip("run `python -m data.generate` first")
        rows = _read_csv(path)
        assert rows == [p.row for p in gen.plan_set(s)]
        sample_dir = gen.DEFAULT_OUT / s
        on_disk = {p.name for p in sample_dir.iterdir() if not p.name.startswith(".")}
        assert {r["filename"] for r in rows} == on_disk


def test_set_sizes_types_and_statuses():
    test_rows = [p.row for p in gen.plan_set("test")]
    dev_rows = [p.row for p in gen.plan_set("dev")]
    assert len(test_rows) == 20
    assert 15 <= len(dev_rows) <= 25
    for rows in (test_rows, dev_rows):
        assert {r["expected_status"] for r in rows} == STATUSES
        assert {r["true_type"] for r in rows} == {t.value for t in DocumentType}
        exts = {r["filename"].rsplit(".", 1)[1] for r in rows}
        assert "pdf" in exts and exts & {"png", "jpg"}
        n_marks = sum(r["true_type"] == "mark_sheet" for r in rows)
        assert 0.4 <= n_marks / len(rows) <= 0.65


def test_status_rules():
    for s in gen.SETS:
        for p in gen.plan_set(s):
            r = p.row
            if p.doc_type in ("student_id_card", "fee_receipt"):
                assert r["expected_status"] == "Not applicable"
            elif p.doc_type == "other":
                assert r["expected_status"] == "Rejected"
                assert r["student_name"] == r["percentage"] == ""
            elif p.variant in ("stamp_over_total", "cgpa_only", "subject_mismatch"):
                assert r["expected_status"] == "Needs review"
            else:
                pct = float(r["percentage"])
                assert r["expected_status"] == ("Eligible" if pct >= 60 else "Not eligible")
            if p.variant == "cgpa_only":
                assert r["percentage"] == ""
            if p.variant == "boundary_60":
                assert float(r["percentage"]) == 60.0 and r["expected_status"] == "Eligible"
            if p.variant == "boundary_59_9":
                assert float(r["percentage"]) == 59.9 and r["expected_status"] == "Not eligible"
            if p.variant == "near_cutoff":
                assert 58 <= float(r["percentage"]) <= 62


def test_required_variants_present():
    variants = {p.variant for p in gen.plan_set("test")}
    assert {"clean", "boundary_60", "boundary_59_9", "near_cutoff", "stamp_over_total", "cgpa_only",
            "rotated_90", "rotated_180", "noisy", "subject_mismatch"} <= variants


def test_mark_arithmetic():
    for s in gen.SETS:
        for p in gen.plan_set(s):
            if p.doc_type != "mark_sheet" or p.data["cgpa"] is not None:
                continue
            d = p.data
            maxes = [x[2] for x in d["subjects"]]
            marks = [x[3] for x in d["subjects"]]
            assert all(0 <= m <= mx for m, mx in zip(marks, maxes))
            assert sum(maxes) == d["max_total"]
            pct = float(p.row["percentage"])
            assert round(d["total"] / d["max_total"] * 100, 2) == pct, p.filename
            subject_pct = sum(marks) / d["max_total"] * 100
            if p.variant == "subject_mismatch":
                assert abs(subject_pct - pct) > 0.5
            else:
                assert sum(marks) == d["total"], p.filename


def test_filenames_and_pii():
    for s in gen.SETS:
        for p in gen.plan_set(s):
            r = p.row
            pii = r["pii_values"].split("|")
            assert pii and all(pii)
            if r["student_name"]:
                assert r["student_name"] in pii
                for part in r["student_name"].lower().split():
                    assert part not in r["filename"].lower()
            if r["roll_number"]:
                assert r["roll_number"] in pii
