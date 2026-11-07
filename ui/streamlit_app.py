"""LearnSphere College admission document checker: client demo page.

Run with: streamlit run ui/streamlit_app.py   (API first: uvicorn app.main:app --port 8000)

The page talks to the API over HTTP only (API_URL, default http://localhost:8000) and never
imports the pipeline, so it shows exactly the object POST /v1/extract returns.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_S = 90.0
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
GENERIC_MESSAGE = "Could not read this document"
START_API_HINT = "Start the API: `uvicorn app.main:app --port 8000`"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples" / "test"
SAMPLE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
NO_SAMPLE = "None (upload a file)"

DOC_TYPE_LABELS = {
    "mark_sheet": "Mark sheet",
    "student_id_card": "Student ID card",
    "fee_receipt": "Fee receipt",
    "other": "Other document",
}

# status -> (background, text colour)
STATUS_STYLES = {
    "Eligible": ("#15803d", "#ffffff"),
    "Not eligible": ("#b91c1c", "#ffffff"),
    "Needs review": ("#d97706", "#ffffff"),
    "Not applicable": ("#475569", "#ffffff"),
    "Rejected": ("#1f2937", "#ffffff"),
}

FIELD_LABELS = (
    ("student_name", "Student name"),
    ("roll_number", "Roll number"),
    ("board_or_school", "Board / school"),
    ("passing_year", "Passing year"),
    ("percentage", "Percentage"),
)

ENTITY_LABELS = {
    "PERSON": "Person",
    "IN_PHONE": "Phone",
    "PHONE_NUMBER": "Phone",
    "EMAIL_ADDRESS": "Email",
    "IN_AADHAAR": "Aadhaar",
    "IN_PAN": "PAN",
    "DATE_OF_BIRTH": "Date of birth",
    "ROLL_NUMBER": "Roll number",
    "LOCATION": "Location",
    "ADDRESS": "Address",
}

STATUS_HINTS = {
    413: "The file is larger than 10 MB.",
    415: "Upload a PDF, PNG or JPG file.",
    400: "The file looks empty.",
    422: "The text could not be read clearly. Try a sharper scan.",
    502: "The AI step did not return a usable answer. Please try again.",
}

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1200px; }
.ls-brand { font-size: 0.85rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: #1e3a8a; margin-bottom: .1rem; }
.ls-title { font-size: 2rem; font-weight: 700; margin: 0 0 .25rem 0; line-height: 1.2; }
.ls-sub { color: #475569; margin: 0 0 .6rem 0; }
.ls-note { display: inline-block; font-size: .8rem; padding: .15rem .6rem; border-radius: 999px;
           background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.ls-badge { border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: .5rem; }
.ls-badge-label { font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; opacity: .85; }
.ls-badge-status { font-size: 1.9rem; font-weight: 700; line-height: 1.2; }
.ls-badge-reason { font-size: .95rem; margin-top: .25rem; opacity: .95; }
.ls-chip { display: inline-block; padding: .2rem .65rem; margin: 0 .35rem .35rem 0; border-radius: 999px;
           background: #eef2ff; color: #1e3a8a; font-size: .85rem; border: 1px solid #c7d2fe; }
.ls-chip b { font-weight: 700; }
.ls-ok { color: #15803d; font-weight: 600; }
.ls-bad { color: #b91c1c; font-weight: 600; }
</style>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sample_files() -> list[Path]:
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in SAMPLE_EXTENSIONS)


def sample_label(path: Path) -> str:
    """'test_004_mark_sheet.pdf' -> 'Mark sheet 004 (PDF)'."""
    parts = path.stem.split("_")
    number = parts[1] if len(parts) > 2 and parts[1].isdigit() else ""
    kind = "_".join(parts[2:]) if number else path.stem
    name = DOC_TYPE_LABELS.get(kind, kind.replace("_", " ").capitalize())
    if kind == "other":
        name = "Other document"
    return f"{name} {number} ({path.suffix.lstrip('.').upper()})".replace("  ", " ")


def kind_of(data: bytes) -> str | None:
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    return None


@st.cache_data(show_spinner=False, max_entries=16)
def pdf_first_page_png(data: bytes) -> bytes | None:
    try:
        import pymupdf

        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.page_count == 0:
                return None
            pix = doc[0].get_pixmap(dpi=110)
            return pix.tobytes("png")
    except Exception:
        return None


def call_api(data: bytes) -> dict[str, Any]:
    """POST the bytes to /v1/extract. Returns {"ok": True, "result": ...} or {"ok": False, ...}.

    The original filename is never sent; the API ignores it anyway.
    """
    ext = kind_of(data) or "bin"
    mime = {"pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg"}.get(ext, "application/octet-stream")
    try:
        resp = httpx.post(
            f"{API_URL}/v1/extract",
            files={"file": (f"document.{ext}", data, mime)},
            timeout=REQUEST_TIMEOUT_S,
        )
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return {"ok": False, "kind": "unreachable"}
    except httpx.TimeoutException:
        return {"ok": False, "kind": "timeout"}
    except httpx.HTTPError:
        return {"ok": False, "kind": "unreachable"}
    if resp.status_code != 200:
        return {"ok": False, "kind": "api_error", "status": resp.status_code}
    try:
        return {"ok": True, "result": resp.json()}
    except ValueError:
        return {"ok": False, "kind": "api_error", "status": resp.status_code}


def fetch_last_telemetry() -> dict[str, Any] | None:
    try:
        resp = httpx.get(f"{API_URL}/v1/telemetry/last", timeout=5.0)
        if resp.status_code == 200:
            body = resp.json()
            return body if isinstance(body, dict) else None
    except (httpx.HTTPError, ValueError):
        pass
    return None


def fmt_field(key: str, value: Any) -> str:
    if value is None or value == "":
        return "—"
    if key == "percentage":
        try:
            return f"{float(value):.2f}".rstrip("0").rstrip(".") + "%"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def fmt_ms(value: Any) -> str:
    try:
        ms = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:.0f} ms"


def entity_label(key: str) -> str:
    return ENTITY_LABELS.get(key, key.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_header() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="ls-brand">LearnSphere College</div>'
        '<div class="ls-title">LearnSphere College · Admission Document Checker</div>'
        '<p class="ls-sub">Upload a mark sheet, student ID card or fee receipt. OCR and personal-data '
        "masking run locally; only masked text reaches the AI model.</p>"
        '<span class="ls-note">Synthetic demo data only</span>',
        unsafe_allow_html=True,
    )
    st.write("")


def render_sidebar(samples: list[Path]) -> Path | None:
    with st.sidebar:
        st.markdown("### How it works")
        st.markdown(
            "1. **Upload**: the file is checked by type and size, then stored under a random ID.\n"
            "2. **Read locally**: Tesseract OCR extracts the text on this machine.\n"
            "3. **Mask locally**: names, IDs, phone numbers and other personal data are replaced "
            "with placeholders before anything leaves the machine.\n"
            "4. **Check**: the AI model classifies the masked text and extracts the fields; "
            "values are restored locally and the 60% eligibility rule is applied."
        )
        st.divider()
        st.markdown("### Or try a sample")
        labels = {sample_label(p): p for p in samples}
        choice = st.selectbox(
            "Or try a sample",
            [NO_SAMPLE, *labels],
            label_visibility="collapsed",
            key="sample_choice",
            help="Synthetic documents from the held-back test set.",
        )
        if not samples:
            st.caption("No samples found in data/samples/test.")
        st.divider()
        st.caption(f"API: {API_URL}")
        return labels.get(choice)


def render_preview(data: bytes) -> None:
    kind = kind_of(data)
    if kind == "pdf":
        png = pdf_first_page_png(data)
        if png:
            st.image(png, caption="Page 1 preview", width="stretch")
        else:
            st.info("Preview not available for this PDF.")
    elif kind in {"png", "jpg"}:
        st.image(data, width="stretch")
    else:
        st.info("Preview not available. Supported types: PDF, PNG, JPG.")


def render_status(check: dict[str, Any]) -> None:
    status = str(check.get("status", "Needs review"))
    bg, fg = STATUS_STYLES.get(status, STATUS_STYLES["Needs review"])
    reason = html.escape(str(check.get("reason", "")))
    st.markdown(
        f'<div class="ls-badge" style="background:{bg};color:{fg}">'
        f'<div class="ls-badge-label">Admission check</div>'
        f'<div class="ls-badge-status" data-testid="ls-status">{html.escape(status)}</div>'
        f'<div class="ls-badge-reason">{reason}</div></div>',
        unsafe_allow_html=True,
    )


def render_fields(result: dict[str, Any]) -> None:
    doc_type = result.get("document_type", "other")
    st.markdown(f"**Document type:** {DOC_TYPE_LABELS.get(doc_type, doc_type)}")
    fields = result.get("fields") or {}
    rows = [{"Field": label, "Value": fmt_field(key, fields.get(key))} for key, label in FIELD_LABELS]
    st.dataframe(rows, hide_index=True, width="stretch")


def render_privacy(pii: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown("**Privacy**")
        masked = bool(pii.get("masked_before_llm"))
        mark = '<span class="ls-ok">✓</span>' if masked else '<span class="ls-bad">✗</span>'
        st.markdown(f"Masked before AI: {mark}", unsafe_allow_html=True)
        counts = pii.get("entity_counts") or {}
        if counts:
            chips = "".join(
                f'<span class="ls-chip">{html.escape(entity_label(str(k)))} <b>{int(v)}</b></span>'
                for k, v in sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
            )
            st.markdown(chips, unsafe_allow_html=True)
            st.caption("Personal-data items replaced with placeholders, by type.")
        else:
            st.caption("No personal-data items were detected.")


def render_pipeline(pipe: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown("**Pipeline**")
        timings = pipe.get("timings_ms") or {}
        stages = [("ocr", "OCR"), ("pii", "Masking"), ("llm", "AI model"), ("rule", "Rules")]
        cols = st.columns(len(stages))
        for col, (key, label) in zip(cols, stages):
            col.metric(label, fmt_ms(timings.get(key)))
        values = {label: float(timings[key]) for key, label in stages if isinstance(timings.get(key), (int, float))}
        if values:
            chart = pd.DataFrame({"Stage": list(values), "Milliseconds": list(values.values())})
            st.bar_chart(
                chart,
                x="Stage",
                y="Milliseconds",
                horizontal=True,
                height=170,
                x_label="ms",
                y_label="",
                color="#1e3a8a",
                sort=False,
            )
        engine = str(pipe.get("ocr_engine", "—"))
        fallback = "Yes" if pipe.get("used_vision_fallback") else "No"
        total = sum(values.values()) if values else None
        st.caption(
            f"OCR engine: {engine} · Vision fallback: {fallback}"
            + (f" · Total: {fmt_ms(total)}" if total is not None else "")
        )


def render_telemetry(summary: dict[str, Any] | None) -> None:
    with st.expander("Last request telemetry (allowlisted)"):
        if not summary:
            st.caption("No telemetry recorded yet.")
            return
        stages = summary.get("stages") or {}
        if stages:
            st.dataframe(
                [{"Stage": k, "Duration": fmt_ms(v)} for k, v in stages.items()],
                hide_index=True,
                width="stretch",
            )
        st.caption(
            f"Outcome: {summary.get('outcome') or '—'} · Check: {summary.get('check_status') or '—'} · "
            f"Document type: {summary.get('document_type') or '—'}"
        )
        st.caption("Telemetry holds only allowlisted values: no document text, names or IDs.")


def render_result(result: dict[str, Any], telemetry_summary: dict[str, Any] | None) -> None:
    render_status(result.get("check") or {})
    render_fields(result)
    left, right = st.columns([2, 3])
    with left:
        render_privacy(result.get("pii") or {})
    with right:
        render_pipeline(result.get("pipeline") or {})
    render_telemetry(telemetry_summary)

    st.markdown("#### Response JSON")
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    request_id = str(result.get("request_id", "result"))[:8] or "result"
    st.download_button(
        "Download JSON",
        data=payload,
        file_name=f"learnsphere_check_{request_id}.json",
        mime="application/json",
        icon=":material/download:",
    )
    st.json(result, expanded=True)


def render_error(outcome: dict[str, Any]) -> None:
    kind = outcome.get("kind")
    if kind == "unreachable":
        st.error(f"The checking service is not reachable at {API_URL}. {START_API_HINT}")
    elif kind == "timeout":
        st.error(f"{GENERIC_MESSAGE}. The check took longer than {int(REQUEST_TIMEOUT_S)} seconds; please try again.")
    else:
        st.error(GENERIC_MESSAGE)
        hint = STATUS_HINTS.get(int(outcome.get("status") or 0))
        if hint:
            st.caption(hint)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="LearnSphere · Document Checker",
        page_icon=":material/school:",
        layout="wide",
    )
    render_header()
    sample = render_sidebar(sample_files())

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("#### Document")
        uploaded = st.file_uploader(
            "Upload a document (PDF, PNG, JPG · max 10 MB)",
            type=["pdf", "png", "jpg", "jpeg"],
            key="upload",
        )
        data: bytes | None = None
        source = ""
        if uploaded is not None:
            data = uploaded.getvalue()
            source = "Uploaded file"
        elif sample is not None:
            data = sample.read_bytes()
            source = f"Sample: {sample_label(sample)}"

        too_big = data is not None and len(data) > MAX_UPLOAD_BYTES
        if data is not None:
            st.caption(source)
            if too_big:
                st.error("This file is larger than 10 MB.")
            else:
                render_preview(data)

        clicked = st.button(
            "Check document",
            type="primary",
            disabled=data is None or too_big,
            width="stretch",
            icon=":material/fact_check:",
        )

    if clicked and data is not None and not too_big:
        with right:
            with st.spinner("Reading, masking and checking the document…"):
                outcome = call_api(data)
                telemetry_summary = fetch_last_telemetry() if outcome.get("ok") else None
        st.session_state["outcome"] = outcome
        st.session_state["telemetry"] = telemetry_summary
        st.session_state["outcome_source"] = source

    with right:
        st.markdown("#### Result")
        outcome = st.session_state.get("outcome")
        if outcome is None:
            st.info("Upload a document or pick a sample in the sidebar, then press **Check document**.")
        elif outcome.get("ok"):
            if st.session_state.get("outcome_source") and st.session_state["outcome_source"] != source:
                st.caption(f"Showing the result for: {st.session_state['outcome_source']}")
            render_result(outcome["result"], st.session_state.get("telemetry"))
        else:
            render_error(outcome)


main()
