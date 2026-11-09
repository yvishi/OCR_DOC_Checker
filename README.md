# OCR_DOC_Checker

Admission document checker for **LearnSphere College** (fictional), Build Fast with AI FDE Fellowship, Week 2, Team Alpha.

An admissions officer uploads a PDF or image. The app classifies it as a mark sheet, student ID card or fee receipt, extracts five fields, and checks mark sheets against a 60% cutoff. OCR and personal-data masking run locally; only masked text ever reaches the LLM.

> **Synthetic data only.** No real mark sheets, IDs or receipts belong in this repo.

Status: build phases 0–8 and 10 are done: synthetic data, upload + OCR + preprocessing, PII masking, telemetry, the LLM call (OpenRouter, `anthropic/claude-haiku-5.5`), local restore + guards + eligibility rule, the FastAPI service with the Streamlit demo page, the eval script, the vision fallback, and the subject recompute. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the workflow and stack.

Latest live eval on all 49 synthetic documents (29 dev, 20 held-back test): status 49/49, document type 49/49, fields 243/245. These documents come from one generator, so the numbers do not predict accuracy on real scans.

## Setup (macOS)

1. Install Homebrew, then Python 3.12 and Tesseract (installs the `eng` and `osd` language data):
   ```bash
   brew install python@3.12 tesseract
   ```
2. Create the environment:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_lg
   ```
3. Copy `.env.example` to `.env` (see step 3 below). `TESSERACT_CMD` can stay empty: the app finds `tesseract` on PATH or in `/opt/homebrew/bin`.

## Setup (Windows)

1. Install Python 3.12 and Tesseract OCR (UB Mannheim build).
2. Create the environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m spacy download en_core_web_lg
   ```
3. Copy `.env.example` to `.env` and fill in the values. The LLM API key is read from the `LLM_API_KEY` environment variable; `.env` is never committed.

## Run (increment 1)

```bash
python -m data.generate                                   # 29 dev + 20 test synthetic documents, CSVs
python -m app.ocr data/samples/test/test_009_mark_sheet.png          # OCR stats only
python -m app.pipeline data/samples/test/test_001_mark_sheet.png --show-masked   # what the LLM will see
python -m app.pipeline data/samples/test/test_001_mark_sheet.png --llm         # plus the LLM call (needs .env)
pytest                                                    # unit tests + telemetry leak test over all samples
python -m eval.evaluate --set all                         # live accuracy by type and variant (one LLM call per document, needs .env)
```

The eval's default set is the held-back `test` set. It prints no names or IDs, and writes `eval/last_report.jsonl` (gitignored).

## Run the demo

Start the API and the page in two terminals (from the repo root, with the venv active):

```bash
uvicorn app.main:app --port 8000          # API
streamlit run ui/streamlit_app.py         # page at http://localhost:8501 (API_URL defaults to http://localhost:8000)
```

Upload a PDF, PNG or JPG (max 10 MB), or pick a synthetic test document under **Or try a sample** in the sidebar, then press **Check document**. The page shows the eligibility status, the five fields, the masking summary, the stage timings and the response JSON, which you can download.

| Endpoint | Returns |
|---|---|
| `POST /v1/extract` (multipart field `file`) | `200` with the §6 response JSON |
| `GET /healthz` | `{"status": "ok"}` |
| `GET /v1/telemetry/last` | allowlisted summary of the latest request, or `null` |

Errors return only `{"error": "Could not read this document"}`: `413` too large, `415` not a PDF/PNG/JPG, `400` empty, `422` unreadable or missing `file`, `502` AI step failed, `500` anything else. CORS allows `http://localhost:8501` plus any comma-separated `CORS_ORIGIN` values.

## Deviations from the architecture so far

- Tesseract runs with `--psm 6` (single uniform block), not the default `--psm 3`. Mark sheets and ID cards are two-column (`Label: value`); `--psm 3` read all labels and then all values, which broke label-anchored masking and scrambled the text for the LLM. `--psm 6` cut masking misses from 5 documents to 2 (then 0 with two new recognizers) for about 2 points of mean confidence.
- The Indian phone entity is `IN_PHONE`, not `PHONE_NUMBER` as in the §6 example.
- Presidio logs raw text at DEBUG; its loggers are pinned to WARNING and do not propagate (found by the leak test).
- A `vision` telemetry span covers image redaction on the fallback path.
- Near the cutoff (58–62) a percentage is confirmed only by a printed total that agrees with it; high OCR confidence alone is not enough.
- A stamp overlapping the percentage or total line always gives Needs review, even when stamp removal recovered the digits cleanly (the phase 5 "a stamped total gives Needs review" rule). Seals are detected as roughly square red/blue regions under 60% of the page.
- The subject recompute (stretch, phase 10) is in: at least 3 parsed subject rows that differ from the printed percentage by more than 0.5 give Needs review.
- When the recompute disagrees, `check.printed_percentage` and `check.recomputed_percentage` carry both figures for the operator view. They are omitted otherwise and never reach telemetry.
- Any overall result that is not a percentage (CGPA, SGPA, GPA, grade points, percentile, letter grades, division) gives Needs review and is never converted. A local check on the document's own label backs up the LLM. A figure with no percentage label or % sign also gives Needs review.
- A blank upload, where OCR finds no word boxes, cannot be redacted, so it returns the generic "Could not read this document" error with no LLM call. Its label in `dev_set.csv` is `Could not read`. Pages with no word boxes are also left out of multi-page vision requests.
- On a Needs review result, `fields.percentage` still shows what was read, so the officer can see what to check.
- The LLM request sends no `temperature`: `claude-haiku-5.5` rejects sampling parameters, and with `require_parameters` OpenRouter then finds no provider. `max_tokens` is 2000 with low reasoning effort, because reasoning tokens count against the limit.

## Layout

```
app/          FastAPI service and pipeline (ocr, pii, llm, rules)
ui/           Streamlit page (calls POST /v1/extract)
data/         synthetic sample generator, dev and held-back test samples
eval/         scoring against the labeled test set
tests/        unit tests
docs/         architecture and workflow
```
