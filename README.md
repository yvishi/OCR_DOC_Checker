# OCR_DOC_Checker

Admission document checker for **LearnSphere College** (fictional), Build Fast with AI FDE Fellowship, Week 2, Team Alpha.

An admissions officer uploads a PDF or image. The app classifies it as a mark sheet, student ID card or fee receipt, extracts five fields, and checks mark sheets against a 60% cutoff. OCR and personal-data masking run locally; only masked text ever reaches the LLM.

> **Synthetic data only.** No real mark sheets, IDs or receipts belong in this repo.

Status: **increment 1** (build phases 0–3): synthetic data, upload + OCR + preprocessing, PII masking, telemetry. Next: phase 4, the LLM call through OpenRouter. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the workflow and stack.

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
python -m data.generate                                   # 20 dev + 20 test synthetic documents, CSVs
python -m app.ocr data/samples/test/test_009_mark_sheet.png          # OCR stats only
python -m app.pipeline data/samples/test/test_001_mark_sheet.png --show-masked   # what the LLM will see
pytest                                                    # unit tests + telemetry leak test over all samples
```

## Deviations from the architecture so far

- Tesseract runs with `--psm 6` (single uniform block), not the default `--psm 3`. Mark sheets and ID cards are two-column (`Label: value`); `--psm 3` read all labels and then all values, which broke label-anchored masking and scrambled the text for the LLM. `--psm 6` cut masking misses from 5 documents to 2 (then 0 with two new recognizers) for about 2 points of mean confidence.
- The Indian phone entity is `IN_PHONE`, not `PHONE_NUMBER` as in the §6 example.
- Presidio logs raw text at DEBUG; its loggers are pinned to WARNING and do not propagate (found by the leak test).
- A `vision` telemetry span covers image redaction on the fallback path.

## Layout

```
app/          FastAPI service and pipeline (ocr, pii, llm, rules)
ui/           Streamlit page (calls POST /v1/extract)
data/         synthetic sample generator, dev and held-back test samples
eval/         scoring against the labeled test set
tests/        unit tests
docs/         architecture and workflow
```
