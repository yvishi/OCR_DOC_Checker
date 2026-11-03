# OCR_DOC_Checker

Admission document checker for **LearnSphere College** (fictional), Build Fast with AI FDE Fellowship, Week 2, Team Alpha.

An admissions officer uploads a PDF or image. The app classifies it as a mark sheet, student ID card or fee receipt, extracts five fields, and checks mark sheets against a 60% cutoff. OCR and personal-data masking run locally; only masked text ever reaches the LLM.

> **Synthetic data only.** No real mark sheets, IDs or receipts belong in this repo.

Status: project scaffold. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the workflow and stack.

## Setup (Windows)

1. Install Python 3.12 and Tesseract OCR (UB Mannheim build).
2. Create the environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m spacy download en_core_web_lg
   ```
3. Copy `.env.example` to `.env` and fill in the values. The LLM API key is read from the `LLM_API_KEY` environment variable; `.env` is never committed.

## Layout

```
app/          FastAPI service and pipeline (ocr, pii, llm, rules)
ui/           Streamlit page (calls POST /v1/extract)
data/         synthetic sample generator, dev and held-back test samples
eval/         scoring against the labeled test set
tests/        unit tests
docs/         architecture and workflow
```
