"""HTTP API: POST /v1/extract, GET /healthz, GET /v1/telemetry/last (docs/ARCHITECTURE.md §5–8).

Run with: uvicorn app.main:app --port 8000

Privacy rules enforced here:
- The upload's filename is never read, used or logged; only the bytes go to the pipeline.
- The upload is read in chunks with a hard cap of `settings.max_upload_bytes`.
- Every error response carries only the generic message. Error classes go to telemetry, never
  to the client.
- The OTel instrumentation records no request or response bodies, and its spans pass through the
  sanitizing exporter in app.telemetry.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import pipeline
from app import telemetry
from app.config import get_settings
from app.errors import GENERIC_MESSAGE, ErrorClass, PipelineError
from app.schemas import ExtractResponse

DEFAULT_UI_ORIGIN = "http://localhost:8501"
READ_CHUNK_BYTES = 64 * 1024
# Room for multipart boundaries and part headers on top of the file itself.
MULTIPART_OVERHEAD_BYTES = 64 * 1024

STATUS_BY_ERROR_CLASS: dict[ErrorClass, int] = {
    ErrorClass.UPLOAD_TOO_LARGE: 413,
    ErrorClass.UPLOAD_BAD_TYPE: 415,
    ErrorClass.UPLOAD_EMPTY: 400,
    ErrorClass.RENDER_FAILED: 422,
    ErrorClass.OCR_FAILED: 422,
    ErrorClass.OCR_EMPTY: 422,
    ErrorClass.PII_FAILED: 422,
    ErrorClass.LLM_FAILED: 502,
    ErrorClass.SCHEMA_INVALID: 502,
    ErrorClass.INTERNAL: 500,
}


def allowed_origins(cors_origin: str) -> list[str]:
    """The Streamlit page's origin plus any comma-separated CORS_ORIGIN values. Never a wildcard."""
    origins = [DEFAULT_UI_ORIGIN]
    for origin in (o.strip().rstrip("/") for o in cors_origin.split(",")):
        if origin and origin != "*" and origin not in origins:
            origins.append(origin)
    return origins


def generic_error(status_code: int) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": GENERIC_MESSAGE})


async def read_capped(file: UploadFile, cap: int) -> bytes:
    """Read the upload in chunks; stop as soon as it exceeds `cap` bytes."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > cap:
            raise PipelineError(ErrorClass.UPLOAD_TOO_LARGE)
        chunks.append(chunk)
    return b"".join(chunks)


def create_app() -> FastAPI:
    settings = get_settings()
    telemetry.setup_telemetry()

    app = FastAPI(
        title="LearnSphere Admission Document Checker",
        version="1.0.0",
        description="Classifies an admission document, extracts five fields and checks eligibility. "
        "OCR and personal-data masking run locally; only masked text reaches the AI model.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(settings.cors_origin),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def reject_oversized_body(request: Request, call_next):
        """Refuse an upload whose declared size is over the cap before the body is parsed."""
        if request.url.path == "/v1/extract":
            declared = request.headers.get("content-length")
            if declared is not None:
                try:
                    too_big = int(declared) > settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
                except ValueError:
                    return generic_error(400)
                if too_big:
                    return generic_error(413)
        return await call_next(request)

    @app.exception_handler(PipelineError)
    async def pipeline_error_handler(_request: Request, exc: PipelineError) -> JSONResponse:
        return generic_error(STATUS_BY_ERROR_CLASS.get(exc.error_class, 500))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default body echoes the offending input; send only the generic message.
        return generic_error(422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": GENERIC_MESSAGE}, headers=exc.headers)

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, _exc: Exception) -> JSONResponse:
        return generic_error(500)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/extract", response_model=ExtractResponse)
    async def extract(file: UploadFile = File(...)) -> ExtractResponse:
        # Only the bytes are used. file.filename and file.content_type are deliberately ignored.
        try:
            data = await read_capped(file, settings.max_upload_bytes)
        finally:
            await file.close()
        try:
            return await run_in_threadpool(pipeline.extract, data)
        except PipelineError:
            raise
        except Exception:
            # Handled here rather than by the app-wide handler, so the server never logs a
            # traceback for it; the pipeline has already recorded an error class.
            return generic_error(500)  # type: ignore[return-value]

    @app.get("/v1/telemetry/last")
    async def telemetry_last() -> dict[str, Any] | None:
        """Allowlisted summary of the latest finished request, for the UI telemetry panel."""
        return telemetry.last_request_summary()

    # Request/response bodies are never recorded by this instrumentation; skipping the ASGI
    # receive/send sub-spans keeps the trace to one server span per request. Header capture
    # stays off (no OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_* is set by this app), and the
    # sanitizing exporter drops any attribute outside the allowlist.
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=telemetry.get_tracer_provider(),
        meter_provider=telemetry.get_meter_provider(),
        exclude_spans=["receive", "send"],
    )
    return app


app = create_app()
