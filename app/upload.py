"""Accept an upload: check magic bytes and size, store under a random UUID, delete when done.

The original filename is never taken, so it can never be stored or logged.
"""

from __future__ import annotations

import os
from uuid import uuid4

from app.config import Settings, get_settings
from app.errors import ErrorClass, PipelineError
from app.schemas import StoredUpload

_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "pdf"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
)


def detect_kind(head: bytes) -> str | None:
    """Return "pdf", "png" or "jpg" from the leading bytes, or None if unrecognised."""
    for magic, kind in _MAGIC:
        if head.startswith(magic):
            return kind
    return None


def store_upload(data: bytes, settings: Settings | None = None) -> StoredUpload:
    """Validate and store the bytes as `<upload_dir>/<uuid4hex>.<ext>` with mode 0o600."""
    settings = settings or get_settings()
    if not data:
        raise PipelineError(ErrorClass.UPLOAD_EMPTY)
    if len(data) > settings.max_upload_bytes:
        raise PipelineError(ErrorClass.UPLOAD_TOO_LARGE)
    kind = detect_kind(data[:16])
    if kind is None:
        raise PipelineError(ErrorClass.UPLOAD_BAD_TYPE)

    upload_dir = settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_id = uuid4().hex
    path = upload_dir / f"{upload_id}.{kind}"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return StoredUpload(upload_id=upload_id, path=path, extension=kind, size_bytes=len(data))


def delete_upload(upload: StoredUpload) -> None:
    """Remove the stored file. Safe to call more than once."""
    try:
        upload.path.unlink(missing_ok=True)
    except OSError:
        pass
