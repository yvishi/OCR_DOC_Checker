import io
import re
import stat

import pytest
from PIL import Image

from app.config import Settings
from app.errors import GENERIC_MESSAGE, ErrorClass, PipelineError
from app.upload import delete_upload, detect_kind, store_upload


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buf, format="PNG")
    return buf.getvalue()


def _jpg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buf, format="JPEG")
    return buf.getvalue()


PDF_BYTES = b"%PDF-1.7\n%fake body\n"


@pytest.fixture
def settings(tmp_path):
    return Settings(upload_dir=tmp_path / "uploads")


@pytest.mark.parametrize(
    "data,kind",
    [(PDF_BYTES, "pdf"), (_png_bytes(), "png"), (_jpg_bytes(), "jpg")],
)
def test_detect_kind(data, kind):
    assert detect_kind(data) == kind


@pytest.mark.parametrize("head", [b"", b"hello", b"GIF89a", b"PK\x03\x04", b"%PD", b"\x89PNG"])
def test_detect_kind_unknown(head):
    assert detect_kind(head) is None


def test_text_named_png_is_rejected(settings, tmp_path):
    fake = tmp_path / "marksheet.png"
    fake.write_text("this is not an image")
    with pytest.raises(PipelineError) as err:
        store_upload(fake.read_bytes(), settings)
    assert err.value.error_class is ErrorClass.UPLOAD_BAD_TYPE
    assert str(err.value) == GENERIC_MESSAGE


def test_empty_rejected(settings):
    with pytest.raises(PipelineError) as err:
        store_upload(b"", settings)
    assert err.value.error_class is ErrorClass.UPLOAD_EMPTY


def test_oversize_rejected(tmp_path):
    small = Settings(upload_dir=tmp_path, max_upload_bytes=100)
    with pytest.raises(PipelineError) as err:
        store_upload(PDF_BYTES + b"x" * 200, small)
    assert err.value.error_class is ErrorClass.UPLOAD_TOO_LARGE
    assert not any(tmp_path.iterdir())


def test_exactly_max_is_accepted(tmp_path):
    data = PDF_BYTES + b"x" * 10
    up = store_upload(data, Settings(upload_dir=tmp_path, max_upload_bytes=len(data)))
    assert up.size_bytes == len(data)


@pytest.mark.parametrize("data,ext", [(PDF_BYTES, "pdf"), (_png_bytes(), "png"), (_jpg_bytes(), "jpg")])
def test_stored_as_uuid_plus_ext(settings, data, ext):
    up = store_upload(data, settings)
    assert up.extension == ext
    assert re.fullmatch(r"[0-9a-f]{32}", up.upload_id)
    assert up.path.name == f"{up.upload_id}.{ext}"
    assert up.path.parent == settings.upload_dir
    assert up.path.read_bytes() == data
    assert up.size_bytes == len(data)
    assert stat.S_IMODE(up.path.stat().st_mode) == 0o600
    assert "marksheet" not in str(up.path)


def test_unique_ids(settings):
    a = store_upload(PDF_BYTES, settings)
    b = store_upload(PDF_BYTES, settings)
    assert a.upload_id != b.upload_id


def test_delete_is_idempotent(settings):
    up = store_upload(PDF_BYTES, settings)
    delete_upload(up)
    assert not up.path.exists()
    delete_upload(up)  # second call must not raise
