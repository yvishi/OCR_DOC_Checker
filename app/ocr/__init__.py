"""Local OCR: render pages, preprocess (rotation, stamps, binarization), run Tesseract."""

from app.ocr.tesseract import ocr_image, passes_text_gate, run_ocr

__all__ = ["ocr_image", "passes_text_gate", "run_ocr"]
