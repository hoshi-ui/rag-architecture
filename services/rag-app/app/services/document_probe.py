import io
import math
import os
import re
import shutil
import zipfile
from typing import Any, Callable, Dict, List, Optional
from xml.etree import ElementTree

from fastapi import HTTPException

from app.documents import common as document_common
from app.documents import ir as document_ir


ModuleAvailable = Callable[[str], bool]


class DocumentProbeAdapter:
    def __init__(
        self,
        *,
        module_available: ModuleAvailable,
        safe_filename: Callable[[str], str],
        looks_like_cid_garbled_text: Callable[[str], bool],
        ocr_service_url: str = "",
        pdf_ocr_max_text_chars_per_page: float = 300.0,
        max_file_size_mb: int = 100,
        max_pdf_pages: int = 300,
        max_image_pixels: int = 40000000,
        max_xlsx_rows: int = 20000,
        max_xlsx_cols: int = 200,
        max_xlsx_sheets: int = 50,
    ) -> None:
        self.module_available = module_available
        self.safe_filename = safe_filename
        self.looks_like_cid_garbled_text = looks_like_cid_garbled_text
        self.ocr_service_url = ocr_service_url
        self.pdf_ocr_max_text_chars_per_page = float(pdf_ocr_max_text_chars_per_page)
        self.max_file_size_mb = int(max_file_size_mb)
        self.max_pdf_pages = int(max_pdf_pages)
        self.max_image_pixels = int(max_image_pixels)
        self.max_xlsx_rows = int(max_xlsx_rows)
        self.max_xlsx_cols = int(max_xlsx_cols)
        self.max_xlsx_sheets = int(max_xlsx_sheets)


def _probe_context(runtime: Any) -> DocumentProbeAdapter:
    if isinstance(runtime, DocumentProbeAdapter):
        return runtime
    raise TypeError("document probe operations require DocumentProbeAdapter")


def detect_parser_capabilities(module_available: ModuleAvailable) -> Dict[str, bool]:
    return {
        "python_magic": module_available("magic"),
        "pymupdf": module_available("fitz"),
        "pymupdf4llm": module_available("pymupdf4llm"),
        "docling": module_available("docling"),
        "unstructured": module_available("unstructured"),
        "paddleocr": module_available("paddleocr"),
        "antiword": shutil.which("antiword") is not None,
    }


def probe_pdf_layout_with_pymupdf(content: bytes, module_available: ModuleAvailable) -> Dict[str, Any]:
    if not module_available("fitz"):
        return {}
    try:
        import fitz  # type: ignore

        document = fitz.open(stream=content, filetype="pdf")
        page_count = getattr(document, "page_count", 0) or 0
        multi_column_pages = 0
        table_dense_pages = 0
        for page in document:
            blocks = [block for block in (page.get_text("blocks") or []) if len(block) >= 5 and str(block[4]).strip()]
            if blocks:
                left = sum(1 for block in blocks if float(block[0]) <= (page.rect.width * 0.45))
                right = sum(1 for block in blocks if float(block[0]) >= (page.rect.width * 0.55))
                if left and right:
                    multi_column_pages += 1
                dense = sum(
                    1
                    for block in blocks
                    if ("|" in str(block[4]))
                    or ("\t" in str(block[4]))
                    or re.search(r"\S+\s{2,}\S+", str(block[4]))
                )
                if dense / max(1, len(blocks)) >= 0.25:
                    table_dense_pages += 1
        document.close()
        return {
            "multi_column": bool(page_count and multi_column_pages >= max(1, math.ceil(page_count / 2))),
            "table_dense": bool(page_count and table_dense_pages >= max(1, math.ceil(page_count / 3))),
            "layout_backend": "pymupdf",
        }
    except Exception:
        return {}


def probe_pdf_document(runtime: Any, content: bytes) -> Dict[str, Any]:
    context = _probe_context(runtime)
    try:
        from pypdf import PdfReader
    except ImportError:
        return {}
    try:
        reader = PdfReader(io.BytesIO(content))
        if getattr(reader, "is_encrypted", False):
            return {"page_count": len(reader.pages), "encrypted": True}
        text_lengths: List[int] = []
        image_pages = 0
        scanned_pages = 0
        garbled_pages = 0
        for page in reader.pages:
            page_text = (page.extract_text() or "").strip()
            text_lengths.append(len(page_text))
            if context.looks_like_cid_garbled_text(page_text):
                garbled_pages += 1
            image_count = 0
            try:
                image_count = len(getattr(page, "images", []) or [])
            except Exception:
                image_count = 0
            if image_count > 0:
                image_pages += 1
            if len(page_text) < 40 and image_count > 0:
                scanned_pages += 1
        page_count = len(text_lengths)
        layout_probe = probe_pdf_layout_with_pymupdf(content, context.module_available)
        return {
            "page_count": page_count,
            "is_scanned_pdf": bool(page_count and scanned_pages >= max(1, math.ceil(page_count * 0.6))),
            "image_page_majority": bool(page_count and image_pages >= max(1, math.ceil(page_count * 0.5))),
            "garbled_text_pages": garbled_pages,
            "garbled_text_majority": bool(page_count and garbled_pages >= max(1, math.ceil(page_count * 0.3))),
            "avg_text_chars_per_page": (sum(text_lengths) / page_count) if page_count else 0.0,
            **layout_probe,
        }
    except Exception:
        return {}


def probe_docx_document(content: bytes) -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = zf.namelist()
            details["has_header"] = any(name.startswith("word/header") for name in names)
            details["has_footer"] = any(name.startswith("word/footer") for name in names)
            details["has_footnotes"] = "word/footnotes.xml" in names
            details["has_endnotes"] = "word/endnotes.xml" in names
            details["has_comments"] = "word/comments.xml" in names
            details["image_count"] = sum(1 for name in names if name.startswith("word/media/"))
            if "word/document.xml" in names:
                root = ElementTree.fromstring(zf.read("word/document.xml"))
                details["revision_insertions"] = len(root.findall(f".//{{{document_ir.DOCX_NS}}}ins"))
                details["revision_deletions"] = len(root.findall(f".//{{{document_ir.DOCX_NS}}}del"))
    except Exception:
        pass
    return details


def probe_xlsx_document(content: bytes) -> Dict[str, Any]:
    try:
        import openpyxl
    except ImportError:
        return {}
    try:
        workbook = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=False)
        sheet_count = len(workbook.worksheets)
        table_count = 0
        max_rows = 0
        max_cols = 0
        for sheet in workbook.worksheets:
            try:
                table_count += len(getattr(sheet, "tables", {}) or {})
            except Exception:
                pass
            try:
                max_rows = max(max_rows, int(getattr(sheet, "max_row", 0) or 0))
                max_cols = max(max_cols, int(getattr(sheet, "max_column", 0) or 0))
            except Exception:
                pass
        workbook.close()
        return {"sheet_count": sheet_count, "table_count": table_count, "max_rows": max_rows, "max_cols": max_cols}
    except Exception:
        return {}


def probe_image_dimensions(content: bytes, module_available: ModuleAvailable) -> Dict[str, Any]:
    if not module_available("PIL"):
        return {}
    try:
        from PIL import Image  # type: ignore

        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
        return {
            "image_width": int(width),
            "image_height": int(height),
            "image_pixels": int(width) * int(height),
        }
    except Exception:
        return {}


def choose_pdf_fast_backend(capabilities: Dict[str, bool]) -> str:
    if capabilities.get("pymupdf4llm"):
        return "pymupdf4llm"
    if capabilities.get("pymupdf"):
        return "pymupdf"
    if capabilities.get("docling"):
        return "docling"
    return "pypdf"


def choose_ocr_backend(runtime: Any, capabilities: Dict[str, bool]) -> Optional[str]:
    context = _probe_context(runtime)
    if (context.ocr_service_url or "").strip():
        return "external_http_ocr"
    return None


def ocr_backend_candidates(runtime: Any, capabilities: Dict[str, bool]) -> List[str]:
    context = _probe_context(runtime)
    candidates: List[str] = []
    if (context.ocr_service_url or "").strip():
        candidates.append("external_http_ocr")
    candidates.extend([name for name in ("docling", "unstructured", "paddleocr") if capabilities.get(name)])
    return candidates


def should_route_pdf_to_ocr(runtime: Any, probe: Dict[str, Any]) -> bool:
    context = _probe_context(runtime)
    if probe.get("is_scanned_pdf"):
        return True
    if probe.get("garbled_text_majority"):
        return True
    if not probe.get("image_page_majority"):
        return False
    try:
        avg_text_chars = float(probe.get("avg_text_chars_per_page") or 0.0)
    except Exception:
        avg_text_chars = 0.0
    return avg_text_chars <= float(context.pdf_ocr_max_text_chars_per_page)


def route_document_parser(runtime: Any, probe: Dict[str, Any]) -> Dict[str, Any]:
    context = _probe_context(runtime)
    capabilities = detect_parser_capabilities(context.module_available)
    detected_ext = (probe.get("detected_ext") or probe.get("extension") or "").lower()
    mime_type = (probe.get("mime_type") or "").lower()
    image_like = detected_ext in document_common.IMAGE_EXTENSIONS or mime_type.startswith("image/")
    if detected_ext == ".pdf":
        if should_route_pdf_to_ocr(runtime, probe):
            ocr_backend = choose_ocr_backend(runtime, capabilities)
            return {
                "route": "pdf_ocr_layout",
                "parser_backend": ocr_backend or "fallback_no_ocr_backend",
                "backend_candidates": ocr_backend_candidates(runtime, capabilities),
                "degraded": ocr_backend is None,
            }
        return {
            "route": "pdf_digital_fast",
            "parser_backend": choose_pdf_fast_backend(capabilities),
            "backend_candidates": [
                name
                for name in ("pymupdf4llm", "pymupdf", "docling", "pypdf")
                if capabilities.get(name) or name == "pypdf"
            ],
            "degraded": False,
        }
    if image_like:
        ocr_backend = choose_ocr_backend(runtime, capabilities)
        return {
            "route": "image_ocr_layout",
            "parser_backend": ocr_backend or "fallback_no_ocr_backend",
            "backend_candidates": ocr_backend_candidates(runtime, capabilities),
            "degraded": ocr_backend is None,
        }
    if detected_ext == ".docx":
        return {
            "route": "docx_structured",
            "parser_backend": "python-docx",
            "backend_candidates": [name for name in ("docling", "python-docx") if name == "python-docx" or capabilities.get("docling")],
            "degraded": False,
        }
    if detected_ext == ".doc":
        return {
            "route": "doc_legacy",
            "parser_backend": "antiword",
            "backend_candidates": ["antiword"],
            "degraded": not capabilities.get("antiword"),
        }
    if detected_ext == ".xlsx":
        return {
            "route": "xlsx_structured",
            "parser_backend": "openpyxl",
            "backend_candidates": [name for name in ("docling", "openpyxl") if name == "openpyxl" or capabilities.get("docling")],
            "degraded": False,
        }
    if detected_ext == ".csv":
        return {"route": "csv_structured", "parser_backend": "csv", "backend_candidates": ["csv"], "degraded": False}
    if detected_ext == ".json":
        return {"route": "json_structured", "parser_backend": "json", "backend_candidates": ["json"], "degraded": False}
    return {"route": "plain_text", "parser_backend": "text", "backend_candidates": ["text"], "degraded": False}


def probe_file_for_parser(runtime: Any, filename: str, content: bytes) -> Dict[str, Any]:
    context = _probe_context(runtime)
    safe_name = context.safe_filename(filename)
    extension = os.path.splitext(safe_name)[1].lower()
    signature = document_common.sniff_file_signature(content)
    detected_ext = (signature.get("suggested_ext") or extension or "").lower()
    mime_type = document_common.sniff_mime_type(safe_name, content, detected_ext, module_available=context.module_available)
    probe: Dict[str, Any] = {
        "filename": safe_name,
        "extension": extension,
        "detected_ext": detected_ext or extension,
        "mime_type": mime_type,
        "signature": signature.get("label") or "unknown",
        "file_size": len(content or b""),
        "page_count": None,
        "sheet_count": None,
        "is_scanned_pdf": False,
        "image_page_majority": False,
        "multi_column": False,
        "table_dense": False,
    }
    if probe["detected_ext"] == ".pdf":
        probe.update(probe_pdf_document(runtime, content))
    elif probe["detected_ext"] == ".docx":
        probe.update(probe_docx_document(content))
    elif probe["detected_ext"] == ".xlsx":
        probe.update(probe_xlsx_document(content))
    elif probe["detected_ext"] in document_common.IMAGE_EXTENSIONS:
        probe["is_scanned_pdf"] = True
        probe["image_page_majority"] = True
        probe["page_count"] = 1
        probe.update(probe_image_dimensions(content, context.module_available))
    probe.update(route_document_parser(runtime, probe))
    return probe


def probe_mime_allowed(detected_ext: str, mime_type: str) -> bool:
    allowed = document_common.ALLOWED_MIME_BY_EXTENSION.get(document_common.canonical_extension(detected_ext), set())
    mime = (mime_type or "").strip().lower()
    if not allowed or not mime:
        return True
    if mime in allowed:
        return True
    if document_common.canonical_extension(detected_ext) in document_common.TEXT_LIKE_EXTENSIONS and mime.startswith("text/"):
        return True
    return False


def build_text_upload_probe(runtime: Any, filename: str, content: str) -> Dict[str, Any]:
    context = _probe_context(runtime)
    safe_name = context.safe_filename(filename)
    extension = os.path.splitext(safe_name)[1].lower()
    detected_ext = extension if extension in document_common.TEXT_LIKE_EXTENSIONS else ".txt"
    body = (content or "").encode("utf-8")
    return {
        "filename": safe_name,
        "extension": extension,
        "detected_ext": detected_ext,
        "mime_type": document_common.sniff_mime_type(safe_name, body, detected_ext, module_available=context.module_available),
        "signature": "text",
        "file_size": len(body),
        "page_count": 1,
        "route": "plain_text",
        "parser_backend": "direct_text",
        "degraded": False,
    }



def validate_upload_probe(runtime: Any, filename: str, probe: Dict[str, Any], is_text_upload: bool = False):
    context = _probe_context(runtime)
    safe_name = context.safe_filename(filename)
    extension = os.path.splitext(safe_name)[1].lower()
    detected_ext = (probe.get("detected_ext") or extension or "").lower()
    file_size = int(probe.get("file_size") or 0)
    if file_size <= 0:
        raise HTTPException(status_code=400, detail="empty_file")
    if file_size > int(context.max_file_size_mb) * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"file_too_large: max {context.max_file_size_mb}MB")
    if detected_ext not in document_common.SUPPORTED_FILE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"unsupported_file_type: {detected_ext or extension or '<none>'}")

    suspicious_reasons: List[str] = []
    if extension and not document_common.extension_matches_detected(extension, detected_ext):
        suspicious_reasons.append("ext_magic_conflict")
    if not probe_mime_allowed(detected_ext, str(probe.get("mime_type") or "")):
        suspicious_reasons.append("mime_mismatch")
    if str(probe.get("signature") or "") == "binary" and detected_ext in document_common.TEXT_LIKE_EXTENSIONS:
        suspicious_reasons.append("binary_payload_for_text_type")
    if suspicious_reasons:
        raise HTTPException(status_code=400, detail=f"suspicious_file_type: {', '.join(suspicious_reasons)}")

    if detected_ext == ".pdf" and probe.get("encrypted"):
        raise HTTPException(status_code=400, detail="encrypted_file")
    if detected_ext == ".pdf" and int(probe.get("page_count") or 0) > int(context.max_pdf_pages):
        raise HTTPException(status_code=400, detail=f"pdf_page_limit_exceeded: max {context.max_pdf_pages}")
    if detected_ext == ".xlsx":
        if int(probe.get("sheet_count") or 0) > int(context.max_xlsx_sheets):
            raise HTTPException(status_code=400, detail=f"xlsx_sheet_limit_exceeded: max {context.max_xlsx_sheets}")
        if int(probe.get("max_rows") or 0) > int(context.max_xlsx_rows):
            raise HTTPException(status_code=400, detail=f"xlsx_row_limit_exceeded: max {context.max_xlsx_rows}")
        if int(probe.get("max_cols") or 0) > int(context.max_xlsx_cols):
            raise HTTPException(status_code=400, detail=f"xlsx_col_limit_exceeded: max {context.max_xlsx_cols}")
    if detected_ext in document_common.IMAGE_EXTENSIONS:
        image_pixels = int(probe.get("image_pixels") or 0)
        if image_pixels and image_pixels > int(context.max_image_pixels):
            raise HTTPException(status_code=400, detail=f"image_pixel_limit_exceeded: max {context.max_image_pixels}")
    if is_text_upload and detected_ext not in document_common.SUPPORTED_FILE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="unsupported_text_upload")
