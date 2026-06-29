import asyncio
import contextlib
import importlib.util
import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, UploadFile

from app.config import Config, logger as app_logger, resolve_legacy_upload_dir, resolve_runtime_database_dir, resolve_runtime_upload_dir
from app.documents import chunking as document_chunking
from app.documents import clause_metadata as document_clause_metadata
from app.documents import common as document_common
from app.documents import ir as document_ir_helpers
from app.documents import ir_store as document_ir_store
from app.documents import parser as document_parser
from app.documents import parser_docx as document_parser_docx
from app.documents import parser_ocr as document_parser_ocr
from app.documents import parser_pdf as document_parser_pdf
from app.documents import parser_visual as document_parser_visual
from app.documents import parser_xlsx as document_parser_xlsx
from app.documents import profile as document_profile
from app.schemas import DocumentRequest
from app.services import document_probe
from app.services.document_indexing_service import DocumentIndexingService
from app.services import document_lifecycle_service
from app.services import document_metadata
from app.storage import milvus as milvus_storage
from app.storage import sqlite as sqlite_storage
from app.storage import task_store as task_store_storage
from app.utils import files as file_utils
from app.utils.text import sanitize_index_text


_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_UPLOAD_DIR = resolve_runtime_upload_dir(_BASE_DIR)
_LEGACY_UPLOAD_DIR = resolve_legacy_upload_dir(_BASE_DIR)
_LEXICAL_DB_FILE = os.path.join(resolve_runtime_database_dir(_BASE_DIR), "lexical_index.db")
_SOURCE_LOCKS: Dict[str, threading.Lock] = {}
_SOURCE_ASYNC_TASKS: Dict[str, set] = {}


def _public_task_status(status: Optional[str]) -> str:
    return document_metadata.public_task_status(status)


def _chunk_plain_display_text(text: str) -> str:
    return document_metadata.chunk_plain_display_text(text)


def _document_detail_plain_text(document_ir: Optional[Dict[str, Any]], chunks: Any) -> str:
    return document_metadata.document_detail_plain_text(document_ir, chunks)


def _summarize_chunk_payload(payload: Any) -> Dict[str, Any]:
    return document_metadata.summarize_chunk_payload(payload)


def _milvus_text_value(text: str) -> str:
    value = str(text or "")
    max_bytes = int(getattr(Config, "MILVUS_TEXT_MAX_BYTES", 60000) or 60000)
    if max_bytes <= 0 or len(value.encode("utf-8")) <= max_bytes:
        return value
    suffix = "\n[TRUNCATED_FOR_MILVUS_TEXT_LIMIT]"
    budget = max(0, max_bytes - len(suffix.encode("utf-8")))
    return value.encode("utf-8")[:budget].decode("utf-8", "ignore").rstrip() + suffix


def _milvus_varchar_value(text: Any, max_bytes: int) -> str:
    value = str(text or "").strip()
    if max_bytes <= 0 or len(value.encode("utf-8")) <= max_bytes:
        return value
    return value.encode("utf-8")[:max_bytes].decode("utf-8", "ignore").rstrip()


def _milvus_safe_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return document_metadata.milvus_safe_metadata(metadata)


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _cid_garbled_char_ratio(text: str) -> float:
    raw = (text or "").strip()
    if not raw:
        return 0.0
    meaningful = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", raw)
    if not meaningful:
        return 0.0
    suspicious = re.findall(r"[闁硅棄娲禒瀵告偖娴楠勯柡鍥珪缁楀倿宕]", raw)
    return len(suspicious) / max(1, len(meaningful))


def _looks_like_cid_garbled_text(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    suspicious_count = len(re.findall(r"[闁硅棄娲禒瀵告偖娴楠勯柡鍥珪缁楀倿宕]", raw))
    if suspicious_count < 3:
        return False
    return _cid_garbled_char_ratio(raw) >= 0.08


def _fault_injection_stages() -> set[str]:
    values = []
    for env_name in ("RAG_FAULT_INJECT_STAGE", "LEX_DB_CRASH_INJECT_STAGE"):
        raw = os.getenv(env_name, "")
        if raw:
            values.extend(raw.split(","))
    return {value.strip().lower() for value in values if value and value.strip()}


def _crash_inject(stage: str) -> None:
    if (stage or "").strip().lower() in _fault_injection_stages():
        raise RuntimeError(f"Crash injection at stage: {stage}")


class DocumentService:
    def __init__(self, backend: Any):
        self._tasks = getattr(backend, "tasks", {})
        self._logger = getattr(backend, "logger", None) or app_logger
        self._upload_dir = getattr(backend, "UPLOAD_DIR", _UPLOAD_DIR)
        self._lex_store = getattr(backend, "_lex_store", None) or sqlite_storage.create_lexical_store(_LEXICAL_DB_FILE)
        self._task_store = getattr(backend, "_task_store", None) or task_store_storage.create_task_store(
            self.tasks,
            self._lex_store,
        )
        self._source_locks = getattr(backend, "_SOURCE_LOCKS", _SOURCE_LOCKS)
        self._source_async_tasks = getattr(backend, "_SOURCE_ASYNC_TASKS", _SOURCE_ASYNC_TASKS)
        self._sqlite_write_lock = getattr(backend, "_sqlite_write_lock", threading.RLock())
        self._upload_index_async_lock = getattr(backend, "_upload_index_async_lock", None)
        if self._upload_index_async_lock is None:
            self._upload_index_async_lock = asyncio.Lock()
            try:
                setattr(backend, "_upload_index_async_lock", self._upload_index_async_lock)
            except Exception:
                pass
        vector_candidate = getattr(backend, "vector_db", None)
        self._vector_db = vector_candidate if vector_candidate is not None and not callable(vector_candidate) else None
        self._indexing_service = DocumentIndexingService(self)

    @property
    def tasks(self) -> Dict[str, Dict[str, Any]]:
        return self._tasks

    @property
    def logger(self) -> Any:
        return self._logger

    async def upload_document(self, doc_req: DocumentRequest) -> Dict[str, Any]:
        return await _upload_document_impl(doc_req, self)

    async def upload_document_file(self, file: UploadFile) -> Dict[str, Any]:
        return await _upload_document_file_impl(file, self)

    async def list_documents(self) -> Dict[str, Any]:
        return await document_lifecycle_service.list_documents(self)

    async def delete_document(self, filename: str) -> Any:
        return await document_lifecycle_service.delete_document(filename, self)

    async def retry_task(self, task_id: str) -> Dict[str, Any]:
        return await document_lifecycle_service.retry_task(task_id, self)

    async def get_document_detail(self, filename: str) -> Any:
        return await document_lifecycle_service.get_document_detail(filename, self)

    def build_task_item(self, task_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        item = {"task_id": task_id, **task}
        item["task_status"] = self.public_task_status(task.get("status"))
        item["status"] = item["task_status"]
        if task.get("filename"):
            doc = self.doc_get(task["filename"])
            item["document_status"] = doc.get("status")
            item["searchable"] = bool(self.doc_searchable_flag(task["filename"]))
        return item

    async def list_tasks(self) -> Dict[str, Any]:
        items = [self.build_task_item(tid, task) for tid, task in self.tasks.items()]
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return {"tasks": items}

    async def get_task(self, task_id: str) -> Dict[str, Any]:
        task = self.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="task not found")
        return self.build_task_item(task_id, task)

    @property
    def upload_dir(self) -> str:
        return self._upload_dir

    def lex_db_connect(self) -> Any:
        return self._lex_store.connect()

    def doc_searchable_flag(self, source: str) -> bool:
        return bool(self._lex_store.document_searchable_flag(source))

    def filename_stem(self, source: str) -> str:
        return file_utils.filename_stem(source)

    def same_title_group(self, title: str) -> str:
        return document_profile.same_title_group(title)

    def public_task_status(self, status: Any) -> str:
        return _public_task_status(status)

    def doc_get(self, source: str) -> Dict[str, Any]:
        return self._lex_store.get_document(source)

    def canonical_doc_id_for_source(self, source: str) -> str:
        safe_source = file_utils.normalize_filename_for_match(source)
        if not safe_source:
            return ""
        info = self.doc_get(safe_source)
        group = str(info.get("same_title_group") or "").strip()
        if group:
            return group
        canonical_title = str(info.get("canonical_title") or file_utils.filename_stem(safe_source) or safe_source).strip()
        group = document_profile.same_title_group(canonical_title)
        if group:
            return group
        return document_profile.normalize_reference_text(canonical_title)

    def milvus_source_stats(self) -> Dict[str, Dict[str, Any]]:
        return {}

    def safe_filename(self, filename: str) -> str:
        return file_utils.safe_filename(filename)

    def vector_db(self) -> Any:
        if self._vector_db is None:
            self._vector_db = milvus_storage.create_vector_db(on_after_insert=self.crash_inject)
        return self._vector_db

    def sqlite_write_lock(self) -> Any:
        lock = getattr(self, "_sqlite_write_lock", None)
        return lock if lock is not None else contextlib.nullcontext()

    def upload_index_async_lock(self) -> Any:
        return self._upload_index_async_lock

    def lex_db_get_status(self, source: str) -> Any:
        return self._lex_store.get_status(source)

    def milvus_query_source_chunks(self, vector_db: Any, source: str, limit: int) -> Any:
        return milvus_storage.query_source_chunks(source, vector_db=vector_db, limit=limit)

    def get_active_version(self, source: str) -> Any:
        return self._lex_store.active_version(source)

    def document_ir_store(self) -> document_ir_store.DocumentIRStoreAdapter:
        return document_ir_store.DocumentIRStoreAdapter(
            connect=self._lex_store.connect,
            read_connect=self._lex_store.read_connect,
            active_version=lambda source: self.doc_get(source).get("active_version"),
            pending_version=lambda source: self.doc_get(source).get("pending_version"),
        )

    def profile_store(self) -> document_profile.ProfileStoreAdapter:
        return document_profile.ProfileStoreAdapter(
            doc_profile_upsert=self._lex_store.upsert_profile,
            replace_aliases=self._lex_store.replace_aliases,
            replace_sections=self._lex_store.replace_sections,
            replace_topics=self._lex_store.replace_topics,
            load_document_ir=lambda source, version: document_ir_store.load_document_ir(
                self.document_ir_store(),
                source,
                version,
            )
            or {},
            source_by_content_hash=self._lex_store.source_by_content_hash,
            doc_get=self._lex_store.get_document,
            sources_by_same_title_group=self._lex_store.sources_by_same_title_group,
            alias_rows=self._lex_store.aliases,
            section_titles=self._lex_store.section_titles,
            topic_terms=self._lex_store.topics,
            doc_sources=self._lex_store.document_sources,
            source_visible=lambda source: bool(self._lex_store.document_searchable_flag(source)),
        )

    def probe_context(self) -> document_probe.DocumentProbeAdapter:
        return document_probe.DocumentProbeAdapter(
            module_available=_module_available,
            safe_filename=file_utils.safe_filename,
            looks_like_cid_garbled_text=_looks_like_cid_garbled_text,
            ocr_service_url=Config.OCR_SERVICE_URL,
            pdf_ocr_max_text_chars_per_page=Config.PDF_OCR_MAX_TEXT_CHARS_PER_PAGE,
            max_file_size_mb=Config.MAX_FILE_SIZE_MB,
            max_pdf_pages=Config.MAX_PDF_PAGES,
            max_image_pixels=Config.MAX_IMAGE_PIXELS,
            max_xlsx_rows=Config.MAX_XLSX_ROWS,
            max_xlsx_cols=Config.MAX_XLSX_COLS,
            max_xlsx_sheets=Config.MAX_XLSX_SHEETS,
        )

    def chunking_context(self) -> document_chunking.DocumentChunkingAdapter:
        return document_chunking.DocumentChunkingAdapter(
            normalize_ir_text=document_ir_helpers.normalize_ir_text,
            chapter_heading_title=self.chapter_heading_title,
            clause_heading_label=self.clause_heading_label,
            serialize_ir_element=lambda element: document_ir_helpers.serialize_ir_element(
                element,
                should_skip=self.should_skip_ir_element_for_chunking,
                normalize_section_path=document_chunking.normalize_section_path,
                section_path_label=document_chunking.section_path_label,
            ),
            doc_title_profile=document_profile.doc_title_profile,
            filename_stem=file_utils.filename_stem,
            split_text=lambda text, chunk_size, overlap: document_common.split_text(text, chunk_size, overlap),
            chunk_prev_context_chars=int(getattr(Config, "CHUNK_PREV_CONTEXT_CHARS", 220)),
            chunk_next_context_chars=int(getattr(Config, "CHUNK_NEXT_CONTEXT_CHARS", 220)),
        )

    def chapter_heading_title(self, text: str) -> str:
        value = (text or "").strip()
        if not value:
            return ""
        if value.startswith("# Sheet: "):
            return value[len("# Sheet: ") :].strip()
        if value.startswith("#"):
            return value.lstrip("#").strip()
        if re.match(r"^第[一二三四五六七八九十百千万0-9]+[章节编部分条][\s:：、.-]*", value):
            return value
        return ""

    def clause_heading_label(self, text: str) -> str:
        value = (text or "").strip()
        article_id = document_chunking.extract_article_id(value)
        if article_id and document_chunking.LEGAL_ARTICLE_RE.match(value):
            return article_id
        match = re.match(r"^(第\s*[一二三四五六七八九十百千万零〇两0-9０-９]+(?:\s*[一二三四五六七八九十百千万零〇两0-9０-９]+)*\s*[款项])", value)
        return document_chunking.normalize_article_id(match.group(1)) if match else ""

    def should_skip_ir_element_for_chunking(self, element: Dict[str, Any]) -> bool:
        element_type = (element.get("element_type") or "").strip().lower()
        raw_text = (element.get("text_raw") or "").strip()
        section_path = document_chunking.normalize_section_path(element.get("section_path"))
        if section_path:
            root = section_path[0]
            if root == "toc" or root.startswith("header_") or root.startswith("footer_"):
                return True
        if element_type == "figure":
            payload = element.get("json_payload") or {}
            if not raw_text and isinstance(payload, dict) and payload.get("kind") == "image_block":
                return True
            if re.fullmatch(r"闁哄倸娲﹀﹢鏉块崫鐪?\d+", raw_text):
                return True
        if element_type in {"paragraph", "list_item", "key_value", "caption", "heading", "title"}:
            return self.is_pdf_noise_text(raw_text, page_no=element.get("page_no"))
        return False

    def split_text_with_sections(self, filename: str, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        return document_chunking.split_text_with_sections(
            self.chunking_context(),
            filename,
            text,
            chunk_size,
            overlap,
        )

    def document_ir_to_structured_items(self, document_ir: Dict[str, Any], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        return document_chunking.document_ir_to_structured_items(self.chunking_context(), document_ir, chunk_size, overlap)

    def prepare_structured_items(
        self,
        filename: str,
        text: str,
        document_ir: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return document_chunking.prepare_structured_items(
            self.chunking_context(),
            filename,
            text,
            int(Config.CHUNK_SIZE),
            int(Config.OVERLAP),
            document_ir=document_ir,
        )

    def contextualize_chunk_items(self, filename: str, items: Any) -> Any:
        return document_chunking.contextualize_chunk_items(self.chunking_context(), filename, list(items or []))

    def add_chunk_sql(self, source: str, text: str, section: str, metadata: Dict[str, Any], chunk_id: int) -> None:
        metadata = dict(metadata or {})
        metadata.setdefault("chunk_id", chunk_id)
        article_id = (
            document_chunking.extract_leading_article_id(text)
            or metadata.get("article_id")
            or metadata.get("article_no")
            or document_chunking.extract_leading_article_id(
                metadata.get("article_id"),
                metadata.get("article_no"),
                metadata.get("clause_label"),
                section,
                metadata.get("section"),
                metadata.get("section_title"),
                metadata.get("parent_section_title"),
                text,
                metadata.get("raw_text"),
                metadata.get("content"),
            )
            or ""
        )
        if article_id:
            metadata["article_id"] = article_id
            metadata["article_no"] = article_id
        with self.sqlite_write_lock():
            self._lex_store.add_chunk(
                source,
                text,
                section,
                metadata,
                chunk_id,
                after_meta_insert=lambda: self.crash_inject("after_meta_insert"),
                after_fts_insert=lambda: self.crash_inject("after_fts_insert"),
            )

    def merge_parser_metadata(self, metadata: Optional[Dict[str, Any]], probe: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(metadata or {})
        merged["parser_probe"] = probe
        merged["parser_route"] = probe.get("route")
        merged["parser_backend"] = probe.get("parser_backend")
        return merged

    def is_pdf_noise_text(self, text: str, page_no: Optional[int] = None) -> bool:
        return document_parser_visual.is_pdf_noise_text(text, page_no=page_no)

    def append_text_block_to_ir(
        self,
        document_ir: Dict[str, Any],
        text: str,
        *,
        page_no: Optional[int],
        base_section_path: Optional[list] = None,
        parser_name: Optional[str] = None,
    ) -> None:
        return document_ir_helpers.append_text_block_to_ir(
            document_ir,
            text,
            page_no=page_no,
            base_section_path=base_section_path,
            parser_name=parser_name,
            is_pdf_noise_text=self.is_pdf_noise_text,
        )

    def ocr_parser_context(self) -> document_parser_ocr.OcrParserAdapter:
        return document_parser_ocr.OcrParserAdapter(
            config=Config,
            upload_dir=self._upload_dir,
            logger=self._logger,
            normalize_bbox=document_parser_visual.normalize_bbox,
            safe_float=document_parser_visual.safe_float,
            bbox_layout_metrics=document_parser_visual.bbox_layout_metrics,
            choose_ocr_backend=lambda capabilities: document_probe.choose_ocr_backend(self.probe_context(), capabilities),
            detect_parser_capabilities=lambda: document_probe.detect_parser_capabilities(_module_available),
            safe_filename=file_utils.safe_filename,
            module_available=_module_available,
            new_document_ir=lambda source, **kwargs: document_ir_helpers.new_document_ir(
                source,
                safe_filename=file_utils.safe_filename,
                **kwargs,
            ),
            append_ir_element=document_ir_helpers.append_ir_element,
            build_visual_page_profile=document_parser_visual.build_visual_page_profile,
            normalize_heading_title=document_parser_visual.normalize_heading_title,
            looks_like_toc_title=document_parser_visual.looks_like_toc_title,
            should_exit_visual_toc=lambda text, next_text, layout, page_profile: document_parser_visual.should_exit_visual_toc(
                text,
                next_text,
                layout,
                page_profile,
                clause_heading_label=self.clause_heading_label,
            ),
            law_semantic_truncation_label=document_parser_visual.law_semantic_truncation_label,
            infer_visual_heading_level=lambda text, layout, page_profile, next_text="": document_parser_visual.infer_visual_heading_level(
                text,
                layout,
                page_profile,
                next_text,
                clause_heading_label=self.clause_heading_label,
            ),
            clause_heading_label=self.clause_heading_label,
        )

    def pdf_parser_context(self) -> document_parser_pdf.PdfParserAdapter:
        return document_parser_pdf.PdfParserAdapter(
            module_available=_module_available,
            new_document_ir=lambda source, **kwargs: document_ir_helpers.new_document_ir(
                source,
                safe_filename=file_utils.safe_filename,
                **kwargs,
            ),
            append_ir_element=document_ir_helpers.append_ir_element,
            append_text_block_to_ir=self.append_text_block_to_ir,
            safe_float=document_parser_visual.safe_float,
            is_pdf_noise_text=self.is_pdf_noise_text,
            normalize_bbox=document_parser_visual.normalize_bbox,
            bbox_layout_metrics=document_parser_visual.bbox_layout_metrics,
            median_number=document_parser_visual.median_number,
            pdf_page_quality_profile=document_parser_visual.pdf_page_quality_profile,
            pdf_page_should_use_ocr=lambda page_profile: document_parser_visual.pdf_page_should_use_ocr(
                page_profile,
                int(Config.PDF_OCR_MAX_TEXT_CHARS_PER_PAGE),
            ),
            ocr_compensate_pdf_page=lambda page, page_no, probe: document_parser_ocr.ocr_compensate_pdf_page(
                self.ocr_parser_context(),
                page,
                page_no,
                probe,
            ),
            summarize_ocr_meta=document_parser_ocr.summarize_ocr_meta,
            build_visual_page_profile=document_parser_visual.build_visual_page_profile,
            normalize_heading_title=document_parser_visual.normalize_heading_title,
            looks_like_toc_title=document_parser_visual.looks_like_toc_title,
            should_exit_visual_toc=lambda text, next_text, layout, page_profile: document_parser_visual.should_exit_visual_toc(
                text,
                next_text,
                layout,
                page_profile,
                clause_heading_label=self.clause_heading_label,
            ),
            pdf_is_toc_entry_text=document_parser_visual.pdf_is_toc_entry_text,
            law_semantic_truncation_label=document_parser_visual.law_semantic_truncation_label,
            is_visual_title_candidate=lambda text, layout, page_profile, page_no, has_elements: document_parser_visual.is_visual_title_candidate(
                text,
                layout,
                page_profile,
                page_no,
                has_elements,
                clause_heading_label=self.clause_heading_label,
            ),
            infer_visual_heading_level=lambda text, layout, page_profile, next_text="": document_parser_visual.infer_visual_heading_level(
                text,
                layout,
                page_profile,
                next_text,
                clause_heading_label=self.clause_heading_label,
            ),
            clause_heading_label=self.clause_heading_label,
            remove_temp_path=document_parser_ocr.remove_temp_path,
            render_pdf_pages_for_ocr=lambda content: document_parser_ocr.render_pdf_pages_for_ocr(
                self.ocr_parser_context(),
                content,
            ),
            call_external_ocr=lambda image_path: document_parser_ocr.call_external_ocr(
                self.ocr_parser_context(),
                image_path,
            ),
            extract_ocr_lines=lambda payload: document_parser_ocr.extract_ocr_lines(
                self.ocr_parser_context(),
                payload,
            ),
            extract_ocr_texts=lambda payload: document_parser_ocr.extract_ocr_texts(
                self.ocr_parser_context(),
                payload,
            ),
            build_ocr_document_ir=lambda filename, metadata, doc_version, parser_name, probe, page_results, empty_notice: document_parser_ocr.build_ocr_document_ir(
                self.ocr_parser_context(),
                filename,
                metadata,
                doc_version,
                parser_name,
                probe,
                page_results,
                empty_notice,
            ),
            safe_filename=file_utils.safe_filename,
            logger=self._logger,
        )

    def parse_pdf_fast_document_ir(
        self,
        filename: str,
        content: bytes,
        metadata: Optional[Dict[str, Any]],
        doc_version: Optional[int],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return document_parser_pdf.parse_pdf_fast_document_ir(
            self.pdf_parser_context(),
            filename,
            content,
            metadata,
            doc_version,
            kwargs.get("backend") or "pypdf",
        )

    def parse_pdf_ocr_document_ir(
        self,
        filename: str,
        content: bytes,
        metadata: Optional[Dict[str, Any]],
        doc_version: Optional[int],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        probe = args[0] if args else kwargs.get("probe")
        return document_parser_pdf.parse_pdf_ocr_document_ir(
            self.pdf_parser_context(),
            filename,
            content,
            metadata,
            doc_version,
            dict(probe or {}),
        )

    def parse_image_document_ir(
        self,
        filename: str,
        content: bytes,
        metadata: Optional[Dict[str, Any]],
        doc_version: Optional[int],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        probe = args[0] if args else kwargs.get("probe")
        return document_parser_ocr.parse_image_document_ir(
            self.ocr_parser_context(),
            filename,
            content,
            metadata,
            doc_version,
            dict(probe or {}),
        )

    def docx_parser_context(self) -> document_parser_docx.DocxParserAdapter:
        return document_parser_docx.DocxParserAdapter(
            clause_heading_label=self.clause_heading_label,
            chapter_heading_title=self.chapter_heading_title,
            build_document_ir_from_text=self.build_document_ir_from_text,
            new_document_ir=lambda source, **kwargs: document_ir_helpers.new_document_ir(
                source,
                safe_filename=file_utils.safe_filename,
                **kwargs,
            ),
            append_ir_element=document_ir_helpers.append_ir_element,
            normalize_ir_text=document_ir_helpers.normalize_ir_text,
            append_text_block_to_ir=self.append_text_block_to_ir,
        )

    def xlsx_parser_context(self) -> document_parser_xlsx.XlsxParserAdapter:
        return document_parser_xlsx.XlsxParserAdapter(
            new_document_ir=lambda source, **kwargs: document_ir_helpers.new_document_ir(
                source,
                safe_filename=file_utils.safe_filename,
                **kwargs,
            ),
            append_ir_element=document_ir_helpers.append_ir_element,
            normalize_ir_text=document_ir_helpers.normalize_ir_text,
        )

    def parse_doc_document_ir(
        self,
        filename: str,
        content: bytes,
        metadata: Optional[Dict[str, Any]],
        doc_version: Optional[int],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return document_parser_docx.parse_doc_document_ir(
            self.docx_parser_context(),
            filename,
            content,
            metadata,
            doc_version,
            kwargs.get("parser_name") or "antiword",
        )

    def parse_docx_document_ir(
        self,
        filename: str,
        content: bytes,
        metadata: Optional[Dict[str, Any]],
        doc_version: Optional[int],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return document_parser_docx.parse_docx_document_ir(
            self.docx_parser_context(),
            filename,
            content,
            metadata,
            doc_version,
            kwargs.get("parser_name") or "python-docx",
        )

    def parse_xlsx_document_ir(
        self,
        filename: str,
        content: bytes,
        metadata: Optional[Dict[str, Any]],
        doc_version: Optional[int],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return document_parser_xlsx.parse_xlsx_document_ir(
            self.xlsx_parser_context(),
            filename,
            content,
            metadata,
            doc_version,
            kwargs.get("parser_name") or "openpyxl",
        )

    def parser_context(self) -> document_parser.DocumentParserAdapter:
        return document_parser.DocumentParserAdapter(
            build_document_ir_from_text=self.build_document_ir_from_text,
            new_document_ir=lambda source, **kwargs: document_ir_helpers.new_document_ir(
                source,
                safe_filename=file_utils.safe_filename,
                **kwargs,
            ),
            append_ir_element=document_ir_helpers.append_ir_element,
            normalize_ir_text=document_ir_helpers.normalize_ir_text,
            probe_file_for_parser=self.probe_file_for_parser,
            merge_parser_metadata=self.merge_parser_metadata,
            supported_file_extensions=document_common.SUPPORTED_FILE_EXTENSIONS,
            parse_pdf_fast_document_ir=self.parse_pdf_fast_document_ir,
            parse_pdf_ocr_document_ir=self.parse_pdf_ocr_document_ir,
            parse_doc_document_ir=self.parse_doc_document_ir,
            parse_docx_document_ir=self.parse_docx_document_ir,
            parse_xlsx_document_ir=self.parse_xlsx_document_ir,
            parse_image_document_ir=self.parse_image_document_ir,
            document_ir_plain_text=document_ir_helpers.document_ir_plain_text,
        )

    def ensure_document_ir(self, source: str, active_version: Any) -> Any:
        return document_ir_store.ensure_document_ir(self.document_ir_store(), source, active_version)

    def get_chunks_for_source(self, source: str, doc_version: Optional[int] = None) -> List[Dict[str, Any]]:
        target_version = doc_version
        if target_version is None:
            target_version = self.get_active_version(source)
            if target_version is None:
                current = self.doc_get(source)
                target_version = current.get("pending_version")
        if target_version is not None:
            try:
                target_version = int(target_version)
            except Exception:
                target_version = None

        document_ir = document_ir_store.load_document_ir(self.document_ir_store(), source, target_version or 1) or {}
        if document_ir and (document_ir.get("elements") or []):
            items = self.document_ir_to_structured_items(document_ir, Config.CHUNK_SIZE, Config.OVERLAP)
            out: List[Dict[str, Any]] = []
            for item in items:
                raw_text = item.get("raw_text") or item.get("text") or ""
                section = item.get("section") or ""
                chunk_id = int(item.get("chunk_id") or 0)
                article_id = document_chunking.chunk_article_id(item)
                clause_meta = document_clause_metadata.build_clause_metadata(
                    source_file=source,
                    doc_title=(self.doc_get(source).get("canonical_title") or file_utils.filename_stem(source) or source),
                    item={**item, "article_id": article_id, "article_no": article_id},
                    base_metadata={"doc_id": self.canonical_doc_id_for_source(source)},
                    text=raw_text,
                ).to_dict()
                out.append(
                    {
                        "text": item.get("text") or "",
                        "raw_text": raw_text,
                        "section": section,
                        "chunk_id": chunk_id,
                        "article_id": article_id,
                        "article_no": article_id,
                        "metadata": {
                            "chunk_id": chunk_id,
                            "section": section,
                            "section_title": item.get("section_title") or section,
                            "article_id": article_id,
                            "article_no": article_id,
                            "doc_id": clause_meta["doc_id"],
                            "clause_id": article_id,
                            "clause_metadata": clause_meta,
                            "clause_label": item.get("clause_label") or article_id,
                            "section_node_id": item.get("section_node_id"),
                            "raw_text": raw_text,
                            "text_normalized": item.get("normalized_text") or document_ir_helpers.normalize_ir_text(raw_text),
                            "page_no": item.get("page_no"),
                            "page_span": item.get("page_span") or [],
                            "section_path": item.get("section_path") or [],
                            "parent_section_id": item.get("parent_section_id"),
                            "parent_section_path": item.get("parent_section_path") or [],
                            "parent_section_title": item.get("parent_section_title"),
                            "section_depth": item.get("section_depth"),
                            "semantic_unit_ids": item.get("semantic_unit_ids") or [],
                            "chunk_role": item.get("chunk_role") or "body",
                            "payload": item.get("payload") or {},
                            "element_id": item.get("element_id"),
                            "element_type": item.get("element_type"),
                            "reading_order": item.get("reading_order"),
                            "prev_chunk_id": item.get("prev_chunk_id"),
                            "next_chunk_id": item.get("next_chunk_id"),
                            "doc_version": document_ir.get("doc_version") or target_version or (self.get_active_version(source) or 1),
                        },
                    }
                )
            if out:
                return out

        fallback_version = target_version or self.get_active_version(source) or 1
        return self._lex_store.list_chunks_for_source(
            source,
            target_version=target_version,
            fallback_version=int(fallback_version),
        )

    def document_detail_plain_text(self, document_ir: Dict[str, Any], chunks: Any) -> str:
        return _document_detail_plain_text(document_ir, chunks)

    def chunk_plain_display_text(self, text: str) -> str:
        return _chunk_plain_display_text(text)

    def new_task_id(self) -> str:
        return uuid.uuid4().hex

    def task_log(self, task_id: str, event: str, payload: Any = None) -> None:
        with self.sqlite_write_lock():
            self._task_store.log(task_id, event, payload)

    def lex_db_set_status(self, source: str, status: str) -> None:
        with self.sqlite_write_lock():
            self._lex_store.set_status(source, status)

    def doc_upsert(self, source: str, **kwargs: Any) -> None:
        with self.sqlite_write_lock():
            self._lex_store.upsert_document(source, **kwargs)

    def build_publish_gate(self, source: str, doc_version: int) -> Dict[str, Any]:
        safe = file_utils.safe_filename(source)
        doc = self.doc_get(safe)
        stats = self._lex_store.source_version_stats(safe, int(doc_version))
        vector_db = self.vector_db()
        milvus_count = milvus_storage.version_count(safe, int(doc_version), vector_db=vector_db)
        profile = self._lex_store.profile(safe, int(doc_version))
        section_count = self._lex_store.section_count(safe, int(doc_version))
        parse_quality_score = float(profile.get("parse_quality_score") or doc.get("parse_quality_score") or 0.0)
        gate = {
            "sqlite_chunks_ok": stats["sqlite_chunks"] > 0,
            "fts_chunks_ok": stats["sqlite_chunks"] > 0 and stats["sqlite_chunks"] == stats["fts_chunks"],
            "milvus_vectors_ok": stats["sqlite_chunks"] > 0 and milvus_count == stats["sqlite_chunks"],
            "visibility_ok": milvus_storage.version_visible(safe, int(doc_version), vector_db=vector_db),
            "profile_ok": bool(profile),
            "section_index_ok": section_count > 0 or stats["sqlite_chunks"] <= 1,
            "parse_quality_ok": parse_quality_score >= float(Config.MIN_PARSE_QUALITY_SCORE),
        }
        gate["ready"] = all(gate.values())
        gate["counts"] = {
            "sqlite_chunks": stats["sqlite_chunks"],
            "fts_chunks": stats["fts_chunks"],
            "milvus_vectors": milvus_count,
            "sections": section_count,
        }
        return gate

    def finalize_pending_version_if_ready(self, source: str) -> bool:
        with self.sqlite_write_lock():
            safe = file_utils.safe_filename(source)
            doc = self.doc_get(safe)
            pending_version = doc.get("pending_version")
            if pending_version is None:
                return False
            try:
                pending_version = int(pending_version)
            except Exception:
                return False
            publish_gate = self.build_publish_gate(safe, pending_version)
            if not publish_gate.get("ready"):
                self.doc_upsert(
                    safe,
                    status="publish_failed",
                    last_error="publish gate not ready",
                    searchable=self.doc_searchable_flag(safe),
                    publish_gate=self.json_dumps(publish_gate),
                )
                self.lex_db_set_status(safe, "publish_failed")
                return False
            self.doc_upsert(
                safe,
                status="completed",
                active_version=pending_version,
                pending_version=None,
                last_error=None,
                searchable=1,
                publish_gate=self.json_dumps(publish_gate),
            )
            self.lex_db_set_status(safe, "completed")
            self._lex_store.cleanup_old_versions(safe, pending_version)
            milvus_storage.cleanup_old_versions(safe, pending_version, vector_db=self.vector_db())
            return True

    async def cancel_source_async_tasks(self, source: str) -> int:
        safe = file_utils.normalize_filename_for_match(source or "") or "__unknown__"
        tasks = [task for task in list(self._source_async_tasks.get(safe) or set()) if not task.done()]
        if not tasks:
            return 0
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    def get_source_lock(self, source: str) -> Any:
        key = (source or "").strip() or "__unknown__"
        lock = self._source_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            self._source_locks[key] = lock
        return lock

    def crash_inject(self, stage: str) -> None:
        _crash_inject(stage)

    def delete_milvus_document_object(self, vector_db: Any, source: str) -> Dict[str, Any]:
        return milvus_storage.delete_document_object(vector_db, source)

    def lex_db_delete_source(self, source: str) -> None:
        self._lex_store.delete_source(source, before_delete=lambda: self.crash_inject("delete_sqlite"))

    def delete_uploaded_artifacts(self, source: str) -> Dict[str, Any]:
        safe = file_utils.safe_filename(source)
        stem = os.path.splitext(safe)[0]
        candidates = []
        for root in {_UPLOAD_DIR, _LEGACY_UPLOAD_DIR}:
            if not root or not os.path.isdir(root):
                continue
            try:
                for name in os.listdir(root):
                    if name == os.path.basename(_LEXICAL_DB_FILE):
                        continue
                    matches = (
                        name == safe
                        or name.endswith(f"__{safe}")
                        or name == stem
                        or name.startswith(f"{safe}__")
                        or name.startswith(f"{stem}__")
                    )
                    if matches:
                        candidates.append(os.path.join(root, name))
            except Exception:
                continue
        for task in self.tasks.values():
            if file_utils.normalize_filename_for_match(task.get("filename") or "") != safe:
                continue
            path = (task.get("path") or "").strip()
            if path:
                candidates.append(path)
        unique_candidates = []
        for item in candidates:
            if item and item not in unique_candidates:
                unique_candidates.append(item)
        removed = []
        missing = []
        failed = []
        for path in unique_candidates:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    removed.append(path)
                elif os.path.isfile(path):
                    os.remove(path)
                    removed.append(path)
                else:
                    missing.append(path)
            except FileNotFoundError:
                missing.append(path)
            except Exception:
                failed.append(path)
        return {"removed": removed, "missing": missing, "failed": failed}

    def save_tasks(self) -> None:
        self._task_store.save()

    def build_text_upload_probe(self, source: str, content: str) -> Dict[str, Any]:
        return document_probe.build_text_upload_probe(self.probe_context(), source, content)

    def probe_file_for_parser(self, source: str, raw: bytes) -> Dict[str, Any]:
        return document_probe.probe_file_for_parser(self.probe_context(), source, raw)

    def validate_upload_probe(self, source: str, probe: Dict[str, Any], *, is_text_upload: bool) -> None:
        document_probe.validate_upload_probe(
            self.probe_context(),
            source,
            probe,
            is_text_upload=is_text_upload,
        )

    def doc_title_profile(self, source: str) -> Dict[str, Any]:
        return document_profile.doc_title_profile(source)

    def content_sha256_text(self, content: str) -> str:
        return document_profile.content_sha256_text(content)

    def content_sha256_bytes(self, raw: bytes) -> str:
        return document_profile.content_sha256_bytes(raw)

    def detect_duplicate_upload(self, source: str, content_sha256: str, canonical_title: str) -> Dict[str, Any]:
        return document_profile.detect_duplicate_upload(
            self.profile_store(),
            source,
            content_sha256,
            canonical_title,
        )

    def build_upload_response(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        fields = dict(kwargs)
        positional_names = ("task_id", "source", "task_status", "document_status", "searchable")
        for name, value in zip(positional_names, args):
            fields.setdefault(name, value)
        source = fields.get("source")
        doc = self.doc_get(source) if source else {}
        return {
            "task_id": fields.get("task_id"),
            "filename": source,
            "task_status": fields.get("task_status"),
            "document_status": fields.get("document_status"),
            "searchable": bool(fields.get("searchable")),
            "active_version": doc.get("active_version"),
            "pending_version": doc.get("pending_version"),
            "duplicate_state": fields.get("duplicate_state"),
            "duplicate_of": fields.get("duplicate_of"),
            "same_title_candidates": list(fields.get("same_title_candidates") or []),
        }

    def build_source_id(self, filename: str, content_sha256: str) -> str:
        return document_profile.build_source_id(filename, content_sha256)

    def docfts_upsert(self, source: str, **kwargs: Any) -> None:
        self._lex_store.upsert_document_fts(source, **kwargs)

    def lex_tx_begin(self) -> None:
        conn = self._lex_store.connect()
        if getattr(conn, "in_transaction", False):
            self._lex_store.rollback()
        self._lex_store.begin_immediate()

    def lex_tx_commit(self) -> None:
        self._lex_store.commit()

    def lex_tx_rollback(self) -> None:
        self._lex_store.rollback()

    def lex_db_checkpoint(self, mode: str) -> None:
        self._lex_store.checkpoint(mode)

    def doc_next_version(self, source: str) -> int:
        return self._lex_store.next_document_version(source)

    def build_document_ir_from_text(self, source: str, content: str, **kwargs: Any) -> Dict[str, Any]:
        return document_ir_helpers.build_document_ir_from_text(
            source,
            content,
            safe_filename=file_utils.safe_filename,
            **kwargs,
        )

    def document_ir_plain_text(self, document_ir: Dict[str, Any], *, normalized: bool = False) -> str:
        return document_ir_helpers.document_ir_plain_text(document_ir, normalized=normalized)

    def assess_document_quality(self, document_ir: Dict[str, Any], probe: Dict[str, Any]) -> Dict[str, Any]:
        return document_profile.assess_document_quality(
            document_ir,
            probe,
            min_parse_text_chars=Config.MIN_PARSE_TEXT_CHARS,
            min_parse_quality_score=Config.MIN_PARSE_QUALITY_SCORE,
        )

    def build_document_profile(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return document_profile.build_document_profile(*args, **kwargs)

    def json_dumps(self, value: Any) -> str:
        return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)

    def purge_source_for_reindex(self, source: str, version: int) -> None:
        safe = file_utils.safe_filename(source)
        self.crash_inject("before_purge")
        if version is not None:
            vector_db = self.vector_db()
            milvus_storage.delete_source_version(safe, version, vector_db=vector_db)
            with self.sqlite_write_lock():
                self._lex_store.delete_source_version(safe, version, drop_control_plane=False)
        self.crash_inject("after_purge")

    def store_document_ir(self, source: str, document_ir: Dict[str, Any]) -> None:
        document_ir_store.store_document_ir(self.document_ir_store(), source, document_ir)

    def index_base_metadata(
        self,
        filename: str,
        text: str,
        metadata: Optional[Dict[str, Any]],
        document_ir: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        source_text = document_ir_helpers.document_ir_plain_text(document_ir, normalized=False) if document_ir else text
        profile = document_profile.build_document_profile(
            filename,
            filename,
            self.build_source_id(filename, self.content_sha256_text(source_text or "")),
            self.content_sha256_text(source_text or ""),
            source_text or "",
            document_ir or self.build_document_ir_from_text(filename, source_text or ""),
            {},
            {"flags": [], "score": None},
        )
        return {
            **(metadata or {}),
            "doc_type": profile.get("doc_type"),
            "topics": profile.get("topic_terms") or [],
        }

    def vector_doc_entry(
        self,
        *,
        filename: str,
        batch_item: Dict[str, Any],
        embedding_value: Any,
        base_metadata: Dict[str, Any],
        chunk_id: int,
        chunk_count: int,
        created_at: str,
        doc_version: Optional[int] = None,
        sparse_embedding_value: Any = None,
    ) -> Dict[str, Any]:
        raw_text = sanitize_index_text(batch_item.get("raw_text") or batch_item.get("text") or "")
        section = (batch_item.get("section") or "").strip()
        title_profile = self.doc_title_profile(filename)
        doc_title = (title_profile.get("canonical_title") or title_profile.get("stem") or filename).strip()
        body_article_id = document_chunking.extract_leading_article_id(raw_text)
        article_id = (
            body_article_id
            or batch_item.get("article_id")
            or batch_item.get("article_no")
            or batch_item.get("clause_label")
            or document_chunking.extract_leading_article_id(
                batch_item.get("section"),
                batch_item.get("section_title"),
                batch_item.get("parent_section_title"),
            )
            or ""
        )
        clause_meta = document_clause_metadata.build_clause_metadata(
            source_file=filename,
            doc_title=doc_title,
            item={**batch_item, "article_id": article_id, "article_no": article_id},
            base_metadata={**base_metadata, "doc_id": self.canonical_doc_id_for_source(filename)},
            text=raw_text,
        ).to_dict()
        applicable_subjects = batch_item.get("applicable_subjects") or []
        if isinstance(applicable_subjects, str):
            applicable_subjects = [part.strip() for part in re.split(r"[,，、/；;]\s*", applicable_subjects) if part.strip()]
        elif not isinstance(applicable_subjects, list):
            applicable_subjects = []
        applicable_subjects = [
            value
            for value in (_milvus_varchar_value(part, 128) for part in applicable_subjects)
            if value
        ][:10]
        metadata = _milvus_safe_metadata(
            {
                **base_metadata,
                "doc_title": doc_title,
                "doc_id": clause_meta["doc_id"],
                "chunk_id": int(chunk_id),
                "chunk_count": int(chunk_count),
                "section": section,
                "section_id": batch_item.get("section_id"),
                "section_title": batch_item.get("section_title") or section,
                "section_node_id": batch_item.get("section_node_id"),
                "clause_label": batch_item.get("clause_label") or "",
                "article_no": article_id or batch_item.get("article_no") or batch_item.get("clause_label") or "",
                "article_id": article_id,
                "clause_id": article_id,
                "clause_metadata": clause_meta,
                "applicable_subjects": applicable_subjects,
                "action_type": batch_item.get("action_type") or "",
                "unit_kind": batch_item.get("unit_kind") or "paragraph",
                "raw_text": raw_text,
                "previous_context": sanitize_index_text(batch_item.get("previous_context") or ""),
                "content": sanitize_index_text(batch_item.get("content") or raw_text),
                "next_context": sanitize_index_text(batch_item.get("next_context") or ""),
                "text_normalized": batch_item.get("normalized_text") or document_ir_helpers.normalize_ir_text(raw_text),
                "fts_text": batch_item.get("fts_text") or raw_text,
                "page_no": batch_item.get("page_no"),
                "page_span": batch_item.get("page_span") or [],
                "section_path": batch_item.get("section_path") or [],
                "parent_section_id": batch_item.get("parent_section_id"),
                "parent_section_path": batch_item.get("parent_section_path") or [],
                "parent_section_title": batch_item.get("parent_section_title"),
                "section_depth": batch_item.get("section_depth"),
                "semantic_unit_ids": batch_item.get("semantic_unit_ids") or [],
                "chunk_role": batch_item.get("chunk_role") or "body",
                "payload": _summarize_chunk_payload(batch_item.get("payload") or {}),
                "element_id": batch_item.get("element_id"),
                "element_type": batch_item.get("element_type"),
                "reading_order": batch_item.get("reading_order"),
                "prev_chunk_id": batch_item.get("prev_chunk_id"),
                "next_chunk_id": batch_item.get("next_chunk_id"),
                "doc_version": doc_version,
                "rebuild_seq": created_at if doc_version is not None else None,
            }
        )
        doc = {
            "embedding": embedding_value,
            "text": _milvus_text_value(raw_text),
            "source": filename,
            "metadata": metadata,
            "article_id": str(article_id or ""),
            "applicable_subjects": applicable_subjects,
            "created_at": created_at,
        }
        if sparse_embedding_value:
            doc["sparse_embedding"] = sparse_embedding_value
        return doc

    async def index_document_incremental(self, *args: Any, **kwargs: Any) -> Any:
        return await self._indexing_service.index_document_incremental(*args, **kwargs)

    async def index_document(self, *args: Any, **kwargs: Any) -> Any:
        return await self._indexing_service.index_document(*args, **kwargs)

    def persist_document_profile(self, source: str, version: int, profile: Dict[str, Any]) -> None:
        with self.sqlite_write_lock():
            document_profile.persist_document_profile(self.profile_store(), source, version, profile)

    def register_source_async_task(self, source: str, task: Any) -> None:
        safe = file_utils.normalize_filename_for_match(source or "") or "__unknown__"
        bucket = self._source_async_tasks.setdefault(safe, set())
        bucket.add(task)

        def cleanup(done_task: Any) -> None:
            tasks = self._source_async_tasks.get(safe)
            if not tasks:
                return
            tasks.discard(done_task)
            if not tasks:
                self._source_async_tasks.pop(safe, None)

        task.add_done_callback(cleanup)

    def extract_document_ir_from_file(self, filename: str, content: bytes, **kwargs: Any) -> Dict[str, Any]:
        return document_parser.extract_document_ir_from_file(
            self.parser_context(),
            filename,
            content,
            metadata=kwargs.get("metadata"),
            doc_version=kwargs.get("doc_version"),
        )


def _document_context(runtime: Any) -> DocumentService:
    if isinstance(runtime, DocumentService):
        return runtime
    factory = getattr(runtime, "document_service", None)
    if callable(factory):
        return factory()
    return DocumentService(runtime)


async def _run_serialized_upload_task(runtime: DocumentService, work: Any) -> None:
    async with runtime.upload_index_async_lock():
        await work()


async def _upload_document_impl(doc_req: DocumentRequest, runtime: Any) -> Dict[str, Any]:
    runtime = _document_context(runtime)
    safe_name = runtime.safe_filename(doc_req.filename)
    probe = runtime.build_text_upload_probe(safe_name, doc_req.content)
    runtime.validate_upload_probe(safe_name, probe, is_text_upload=True)
    title_profile = runtime.doc_title_profile(safe_name)
    content_sha256 = runtime.content_sha256_text(doc_req.content)
    duplicate_info = runtime.detect_duplicate_upload(
        safe_name,
        content_sha256,
        title_profile["canonical_title"],
    )
    if duplicate_info.get("duplicate_state") in {"no_change", "already_exists"}:
        return runtime.build_upload_response(
            task_id=None,
            source=safe_name,
            task_status="completed",
            document_status=str(duplicate_info.get("duplicate_state") or "no_change"),
            searchable=bool(runtime.doc_get(duplicate_info.get("duplicate_of") or safe_name).get("searchable")),
            duplicate_state=duplicate_info.get("duplicate_state"),
            duplicate_of=duplicate_info.get("duplicate_of"),
            same_title_candidates=duplicate_info.get("same_title_candidates") or [],
        )

    task_id = runtime.new_task_id()
    runtime.tasks[task_id] = {
        "status": "accepted",
        "stage": "validating",
        "filename": safe_name,
        "created_at": datetime.now().isoformat(),
        "payload": {"text": doc_req.content, "metadata": doc_req.metadata},
        "document_status": "accepted",
    }
    runtime.task_log(task_id, "validating", {"filename": safe_name})
    runtime.lex_db_set_status(safe_name, "accepted")
    doc_type = (doc_req.metadata or {}).get("doc_type")
    topic = (doc_req.metadata or {}).get("topic")
    source_id = runtime.build_source_id(doc_req.filename, content_sha256)
    runtime.doc_upsert(
        safe_name,
        status="accepted",
        canonical_title=title_profile["canonical_title"],
        title_tokens=title_profile["title_tokens"],
        aliases=title_profile["aliases"],
        filename_stem=title_profile["stem"],
        doc_type=doc_type,
        topic=topic,
        source_id=source_id,
        original_filename=doc_req.filename,
        content_sha256=content_sha256,
        mime_type=probe.get("mime_type"),
        detected_ext=probe.get("detected_ext"),
        file_size=probe.get("file_size"),
        page_count=probe.get("page_count"),
        parser_route=probe.get("route"),
        parser_backend=probe.get("parser_backend"),
        parse_status="accepted",
        searchable=0,
        duplicate_state=duplicate_info.get("duplicate_state"),
        duplicate_of=duplicate_info.get("duplicate_of"),
        same_title_group=runtime.same_title_group(title_profile["canonical_title"]),
        suspicious_file_type=0,
    )
    runtime.docfts_upsert(
        safe_name,
        title=title_profile["canonical_title"],
        aliases=title_profile["aliases"],
        doc_type=doc_type,
        topic=topic,
    )

    async def _run() -> None:
        try:
            lock = runtime.get_source_lock(safe_name)
            if not lock.acquire(timeout=30):
                raise HTTPException(status_code=429, detail="文档正在处理中，请稍后重试")
            runtime.lex_tx_begin()
            runtime.lex_db_set_status(safe_name, "reindexing")
            version_next = runtime.doc_next_version(safe_name)
            runtime.doc_upsert(
                safe_name,
                status="reindexing",
                pending_version=version_next,
                parse_status="parsing",
                searchable=runtime.doc_searchable_flag(safe_name),
            )
            runtime.tasks[task_id]["status"] = "indexing"
            runtime.tasks[task_id]["stage"] = "parsing"
            runtime.task_log(task_id, "parsing")
            document_ir = runtime.build_document_ir_from_text(
                safe_name,
                doc_req.content,
                metadata=doc_req.metadata,
                parser_name="direct_text",
                doc_version=version_next,
            )
            text = runtime.document_ir_plain_text(document_ir, normalized=False)
            quality = runtime.assess_document_quality(document_ir, probe)
            if quality["status"] == "parse_empty":
                raise HTTPException(status_code=400, detail="文档解析结果为空，请检查文件内容")
            if quality["status"] == "parse_low_quality":
                raise HTTPException(status_code=400, detail="parse_low_quality: 文档解析质量过低，请检查文件或使用 OCR")

            runtime.tasks[task_id]["status"] = "indexing"
            runtime.tasks[task_id]["stage"] = "profile_building"
            runtime.task_log(task_id, "profile_building", {"quality": quality})
            profile = runtime.build_document_profile(
                safe_name,
                doc_req.filename,
                source_id,
                content_sha256,
                text,
                document_ir,
                probe,
                quality,
                metadata=doc_req.metadata,
            )
            runtime.doc_upsert(
                safe_name,
                status="reindexing",
                pending_version=version_next,
                parse_status=quality["status"],
                parse_quality_score=quality["score"],
                quality_flags=runtime.json_dumps(quality["flags"]),
                canonical_title=profile["canonical_title"],
                title_tokens=" ".join(profile.get("title_aliases") or []),
                aliases=",".join((profile.get("title_aliases") or [])[1:]),
                filename_stem=runtime.filename_stem(safe_name),
                doc_type=profile.get("doc_type"),
                topic=",".join((profile.get("topic_terms") or [])[:8]),
                source_id=source_id,
                original_filename=doc_req.filename,
                content_sha256=content_sha256,
                mime_type=probe.get("mime_type"),
                detected_ext=probe.get("detected_ext"),
                file_size=probe.get("file_size"),
                page_count=probe.get("page_count"),
                parser_route=probe.get("route"),
                parser_backend=probe.get("parser_backend"),
                searchable=runtime.doc_searchable_flag(safe_name),
                duplicate_state=duplicate_info.get("duplicate_state"),
                duplicate_of=duplicate_info.get("duplicate_of"),
                same_title_group=runtime.same_title_group(profile["canonical_title"]),
            )
            runtime.tasks[task_id]["status"] = "indexing"
            runtime.tasks[task_id]["stage"] = "embedding"
            runtime.task_log(task_id, "embedding")
            runtime.purge_source_for_reindex(safe_name, version_next)
            total_done = await runtime.index_document_incremental(
                task_id=task_id,
                filename=safe_name,
                text=text,
                metadata=doc_req.metadata,
                document_ir=document_ir,
            )
            runtime.persist_document_profile(safe_name, version_next, profile)
            runtime.crash_inject("before_commit")
            runtime.lex_db_set_status(safe_name, "vector_pending")
            runtime.doc_upsert(
                safe_name,
                status="vector_pending",
                pending_version=version_next,
                last_error=None,
                parse_status=quality["status"],
                searchable=runtime.doc_searchable_flag(safe_name),
            )
            runtime.lex_tx_commit()
            runtime.lex_db_checkpoint("PASSIVE")
            finalized = runtime.finalize_pending_version_if_ready(safe_name)
            runtime.tasks[task_id]["status"] = "completed" if finalized else "failed"
            runtime.tasks[task_id]["stage"] = "completed" if finalized else "publish_failed"
            runtime.tasks[task_id]["chunks_indexed"] = total_done
            runtime.tasks[task_id]["document_status"] = "completed" if finalized else "publish_failed"
            runtime.tasks[task_id]["searchable"] = runtime.doc_searchable_flag(safe_name)
            if finalized:
                runtime.task_log(task_id, "publish_completed", {"chunks_indexed": total_done, "active_version": version_next})
            else:
                doc = runtime.doc_get(safe_name)
                runtime.tasks[task_id]["error"] = doc.get("last_error") or "publish gate not ready"
                runtime.task_log(
                    task_id,
                    "publish_failed",
                    {
                        "chunks_indexed": total_done,
                        "pending_version": version_next,
                        "publish_gate": doc.get("publish_gate"),
                    },
                )
        except asyncio.CancelledError:
            runtime.lex_tx_rollback()
            runtime.tasks[task_id]["status"] = "failed"
            runtime.tasks[task_id]["stage"] = "cancelled"
            runtime.tasks[task_id]["error"] = "cancelled_for_delete"
            runtime.doc_upsert(safe_name, status="delete_failed", last_error="cancelled_for_delete", searchable=0)
            runtime.task_log(task_id, "cancelled", {"reason": "delete_requested"})
            raise
        except HTTPException as exc:
            runtime.lex_tx_rollback()
            detail = str(exc.detail or "upload_failed")
            status_code = detail.split(":", 1)[0]
            runtime.tasks[task_id]["status"] = "failed"
            runtime.tasks[task_id]["stage"] = status_code
            runtime.tasks[task_id]["error"] = detail
            runtime.doc_upsert(
                safe_name,
                status=status_code,
                last_error=detail,
                parse_status=status_code,
                searchable=runtime.doc_searchable_flag(safe_name),
            )
            runtime.task_log(task_id, "failed", {"error": detail})
        except Exception as exc:
            runtime.lex_tx_rollback()
            runtime.tasks[task_id]["status"] = "failed"
            runtime.tasks[task_id]["stage"] = "failed"
            runtime.tasks[task_id]["error"] = str(exc)
            runtime.doc_upsert(safe_name, status="vector_failed", last_error=str(exc), searchable=runtime.doc_searchable_flag(safe_name))
            runtime.task_log(task_id, "failed", {"error": str(exc)})
        finally:
            try:
                lock.release()
            except Exception:
                pass

    runtime.register_source_async_task(safe_name, asyncio.create_task(_run_serialized_upload_task(runtime, _run)))
    return runtime.build_upload_response(
        task_id,
        safe_name,
        "accepted",
        "accepted",
        False,
        duplicate_state=duplicate_info.get("duplicate_state"),
        duplicate_of=duplicate_info.get("duplicate_of"),
        same_title_candidates=duplicate_info.get("same_title_candidates") or [],
    )


async def upload_document(doc_req: DocumentRequest, runtime: Any) -> Dict[str, Any]:
    return await _upload_document_impl(doc_req, runtime)


async def _upload_document_file_impl(file: UploadFile, runtime: Any) -> Dict[str, Any]:
    runtime = _document_context(runtime)
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少上传文件名")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上传文件为空")
    safe_name = runtime.safe_filename(file.filename)
    probe = runtime.probe_file_for_parser(safe_name, raw)
    runtime.validate_upload_probe(safe_name, probe, is_text_upload=False)
    title_profile = runtime.doc_title_profile(safe_name)
    content_sha256 = runtime.content_sha256_bytes(raw)
    duplicate_info = runtime.detect_duplicate_upload(
        safe_name,
        content_sha256,
        title_profile["canonical_title"],
    )
    if duplicate_info.get("duplicate_state") in {"no_change", "already_exists"}:
        return runtime.build_upload_response(
            task_id=None,
            source=safe_name,
            task_status="completed",
            document_status=str(duplicate_info.get("duplicate_state") or "no_change"),
            searchable=bool(runtime.doc_get(duplicate_info.get("duplicate_of") or safe_name).get("searchable")),
            duplicate_state=duplicate_info.get("duplicate_state"),
            duplicate_of=duplicate_info.get("duplicate_of"),
            same_title_candidates=duplicate_info.get("same_title_candidates") or [],
        )

    task_id = runtime.new_task_id()
    path = os.path.join(runtime.upload_dir, f"{task_id}__{safe_name}")
    with open(path, "wb") as f:
        f.write(raw)
    runtime.tasks[task_id] = {
        "status": "accepted",
        "stage": "validating",
        "filename": safe_name,
        "path": path,
        "created_at": datetime.now().isoformat(),
        "document_status": "accepted",
    }
    runtime.task_log(task_id, "validating", {"filename": safe_name, "probe": probe})
    runtime.lex_db_set_status(safe_name, "accepted")
    source_id = runtime.build_source_id(file.filename, content_sha256)
    runtime.doc_upsert(
        safe_name,
        status="accepted",
        canonical_title=title_profile["canonical_title"],
        title_tokens=title_profile["title_tokens"],
        aliases=title_profile["aliases"],
        filename_stem=title_profile["stem"],
        doc_type=None,
        topic=None,
        source_id=source_id,
        original_filename=file.filename,
        content_sha256=content_sha256,
        mime_type=probe.get("mime_type"),
        detected_ext=probe.get("detected_ext"),
        file_size=probe.get("file_size"),
        page_count=probe.get("page_count"),
        parser_route=probe.get("route"),
        parser_backend=probe.get("parser_backend"),
        parse_status="accepted",
        searchable=0,
        duplicate_state=duplicate_info.get("duplicate_state"),
        duplicate_of=duplicate_info.get("duplicate_of"),
        same_title_group=runtime.same_title_group(title_profile["canonical_title"]),
        suspicious_file_type=0,
    )
    runtime.docfts_upsert(
        safe_name,
        title=title_profile["canonical_title"],
        aliases=title_profile["aliases"],
        doc_type=None,
        topic=None,
    )
    metadata = {"file_type": os.path.splitext(safe_name)[1].lstrip(".").lower(), "file_size": len(raw)}

    async def _run() -> None:
        try:
            lock = runtime.get_source_lock(safe_name)
            if not lock.acquire(timeout=30):
                raise HTTPException(status_code=429, detail="文档正在处理中，请稍后重试")
            runtime.lex_tx_begin()
            runtime.lex_db_set_status(safe_name, "reindexing")
            version_next = runtime.doc_next_version(safe_name)
            runtime.doc_upsert(
                safe_name,
                status="reindexing",
                pending_version=version_next,
                parse_status="parsing",
                searchable=runtime.doc_searchable_flag(safe_name),
            )
            runtime.tasks[task_id]["status"] = "indexing"
            runtime.tasks[task_id]["stage"] = "parsing"
            runtime.task_log(task_id, "parsing")
            document_ir = runtime.extract_document_ir_from_file(
                safe_name,
                raw,
                metadata=metadata,
                doc_version=version_next,
            )
            text = runtime.document_ir_plain_text(document_ir, normalized=False)
            quality = runtime.assess_document_quality(document_ir, probe)
            if quality["status"] == "parse_empty":
                raise HTTPException(status_code=400, detail="文档解析结果为空，请检查文件内容")
            if quality["status"] == "parse_low_quality":
                raise HTTPException(status_code=400, detail="parse_low_quality: 文档解析质量过低，请检查文件或使用 OCR")

            runtime.tasks[task_id]["stage"] = "profile_building"
            runtime.task_log(task_id, "profile_building", {"quality": quality})
            profile = runtime.build_document_profile(
                safe_name,
                file.filename,
                source_id,
                content_sha256,
                text,
                document_ir,
                probe,
                quality,
                metadata=metadata,
            )
            runtime.doc_upsert(
                safe_name,
                status="reindexing",
                pending_version=version_next,
                parse_status=quality["status"],
                parse_quality_score=quality["score"],
                quality_flags=runtime.json_dumps(quality["flags"]),
                canonical_title=profile["canonical_title"],
                title_tokens=" ".join(profile.get("title_aliases") or []),
                aliases=",".join((profile.get("title_aliases") or [])[1:]),
                filename_stem=runtime.filename_stem(safe_name),
                doc_type=profile.get("doc_type"),
                topic=",".join((profile.get("topic_terms") or [])[:8]),
                source_id=source_id,
                original_filename=file.filename,
                content_sha256=content_sha256,
                mime_type=probe.get("mime_type"),
                detected_ext=probe.get("detected_ext"),
                file_size=probe.get("file_size"),
                page_count=probe.get("page_count"),
                parser_route=probe.get("route"),
                parser_backend=probe.get("parser_backend"),
                searchable=runtime.doc_searchable_flag(safe_name),
                duplicate_state=duplicate_info.get("duplicate_state"),
                duplicate_of=duplicate_info.get("duplicate_of"),
                same_title_group=runtime.same_title_group(profile["canonical_title"]),
            )
            runtime.tasks[task_id]["status"] = "indexing"
            runtime.tasks[task_id]["stage"] = "embedding"
            runtime.task_log(task_id, "embedding")
            runtime.purge_source_for_reindex(safe_name, version_next)
            total_done = await runtime.index_document_incremental(
                task_id=task_id,
                filename=safe_name,
                text=text,
                metadata=metadata,
                document_ir=document_ir,
            )
            runtime.persist_document_profile(safe_name, version_next, profile)
            runtime.crash_inject("before_commit")
            runtime.lex_db_set_status(safe_name, "vector_pending")
            runtime.doc_upsert(
                safe_name,
                status="vector_pending",
                pending_version=version_next,
                last_error=None,
                parse_status=quality["status"],
                searchable=runtime.doc_searchable_flag(safe_name),
            )
            runtime.lex_tx_commit()
            runtime.lex_db_checkpoint("PASSIVE")
            finalized = runtime.finalize_pending_version_if_ready(safe_name)
            runtime.tasks[task_id]["status"] = "completed" if finalized else "failed"
            runtime.tasks[task_id]["stage"] = "completed" if finalized else "publish_failed"
            runtime.tasks[task_id]["chunks_indexed"] = total_done
            runtime.tasks[task_id]["document_status"] = "completed" if finalized else "publish_failed"
            runtime.tasks[task_id]["searchable"] = runtime.doc_searchable_flag(safe_name)
            if finalized:
                runtime.task_log(task_id, "publish_completed", {"chunks_indexed": total_done, "active_version": version_next})
            else:
                doc = runtime.doc_get(safe_name)
                runtime.tasks[task_id]["error"] = doc.get("last_error") or "publish gate not ready"
                runtime.task_log(
                    task_id,
                    "publish_failed",
                    {
                        "chunks_indexed": total_done,
                        "pending_version": version_next,
                        "publish_gate": doc.get("publish_gate"),
                    },
                )
        except asyncio.CancelledError:
            runtime.lex_tx_rollback()
            runtime.tasks[task_id]["status"] = "failed"
            runtime.tasks[task_id]["stage"] = "cancelled"
            runtime.tasks[task_id]["error"] = "cancelled_for_delete"
            runtime.doc_upsert(safe_name, status="delete_failed", last_error="cancelled_for_delete", searchable=0)
            runtime.task_log(task_id, "cancelled", {"reason": "delete_requested"})
            raise
        except HTTPException as exc:
            runtime.lex_tx_rollback()
            detail = str(exc.detail or "upload_failed")
            status_code = detail.split(":", 1)[0]
            runtime.tasks[task_id]["status"] = "failed"
            runtime.tasks[task_id]["stage"] = status_code
            runtime.tasks[task_id]["error"] = detail
            runtime.doc_upsert(
                safe_name,
                status=status_code,
                last_error=detail,
                parse_status=status_code,
                searchable=runtime.doc_searchable_flag(safe_name),
            )
            runtime.task_log(task_id, "failed", {"error": detail})
        except Exception as exc:
            runtime.lex_tx_rollback()
            runtime.tasks[task_id]["status"] = "failed"
            runtime.tasks[task_id]["stage"] = "failed"
            runtime.tasks[task_id]["error"] = str(exc)
            runtime.doc_upsert(safe_name, status="vector_failed", last_error=str(exc), searchable=runtime.doc_searchable_flag(safe_name))
            runtime.task_log(task_id, "failed", {"error": str(exc)})
        finally:
            try:
                lock.release()
            except Exception:
                pass

    runtime.register_source_async_task(safe_name, asyncio.create_task(_run_serialized_upload_task(runtime, _run)))
    return runtime.build_upload_response(
        task_id,
        safe_name,
        "accepted",
        "accepted",
        False,
        duplicate_state=duplicate_info.get("duplicate_state"),
        duplicate_of=duplicate_info.get("duplicate_of"),
        same_title_candidates=duplicate_info.get("same_title_candidates") or [],
    )


async def upload_document_file(file: UploadFile, runtime: Any) -> Dict[str, Any]:
    return await _upload_document_file_impl(file, runtime)


async def list_documents(runtime: Any) -> Dict[str, Any]:
    return await document_lifecycle_service.list_documents(runtime)


async def delete_document(filename: str, runtime: Any) -> Any:
    return await document_lifecycle_service.delete_document(filename, runtime)


async def retry_task(task_id: str, runtime: Any) -> Dict[str, Any]:
    return await document_lifecycle_service.retry_task(task_id, runtime)


async def get_document_detail(filename: str, runtime: Any) -> Any:
    return await document_lifecycle_service.get_document_detail(filename, runtime)
