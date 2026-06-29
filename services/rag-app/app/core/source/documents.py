"""Runtime adapter document helpers for source operations."""

import re
from typing import Any, Dict, List, Optional, Tuple

from app.core import source as source_resolution_core
from app.core.source import state as source_state_core
from app.documents import chunking as document_chunking
from app.documents import clause_metadata as document_clause_metadata
from app.documents import common as document_common
from app.documents import ir as document_ir_helpers
from app.documents import ir_store as document_ir_store
from app.documents import profile as document_profile
from app.utils import files as file_utils

class SourceDocumentMixin:
    @property
    def lex_store(self) -> Any:
        return self.runtime.lex_store
    def doc_get(self, source: str) -> Dict[str, Any]:
        return self.lex_store.get_document(source)
    def document_ir_store(self) -> document_ir_store.DocumentIRStoreAdapter:
        return document_ir_store.DocumentIRStoreAdapter(
            connect=self.lex_store.connect,
            read_connect=self.lex_store.read_connect,
            active_version=lambda source: self.doc_get(source).get("active_version"),
            pending_version=lambda source: self.doc_get(source).get("pending_version"),
        )
    def load_document_ir(self, source: str, version: Optional[int] = None) -> Dict[str, Any]:
        return document_ir_store.load_document_ir(self.document_ir_store(), source, version) or {}
    def ensure_document_ir(self, source: str, version: Optional[int] = None) -> Dict[str, Any]:
        return document_ir_store.ensure_document_ir(self.document_ir_store(), source, version) or {}
    def filename_stem(self, source: str) -> str:
        return file_utils.filename_stem(source)
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
        if re.match(r"^绗琜涓€浜屼笁鍥涗簲鍏竷鍏節鍗佺櫨鍗冧竾0-9]+[绔犺妭缂栭儴鍒嗘潯][\s:锛氥€?-]*", value):
            return value
        return ""
    def clause_heading_label(self, text: str) -> str:
        match = re.match(r"^(第[一二三四五六七八九十百千万0-9]+[条款项])", (text or "").strip())
        return match.group(1) if match else ""
        match = re.match(r"^(绗琜涓€浜屼笁鍥涗簲鍏竷鍏節鍗佺櫨鍗冧竾0-9]+[鏉℃椤筣)", (text or "").strip())
        return match.group(1) if match else ""
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
        match = re.match(r"^(第[一二三四五六七八九十百千万0-9]+[条款项])", (text or "").strip())
        return match.group(1) if match else ""

    def looks_like_pdf_page_number(self, text: str, page_no: Optional[int] = None) -> bool:
        del page_no
        compact = re.sub(r"[\s\-\u2010-\u2015|]+", "", text or "")
        return bool(compact and re.fullmatch(r"\d{1,3}", compact))
    def is_pdf_noise_text(self, text: str, page_no: Optional[int] = None) -> bool:
        raw = (text or "").strip()
        if not raw:
            return True
        line_fragments = [re.sub(r"\s+", "", line) for line in raw.splitlines() if line.strip()]
        if len(line_fragments) >= 6 and line_fragments and max(len(line) for line in line_fragments) <= 1:
            return True
        compact = re.sub(r"\s+", "", raw)
        if not compact:
            return True
        if self.looks_like_pdf_page_number(compact, page_no=page_no):
            return True
        meaningful_chars = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", compact)
        meaningful_runs = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", raw)
        symbol_chars = len(compact) - len(meaningful_chars)
        if not meaningful_chars:
            return True
        if len(meaningful_chars) <= 2 and symbol_chars >= max(2, len(meaningful_chars)):
            return True
        if meaningful_runs and max(len(run) for run in meaningful_runs) <= 2 and symbol_chars >= len(meaningful_chars):
            return True
        if len(compact) >= 6 and symbol_chars / max(1, len(compact)) >= 0.6 and len(meaningful_chars) <= 3:
            return True
        return False
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
            if re.fullmatch(r"鍥綷s*\d+|figure\s*\d+", raw_text, flags=re.IGNORECASE):
                return True
        if element_type in {"paragraph", "list_item", "key_value", "caption", "heading", "title"}:
            return self.is_pdf_noise_text(raw_text, page_no=element.get("page_no"))
        return False
    def serialize_ir_element(self, element: Dict[str, Any]) -> Dict[str, str]:
        return document_ir_helpers.serialize_ir_element(
            element,
            should_skip=self.should_skip_ir_element_for_chunking,
            normalize_section_path=document_chunking.normalize_section_path,
            section_path_label=document_chunking.section_path_label,
        )
    def document_chunking_context(self) -> document_chunking.DocumentChunkingAdapter:
        return document_chunking.DocumentChunkingAdapter(
            normalize_ir_text=document_ir_helpers.normalize_ir_text,
            chapter_heading_title=self.chapter_heading_title,
            clause_heading_label=self.clause_heading_label,
            serialize_ir_element=self.serialize_ir_element,
            doc_title_profile=self.runtime.common.doc_title_profile,
            filename_stem=self.filename_stem,
            split_text=document_common.split_text,
            chunk_prev_context_chars=int(getattr(self.runtime.config, "CHUNK_PREV_CONTEXT_CHARS", 220)),
            chunk_next_context_chars=int(getattr(self.runtime.config, "CHUNK_NEXT_CONTEXT_CHARS", 220)),
        )
    def document_ir_to_structured_items(
        self,
        document_ir: Dict[str, Any],
        chunk_size: int,
        overlap: int,
    ) -> List[Dict[str, Any]]:
        return document_chunking.document_ir_to_structured_items(
            self.document_chunking_context(),
            document_ir,
            chunk_size,
            overlap,
        )
    def get_chunks_for_source(self, source: str, doc_version: Optional[int] = None) -> List[Dict[str, Any]]:
        target_version = doc_version
        if target_version is None:
            target_version = self.doc_get(source).get("active_version")
            if target_version is None:
                target_version = self.doc_get(source).get("pending_version")
        if target_version is not None:
            try:
                target_version = int(target_version)
            except Exception:
                target_version = None
        document_ir = self.load_document_ir(source, target_version or 1)
        if document_ir and (document_ir.get("elements") or []):
            items = self.document_ir_to_structured_items(
                document_ir,
                int(self.runtime.config.CHUNK_SIZE),
                int(self.runtime.config.OVERLAP),
            )
            out: List[Dict[str, Any]] = []
            for item in items:
                chunk_id = int(item.get("chunk_id") or 0)
                raw_text = item.get("raw_text") or item.get("text") or ""
                article_id = document_chunking.chunk_article_id(item)
                clause_meta = document_clause_metadata.build_clause_metadata(
                    source_file=source,
                    doc_title=(self.doc_get(source).get("canonical_title") or file_utils.filename_stem(source) or source),
                    item={**item, "article_id": article_id, "article_no": article_id},
                    base_metadata={"doc_id": self.canonical_doc_id(source)},
                    text=raw_text,
                ).to_dict()
                out.append(
                    {
                        "text": item.get("text") or "",
                        "raw_text": raw_text,
                        "section": item.get("section") or "",
                        "chunk_id": chunk_id,
                        "article_id": article_id,
                        "article_no": article_id,
                        "metadata": {
                            "chunk_id": chunk_id,
                            "section": item.get("section") or "",
                            "section_title": item.get("section_title") or item.get("section") or "",
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
                            "doc_version": document_ir.get("doc_version")
                            or target_version
                            or (self.doc_get(source).get("active_version") or 1),
                        },
                    }
                )
            if out:
                return out
        fallback_version = target_version or self.doc_get(source).get("active_version") or 1
        try:
            fallback_version = int(fallback_version)
        except Exception:
            fallback_version = 1
        return self.lex_store.list_chunks_for_source(
            source,
            target_version=target_version,
            fallback_version=fallback_version,
        )
