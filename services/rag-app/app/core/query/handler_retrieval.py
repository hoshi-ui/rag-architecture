import asyncio
import time
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from app.core import retrieval as retrieval_core
from app.core.query import (
    _build_compare_status_matrix_answer,
    _compare_source_status_prompt_lines,
    build_empty_search_result,
    build_lightweight_recall_result,
    build_multi_doc_compare_result,
    build_process_evidence_refusal_result,
    build_process_recall_blocked_result,
    build_process_retrieval_error_result,
    build_process_soft_clarification_result,
    build_process_source_lock_result,
    build_process_success_result,
    build_retrieve_recall_blocked_result,
    build_retrieve_soft_clarification_result,
    build_retrieve_source_lock_result,
    build_retrieve_success_result,
    compute_recall_window,
    downgrade_evidence_refusal_to_prompt_warning,
    finalize_generated_answer,
    handle_required_source_lock,
    prepare_answer_generation_context,
    prepare_lightweight_recall_prelude,
    prepare_process_evidence_context,
    prepare_process_query,
    prepare_recall_source_context,
    prepare_retrieval_query_context,
    prepare_retrieve_evidence_context,
    prepare_retrieve_query,
    process_refusal_reason,
)
from app.core.query.recall_flow import has_forced_retrieval_signal


class QueryRetrievalProcessMixin:
    def _speculative_dense_query_candidate(
        self,
        query: str,
        llm_parse: Dict[str, Any],
        intent_classification: Dict[str, Any],
    ) -> str:
        runtime = self.runtime
        for value in (
            (llm_parse or {}).get("dense_query"),
            (llm_parse or {}).get("retrieval_query"),
            (llm_parse or {}).get("search_database_tool_query"),
            (intent_classification or {}).get("search_database_tool_query"),
            query,
        ):
            normalized = runtime.common.normalize_query(str(value or ""))
            if normalized:
                return normalized
        return ""

    def _start_speculative_embedding(self, dense_query_candidate: str) -> Optional[asyncio.Task]:
        candidate = str(dense_query_candidate or "").strip()
        if not candidate:
            return None
        return asyncio.create_task(
            self.embedding_service.embed_with_sparse(
                [candidate],
                return_sparse=True,
            )
        )

    def _consume_speculative_embedding_result(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            try:
                self.runtime.logger.warning("speculative_embedding_discarded: %s", exc)
            except Exception:
                pass

    def _discard_speculative_embedding(self, task: Optional[asyncio.Task]) -> None:
        if task is None:
            return
        if not task.done():
            task.cancel()
        task.add_done_callback(self._consume_speculative_embedding_result)

    async def _embed_dense_query(
        self,
        dense_query: str,
        speculative_query: str,
        speculative_task: Optional[asyncio.Task],
    ) -> tuple[List[float], Optional[Dict[int, float]]]:
        runtime = self.runtime
        final_query = runtime.common.normalize_query(str(dense_query or ""))
        candidate = runtime.common.normalize_query(str(speculative_query or ""))
        if speculative_task is not None and final_query and final_query == candidate:
            query_embeddings, query_sparse_embeddings = await speculative_task
        else:
            self._discard_speculative_embedding(speculative_task)
            query_embeddings, query_sparse_embeddings = await self.embedding_service.embed_with_sparse(
                [dense_query],
                return_sparse=True,
            )
        query_embedding = query_embeddings[0]
        query_sparse_embedding = query_sparse_embeddings[0] if query_sparse_embeddings else None
        return query_embedding, query_sparse_embedding

    async def _run_target_scoped_recall(
        self,
        query: str,
        retrieval_query: str,
        query_embedding: List[float],
        query_sparse_embedding: Optional[Dict[int, float]],
        qtype: str,
        qfilters: Dict[str, Any],
        recall_k: int,
        final_n: int,
        pool_n: int,
        enable_rerank: bool,
        target_source: str,
        compare_subquery: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        return await retrieval_core.run_target_scoped_recall(
            self.runtime.retrieval,
            self,
            query=query,
            retrieval_query=retrieval_query,
            query_embedding=query_embedding,
            query_sparse_embedding=query_sparse_embedding,
            qtype=qtype,
            qfilters=qfilters,
            recall_k=recall_k,
            final_n=final_n,
            pool_n=pool_n,
            enable_rerank=enable_rerank,
            target_source=target_source,
            compare_subquery=compare_subquery,
        )

    async def _run_lightweight_recall(
        self,
        query: str,
        top_k: int,
        enable_rerank: bool,
        filename_hints: Optional[List[str]] = None,
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        runtime = self.runtime
        prelude = await prepare_lightweight_recall_prelude(
            runtime,
            query,
            filename_hints=filename_hints,
            user_id=user_id,
        )
        if prelude.get("early_return") is not None:
            return prelude["early_return"]

        intent_classification = prelude["intent_classification"]
        qtype = prelude["qtype"]
        llm_parse = prelude["llm_parse"]
        query_explicit_set = prelude["query_explicit_set"]
        source_resolution = prelude["source_resolution"]
        query_route = prelude["query_route"]
        classifier_compare = prelude["classifier_compare"]
        query_quality = prelude["query_quality"]
        intent_tier = prelude["intent_tier"]
        speculative_dense_query = self._speculative_dense_query_candidate(query, llm_parse, intent_classification)
        speculative_embedding_task = self._start_speculative_embedding(speculative_dense_query)

        source_context = prepare_recall_source_context(
            runtime,
            query,
            qtype,
            llm_parse,
            intent_classification,
            source_resolution,
            query_route,
            classifier_compare,
            query_quality,
            intent_tier,
            filename_hints=filename_hints,
            user_id=user_id,
        )
        if source_context.get("early_return") is not None:
            self._discard_speculative_embedding(speculative_embedding_task)
            return source_context["early_return"]

        compare_plan = source_context["compare_plan"]
        is_comparison_hint = source_context["is_comparison_hint"]
        clarification_limit = source_context["clarification_limit"]
        source_resolution = source_context["source_resolution"]
        query_route = source_context["query_route"]
        fnames = source_context["fnames"]
        active_fnames = source_context["active_fnames"]
        compare_source_set = source_context["compare_source_set"]
        topical_multi_doc_mode = source_context["topical_multi_doc_mode"]

        lock_context = handle_required_source_lock(
            runtime,
            query,
            query,
            qtype,
            llm_parse,
            intent_classification,
            source_resolution,
            query_route,
            classifier_compare,
            is_comparison_hint,
            query_quality,
            intent_tier,
            compare_plan,
            clarification_limit,
            fnames,
            active_fnames,
            topical_multi_doc_mode,
            user_id=user_id,
        )
        if lock_context.get("early_return") is not None:
            self._discard_speculative_embedding(speculative_embedding_task)
            return lock_context["early_return"]

        source_resolution = lock_context["source_resolution"]
        query_route = lock_context["query_route"]
        fnames = lock_context["fnames"]
        active_fnames = lock_context["active_fnames"]
        topical_multi_doc_mode = lock_context["topical_multi_doc_mode"]

        query_context = await prepare_retrieval_query_context(
            runtime,
            query,
            qtype,
            llm_parse,
            intent_classification,
            source_resolution,
            query_route,
            classifier_compare,
            is_comparison_hint,
            query_quality,
            intent_tier,
            compare_plan,
            compare_source_set,
            clarification_limit,
            fnames,
            active_fnames,
            topical_multi_doc_mode,
            query_explicit_set,
        )
        if query_context.get("early_return") is not None:
            self._discard_speculative_embedding(speculative_embedding_task)
            return query_context["early_return"]

        retrieval_query = query_context["retrieval_query"]
        retrieval_query_raw = query_context["retrieval_query_raw"]
        dense_query = query_context["dense_query"]
        qfilters = query_context["qfilters"]
        llm_parse = query_context["llm_parse"]
        source_resolution = query_context["source_resolution"]
        active_fnames = query_context["active_fnames"]
        is_comparison = query_context["is_comparison"]

        query_embedding, query_sparse_embedding = await self._embed_dense_query(
            dense_query,
            speculative_dense_query,
            speculative_embedding_task,
        )
        recall_window = compute_recall_window(runtime.config, top_k, enable_rerank, active_fnames)
        requested_k = recall_window["requested_k"]
        recall_k = recall_window["recall_k"]
        final_n = recall_window["final_n"]
        pool_n = recall_window["pool_n"]

        if query_route != "multi_doc_compare" and len(active_fnames) == 1 and not topical_multi_doc_mode:
            scoped = await self._run_target_scoped_recall(
                query=query,
                retrieval_query=retrieval_query,
                query_embedding=query_embedding,
                query_sparse_embedding=query_sparse_embedding,
                qtype=qtype,
                qfilters=qfilters,
                recall_k=recall_k,
                final_n=final_n,
                pool_n=pool_n,
                enable_rerank=enable_rerank,
                target_source=active_fnames[0],
            )
            scoped_source_resolution = {
                **dict(source_resolution or {}),
                "source_resolution_trace": {
                    **dict((source_resolution or {}).get("source_resolution_trace") or {}),
                    "clause_rerank": dict(scoped.get("clause_rerank") or {}),
                    **({"retrieval_stage_trace": dict(scoped.get("stage_trace") or {})} if scoped.get("stage_trace") else {}),
                },
            }
            return build_lightweight_recall_result(
                runtime,
                query,
                retrieval_query,
                retrieval_query_raw,
                dense_query,
                qtype,
                qfilters,
                llm_parse,
                intent_classification,
                is_comparison,
                query_route,
                scoped.get("docs") or [],
                {
                    "hits": scoped.get("dense_hits") or [],
                    "dropped": int(scoped.get("early_filtered") or 0),
                    "states": dict(scoped.get("dense_visible_states") or {}),
                },
                {
                    "hits": scoped.get("lexical_hits") or [],
                    "dropped": 0,
                    "states": dict(scoped.get("lexical_visible_states") or {}),
                },
                scoped.get("selected_docs") or [],
                scoped.get("post_filter_docs") or [],
                scoped.get("retrieve_docs") or [],
                scoped.get("dense_source_scores") or {},
                scoped.get("score_mode") or "score",
                {
                    "used": bool(scoped.get("rerank_used")),
                    "hits": scoped.get("docs") or [],
                    "stage_trace": dict(scoped.get("stage_trace") or {}),
                },
                recall_k,
                final_n,
                bool(runtime.routing.is_weak_reference_query(query)),
                scoped_source_resolution,
                active_fnames,
                topical_multi_doc_mode,
                compare_plan,
                intent_tier,
            )

        if query_route == "multi_doc_compare" and bool(compare_source_set.get("sources") or active_fnames):
            compare_sources = [
                runtime.common.normalize_filename(name)
                for name in (compare_source_set.get("sources") or active_fnames)
                if runtime.common.normalize_filename(name)
            ]
            compare_sources = list(dict.fromkeys(compare_sources))
            compare_subqueries = dict(source_resolution.get("compare_source_subqueries") or compare_plan.get("source_subqueries") or {})
            compare_source_results = await asyncio.gather(*[
                self._run_target_scoped_recall(
                    query=query,
                    retrieval_query=retrieval_query,
                    query_embedding=query_embedding,
                    query_sparse_embedding=query_sparse_embedding,
                    qtype=qtype,
                    qfilters=qfilters,
                    recall_k=recall_k,
                    final_n=final_n,
                    pool_n=pool_n,
                    enable_rerank=enable_rerank,
                    target_source=source,
                    compare_subquery=compare_subqueries.get(source) or None,
                )
                for source in compare_sources
            ])
            return build_multi_doc_compare_result(
                runtime,
                query,
                retrieval_query,
                retrieval_query_raw,
                dense_query,
                qtype,
                qfilters,
                llm_parse,
                intent_classification,
                is_comparison,
                "multi_doc_compare",
                source_resolution,
                compare_plan,
                compare_source_set,
                compare_sources,
                compare_subqueries,
                compare_source_results,
                requested_k,
                recall_k,
                final_n,
            )

        search_candidates = retrieval_core.run_lightweight_search_candidates(
            runtime.retrieval,
            self,
            query_embedding,
            retrieval_query,
            active_fnames,
            recall_k,
            qfilters=qfilters,
            query_sparse_embedding=query_sparse_embedding,
        )
        docs = search_candidates["docs"]
        lex_items = search_candidates["lex_items"]
        docs_all = search_candidates["docs_all"]
        visible_dense = search_candidates["visible_dense"]
        visible_lex = search_candidates["visible_lex"]
        dense_source_scores = search_candidates["dense_source_scores"]

        if not docs_all:
            return build_empty_search_result(
                runtime,
                query,
                retrieval_query,
                retrieval_query_raw,
                dense_query,
                qtype,
                qfilters,
                llm_parse,
                intent_classification,
                is_comparison,
                query_route,
                docs,
                lex_items,
                visible_dense,
                visible_lex,
                dense_source_scores,
                recall_k,
                final_n,
                source_resolution,
                active_fnames,
            )

        recall_processing = await retrieval_core.rerank_and_postprocess_lightweight_docs(
            runtime.retrieval,
            self.rerank_service,
            docs,
            lex_items,
            retrieval_query,
            qtype,
            qfilters,
            active_fnames,
            recall_k,
            dense_source_scores=dense_source_scores,
            final_n=final_n,
            pool_n=pool_n,
            enable_rerank=enable_rerank,
        )
        docs = recall_processing["docs"]
        weak_query = recall_processing["weak_query"]
        reranked_chunk = recall_processing["reranked_chunk"]
        score_mode = recall_processing["score_mode"]
        retrieve_docs = recall_processing["retrieve_docs"]
        selected_docs = recall_processing["selected_docs"]
        post_filter_docs = recall_processing["post_filter_docs"]

        return build_lightweight_recall_result(
            runtime,
            query,
            retrieval_query,
            retrieval_query_raw,
            dense_query,
            qtype,
            qfilters,
            llm_parse,
            intent_classification,
            is_comparison,
            query_route,
            docs,
            visible_dense,
            visible_lex,
            selected_docs,
            post_filter_docs,
            retrieve_docs,
            dense_source_scores,
            score_mode,
            reranked_chunk,
            recall_k,
            final_n,
            weak_query,
            source_resolution,
            active_fnames,
            topical_multi_doc_mode,
            compare_plan,
            intent_tier,
        )

    async def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int,
        enable_rerank: bool,
    ) -> Dict[str, Any]:
        runtime = self.runtime
        try:
            retrieve_input = prepare_retrieve_query(runtime, query, user_id)
            query = retrieve_input["query"]
            fnames = retrieve_input["fnames"]
            if retrieve_input.get("early_return") is not None:
                return retrieve_input["early_return"]
            recall = await self._run_lightweight_recall(
                query,
                top_k=top_k,
                enable_rerank=enable_rerank,
                filename_hints=fnames,
                user_id=user_id,
            )
            if recall.get("blocked_reason"):
                return build_retrieve_recall_blocked_result(runtime, query, user_id, recall)
            if recall.get("soft_clarification_required") and (
                recall.get("search_database_tool_empty") or not has_forced_retrieval_signal(runtime, query)
            ):
                clarification = await self._build_rule_backed_clarification(
                    query,
                    reason=str(recall.get("soft_clarification_reason") or recall.get("intent_tier") or "document_clarification"),
                    seed_sources=list(recall.get("source_lock_candidates") or []),
                )
                return build_retrieve_soft_clarification_result(runtime, query, user_id, recall, clarification)
            if (
                recall.get("source_lock_required")
                and not recall.get("resolved_source_lock")
                and (
                    recall.get("source_resolution_status") in {"ambiguous", "not_found"}
                    or not has_forced_retrieval_signal(runtime, query)
                )
            ):
                return build_retrieve_source_lock_result(runtime, query, user_id, recall)

            evidence_context = await prepare_retrieve_evidence_context(runtime, query, recall, fnames, top_k)
            resolved_targets = evidence_context["resolved_targets"]
            retrieve_docs = evidence_context["retrieve_docs"]
            observations = evidence_context["observations"]
            return build_retrieve_success_result(runtime, query, user_id, recall, retrieve_docs, resolved_targets, observations)
        except Exception as exc:
            runtime.logger.error(f"Retrieve processing error: {str(exc)}")
            raise HTTPException(status_code=500, detail=str(exc))

    async def process(
        self,
        query: str,
        user_id: str,
        top_k: int,
        enable_rerank: bool,
        forced_fnames: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        runtime = self.runtime
        try:
            process_started = time.perf_counter()
            recall_done_at = process_started
            recall_ms = 0.0
            process_input = prepare_process_query(
                runtime,
                query,
                user_id,
                forced_fnames=forced_fnames,
                too_short_answer=runtime.guardrails.invalid_query_message("query_too_short"),
                blocked_answer=runtime.guardrails.invalid_query_message("blocked_query"),
            )
            query = process_input["query"]
            runtime.logger.info({"event": "query.extracted", "query": query, "query_len": len(query or "")})
            qtype = process_input["qtype"]
            original_fnames = process_input["original_fnames"]
            if process_input.get("early_return") is not None:
                return process_input["early_return"]

            try:
                recall_started = time.perf_counter()
                recall = await self._run_lightweight_recall(
                    query,
                    top_k=top_k,
                    enable_rerank=enable_rerank,
                    filename_hints=original_fnames,
                    user_id=user_id,
                )
                recall_done_at = time.perf_counter()
                recall_ms = recall_done_at - recall_started
            except Exception as exc:
                runtime.logger.exception("retrieval_recall_exception: type=%s msg=%s", type(exc).__name__, exc)
                return build_process_retrieval_error_result(
                    runtime,
                    query,
                    user_id,
                    qtype,
                    exc,
                    runtime.guardrails.invalid_query_message("retrieval_error"),
                )

            if recall.get("blocked_reason"):
                clarification = await self._build_rule_backed_clarification(
                    query,
                    reason=str(recall.get("blocked_reason") or "low_information_query"),
                    seed_sources=list(recall.get("source_lock_candidates") or recall.get("target_sources") or []),
                )
                return build_process_recall_blocked_result(runtime, query, user_id, qtype, recall, clarification)

            if recall.get("soft_clarification_required") and (
                recall.get("search_database_tool_empty") or not has_forced_retrieval_signal(runtime, query)
            ):
                clarification = await self._build_rule_backed_clarification(
                    query,
                    reason=str(recall.get("soft_clarification_reason") or recall.get("intent_tier") or "document_clarification"),
                    seed_sources=list(recall.get("source_lock_candidates") or []),
                )
                return build_process_soft_clarification_result(runtime, query, user_id, qtype, recall, clarification)

            resolved_targets = [
                runtime.common.normalize_filename(x)
                for x in (recall.get("target_sources") or original_fnames)
                if runtime.common.normalize_filename(x)
            ]
            if (
                recall.get("source_lock_required")
                and not recall.get("resolved_source_lock")
                and (
                    recall.get("source_resolution_status") in {"ambiguous", "not_found"}
                    or not has_forced_retrieval_signal(runtime, query)
                )
            ):
                runtime.source.clear_current_locked_document(user_id)
                source_lock_reason = recall.get("source_lock_reason") or "document_target_required"
                if source_lock_reason in {"compare_target_not_found", "compare_targets_not_found", "compare_source_set_incomplete", "document_not_found"}:
                    return build_process_source_lock_result(runtime, query, user_id, qtype, recall)
                clarification = await self._build_rule_backed_clarification(
                    query,
                    reason=str(source_lock_reason or "document_target_required"),
                    seed_sources=list(recall.get("source_lock_candidates") or []),
                )
                return build_process_source_lock_result(runtime, query, user_id, qtype, recall, clarification)

            process_evidence = await prepare_process_evidence_context(runtime, query, recall, original_fnames)
            resolved_targets = process_evidence["resolved_targets"]
            process_docs = process_evidence["process_docs"]
            display_seed_docs = process_evidence["display_seed_docs"]
            compare_process_groups = process_evidence["compare_process_groups"]
            observations = process_evidence["observations"]
            process_stage_trace = dict(process_evidence.get("process_stage_trace") or {})
            if process_stage_trace:
                recall = {
                    **dict(recall),
                    "source_resolution_trace": {
                        **dict(recall.get("source_resolution_trace") or {}),
                        "process_stage_trace": process_stage_trace,
                    },
                }

            observations = downgrade_evidence_refusal_to_prompt_warning(observations, process_docs)
            refusal_reason = process_refusal_reason(recall, resolved_targets, observations, process_docs)
            if refusal_reason:
                result = build_process_evidence_refusal_result(
                    runtime,
                    query,
                    user_id,
                    qtype,
                    recall,
                    process_docs,
                    observations,
                    refusal_reason,
                )
                metadata = dict(result.get("metadata") or {})
                server_timing = dict(metadata.get("server_timing_ms") or {})
                now = time.perf_counter()
                server_timing.update(
                    {
                        "recall": round(recall_ms * 1000, 1),
                        "pre_answer": round(max(0.0, (now - recall_done_at) * 1000), 1),
                        "draft_answer": 0.0,
                        "verify_answer": 0.0,
                        "answer": 0.0,
                        "handler_total": round((now - process_started) * 1000, 1),
                    }
                )
                metadata["server_timing_ms"] = server_timing
                result["metadata"] = metadata
                return result

            self._update_current_locked_document_state(user_id, recall, resolved_targets, observations)
            answer_context = prepare_answer_generation_context(
                runtime,
                query,
                qtype,
                recall,
                resolved_targets,
                process_docs,
                compare_process_groups,
                observations,
            )
            qtype = answer_context["qtype"]
            answer_mode = answer_context["answer_mode"]
            evidence = answer_context["evidence"]
            aspect_plan = answer_context["aspect_plan"]
            limits = answer_context["limits"]
            compare_refs = answer_context["compare_refs"]
            compare_answer_refs = answer_context["compare_answer_refs"]
            compare_source_statuses = answer_context["compare_source_statuses"]
            compare_target_count = answer_context["compare_target_count"]
            compare_matrix_mode = answer_context["compare_matrix_mode"]

            runtime.logger.info(
                f"DEBUG ANSWER INPUT -> qtype={qtype} answer_mode={answer_mode} "
                f"selected_docs={len(process_docs)} evidence_chars={len(evidence)}"
            )
            answer_started = time.perf_counter()
            structured_answer: Optional[Dict[str, Any]] = None
            structured_answer_origin = ""
            if compare_matrix_mode:
                answer = _build_compare_status_matrix_answer(
                    compare_source_statuses,
                    compare_answer_refs,
                    recall.get("compare_plan"),
                )
            else:
                if runtime.answer.should_use_structured_schema(qtype, answer_mode, aspect_plan):
                    answer, structured_answer = await self.generate_structured_answer(
                        query,
                        evidence,
                        qtype=qtype,
                        max_tokens=limits["max_tokens"],
                        answer_mode=answer_mode,
                        uncovered_aspects=list(observations.get("uncovered_aspects") or []),
                        aspect_plan=aspect_plan,
                        docs=process_docs,
                        evidence_gate_warning=str(observations.get("evidence_gate_warning") or ""),
                    )
                    if structured_answer:
                        structured_answer_origin = "llm_json"
                else:
                    answer = await self.generate_answer(
                        query,
                        evidence,
                        qtype=qtype,
                        max_tokens=limits["max_tokens"],
                        answer_mode=answer_mode,
                        compare_missing_targets=list(recall.get("compare_missing_targets") or []),
                        compare_source_status_hints=_compare_source_status_prompt_lines(compare_source_statuses),
                        uncovered_aspects=list(observations.get("uncovered_aspects") or []),
                        aspect_plan=aspect_plan,
                        evidence_gate_warning=str(observations.get("evidence_gate_warning") or ""),
                    )
            draft_answer_ms = time.perf_counter() - answer_started
            verify_answer_ms = 0.0
            if structured_answer:
                answer = runtime.answer.render_structured_markdown(structured_answer)
            answer_ms = draft_answer_ms + verify_answer_ms

            refusal_answer = runtime.guardrails.invalid_query_message("evidence_insufficient")
            finalized_answer = finalize_generated_answer(
                runtime,
                query,
                answer,
                qtype,
                answer_mode,
                evidence,
                aspect_plan,
                process_docs,
                observations,
                recall,
                structured_answer,
                compare_matrix_mode,
                compare_target_count,
                compare_answer_refs,
                compare_refs,
                refusal_answer,
            )
            answer = self._decorate_answer(finalized_answer["answer"], answer_mode)
            for event in finalized_answer["events"]:
                runtime.logger.info(event)

            result = build_process_success_result(
                runtime,
                query,
                user_id,
                answer,
                qtype,
                answer_mode,
                recall,
                resolved_targets,
                process_docs,
                display_seed_docs,
                observations,
                compare_source_statuses,
                structured_answer,
                structured_answer_origin,
                {
                    "recall": round(recall_ms * 1000, 1),
                    "pre_answer": round(max(0.0, (answer_started - recall_done_at) * 1000), 1),
                    "draft_answer": round(draft_answer_ms * 1000, 1),
                    "verify_answer": round(verify_answer_ms * 1000, 1),
                    "answer": round(answer_ms * 1000, 1),
                    "handler_total": round((time.perf_counter() - process_started) * 1000, 1),
                },
            )
            if not result["sources"] and process_docs:
                runtime.logger.info("obs: display_layer_cleared=1")
            display_docs_count = result.pop("display_docs_count", 0)
            runtime.logger.info(
                f"obs: context_chunk_count={len(process_docs)} "
                f"final_sources_count={len(result.get('sources') or [])} "
                f"display_docs_count={display_docs_count}"
            )
            return result
        except Exception as exc:
            runtime.logger.exception("Query processing error")
            raise HTTPException(status_code=500, detail=str(exc))
