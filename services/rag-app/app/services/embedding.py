import time
from typing import Any, Dict, List, Optional, Tuple

import requests


class EmbeddingService:
    def __init__(self, embedding_url: str) -> None:
        self.embedding_url = embedding_url.rstrip("/")

    async def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings, _ = await self.embed_with_sparse(texts, return_sparse=False)
        return embeddings

    def embed_one_sync(self, text: str, timeout: int = 20) -> List[float]:
        response = requests.post(
            f"{self.embedding_url}/embed",
            json={
                "texts": [text],
                "normalize": True,
                "batch_size": 1,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings") or []
        return list(embeddings[0] or []) if embeddings else []

    @staticmethod
    def normalize_sparse_vector(value: Any) -> Optional[Dict[int, float]]:
        if not value:
            return None
        if isinstance(value, dict) and "indices" in value and "values" in value:
            raw_items = zip(value.get("indices") or [], value.get("values") or [])
        elif isinstance(value, dict):
            raw_items = value.items()
        else:
            raw_items = value
        out: Dict[int, float] = {}
        try:
            iterator = iter(raw_items)
        except TypeError:
            return None
        for item in iterator:
            try:
                key, weight = item
                token_id = int(key)
                score = float(weight)
            except Exception:
                continue
            if score > 0.0:
                out[token_id] = score
        return out or None

    @staticmethod
    def _parse_embedding_payload(payload: Dict[str, Any]) -> Tuple[List[List[float]], List[Optional[Dict[int, float]]]]:
        embeddings = [list(item or []) for item in (payload.get("embeddings") or [])]
        sparse_raw = payload.get("sparse_embeddings") or []
        sparse = [EmbeddingService.normalize_sparse_vector(item) for item in sparse_raw]
        if len(sparse) < len(embeddings):
            sparse.extend([None] * (len(embeddings) - len(sparse)))
        return embeddings, sparse[: len(embeddings)]

    async def embed_with_sparse(
        self,
        texts: List[str],
        *,
        return_sparse: bool = True,
        timeout: int = 30,
        batch_size: int = 32,
    ) -> Tuple[List[List[float]], List[Optional[Dict[int, float]]]]:
        response = requests.post(
            f"{self.embedding_url}/embed",
            json={
                "texts": texts,
                "normalize": True,
                "batch_size": batch_size,
                "return_sparse": bool(return_sparse),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return self._parse_embedding_payload(response.json())

    async def embed_batched(
        self,
        texts: List[str],
        per_request: int = 64,
        timeout: int = 60,
        retries: int = 2,
    ) -> List[List[float]]:
        embeddings, _ = await self.embed_batched_with_sparse(
            texts,
            per_request=per_request,
            timeout=timeout,
            retries=retries,
            return_sparse=False,
        )
        return embeddings

    async def embed_batched_with_sparse(
        self,
        texts: List[str],
        per_request: int = 64,
        timeout: int = 60,
        retries: int = 2,
        *,
        return_sparse: bool = True,
    ) -> Tuple[List[List[float]], List[Optional[Dict[int, float]]]]:
        out: List[List[float]] = []
        sparse_out: List[Optional[Dict[int, float]]] = []
        total = len(texts)
        index = 0
        while index < total:
            batch = texts[index : index + per_request]
            attempt = 0
            while True:
                try:
                    response = requests.post(
                        f"{self.embedding_url}/embed",
                        json={
                            "texts": batch,
                            "normalize": True,
                            "batch_size": min(32, per_request),
                            "return_sparse": bool(return_sparse),
                        },
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    embeddings, sparse_embeddings = self._parse_embedding_payload(response.json())
                    out.extend(embeddings)
                    sparse_out.extend(sparse_embeddings)
                    break
                except Exception as exc:
                    if attempt >= retries:
                        raise exc
                    time.sleep(min(2**attempt, 4))
                    attempt += 1
            index += per_request
        if len(sparse_out) < len(out):
            sparse_out.extend([None] * (len(out) - len(sparse_out)))
        return out, sparse_out[: len(out)]
