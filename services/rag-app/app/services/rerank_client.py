from typing import Dict, List

import requests


class RerankService:
    def __init__(self, rerank_url: str) -> None:
        self.rerank_url = rerank_url.rstrip("/")

    async def rerank(self, query: str, documents: List[str], top_k: int) -> List[Dict]:
        response = requests.post(
            f"{self.rerank_url}/rerank",
            json={
                "query": query,
                "documents": documents,
                "top_n": top_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["results"]

