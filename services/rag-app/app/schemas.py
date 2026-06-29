from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentRequest(BaseModel):
    """Document upload request."""

    filename: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    """Query request."""

    query: str
    user_id: str = "anonymous"
    top_k: int = 10
    enable_rerank: bool = True


class QueryResponse(BaseModel):
    """Query response."""

    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_contexts: List[Dict[str, Any]] = Field(default_factory=list)
