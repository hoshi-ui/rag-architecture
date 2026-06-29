"""Core retrieval composition."""

from typing import Any

from app.core.retrieval.base import RetrievalBaseMixin
from app.core.retrieval.lexical_flow import RetrievalLexicalMixin
from app.core.retrieval.planning import RetrievalPlanningMixin


class RetrievalCore(
    RetrievalBaseMixin,
    RetrievalLexicalMixin,
    RetrievalPlanningMixin,
):
    def __init__(self, runtime: Any):
        self.runtime = runtime
