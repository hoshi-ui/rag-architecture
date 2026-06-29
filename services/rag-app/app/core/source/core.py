"""Core source-resolution composition."""

from typing import Any, Dict, Optional, Tuple

from app.core.source.documents import SourceDocumentMixin
from app.core.source.profile import SourceProfileMixin
from app.core.source.resolution_flow import SourceResolutionMixin


class SourceCore(
    SourceDocumentMixin,
    SourceProfileMixin,
    SourceResolutionMixin,
):
    def __init__(self, runtime: Any):
        self.runtime = runtime
        self._embedding_cache: Dict[str, Tuple[float, ...]] = {}
        self._dense_title_probe_cache: Optional[Tuple[Tuple[str, str, str], ...]] = None
