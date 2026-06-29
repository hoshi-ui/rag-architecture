from typing import Any, Callable, Dict, List


def build_source_state(
    source: str,
    normalize_filename: Callable[[str], str],
    doc_get: Callable[[str], Dict[str, Any]],
    public_task_status: Callable[[Any], str],
) -> Dict[str, Any]:
    safe_source = normalize_filename(source or "")
    doc = doc_get(safe_source)
    status = public_task_status(doc.get("status")) if doc.get("status") else ""
    active_version = doc.get("active_version")
    pending_version = doc.get("pending_version")
    try:
        active_version = int(active_version) if active_version is not None else None
    except Exception:
        active_version = None
    try:
        pending_version = int(pending_version) if pending_version is not None else None
    except Exception:
        pending_version = None
    hidden_statuses = {"deleting", "delete_failed"}
    if status in hidden_statuses:
        visible = False
    else:
        visible = bool(active_version is not None) or (not status) or (status == "completed")
    return {
        "source": safe_source,
        "status": status,
        "active_version": active_version,
        "pending_version": pending_version,
        "visible": visible,
    }


def hit_matches_source_state(
    hit: Any,
    state: Dict[str, Any],
    hit_metadata: Callable[[Any], Dict[str, Any]],
) -> bool:
    if not state.get("visible"):
        return False
    active_version = state.get("active_version")
    if active_version is None:
        status = state.get("status") or ""
        return status in {"", "completed"}
    metadata = hit_metadata(hit)
    hit_version = metadata.get("doc_version")
    if hit_version is None:
        return True
    try:
        return int(hit_version) == int(active_version)
    except Exception:
        return True


def filter_hits_by_source_state(
    hits: List[Any],
    normalize_filename: Callable[[str], str],
    hit_entity_source: Callable[[Any], str],
    hit_metadata: Callable[[Any], Dict[str, Any]],
    source_state: Callable[[str], Dict[str, Any]],
) -> Dict[str, Any]:
    out: List[Any] = []
    dropped = 0
    states: Dict[str, Dict[str, Any]] = {}
    for hit in hits:
        source = normalize_filename(hit_entity_source(hit) or "")
        if source not in states:
            states[source] = source_state(source)
        if hit_matches_source_state(hit, states[source], hit_metadata):
            out.append(hit)
        else:
            dropped += 1
    return {"hits": out, "dropped": dropped, "states": states}
