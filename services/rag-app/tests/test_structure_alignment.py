from app.core.retrieval import rerank
from app.utils import text as text_utils


FORBIDDEN_CORE_API_IMPORT_PREFIXES = ("app.storage", "app.services")


class _Runtime:
    class config:
        HYBRID_STRUCT_PROFILE_WEIGHTS = {
            "balanced": {
                "base": 0.25,
            }
        }


def test_runtime_context_exposes_flat_core_services():
    from app.runtime import runtime_context

    context = runtime_context()
    assert callable(context.query_core().process)
    assert callable(context.query_core().retrieve)
    assert callable(context.document_service().list_tasks)
    assert callable(context.document_service().get_task)


def test_api_and_core_do_not_import_bottom_layer_modules():
    import ast
    from pathlib import Path

    app_dir = Path(__file__).resolve().parents[1] / "app"
    checked_roots = [app_dir / "api", app_dir / "core"]
    violations = []

    for root in checked_roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                module = ""
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.name or ""
                        if name.startswith(FORBIDDEN_CORE_API_IMPORT_PREFIXES):
                            violations.append((path.relative_to(app_dir.parent), name))
                    continue
                if module.startswith(FORBIDDEN_CORE_API_IMPORT_PREFIXES):
                    violations.append((path.relative_to(app_dir.parent), module))

    assert violations == []


def test_rerank_profile_weights_merge_configured_values():
    weights = rerank.rerank_profile_weights(_Runtime(), "balanced")

    assert weights["base"] == 0.25
    assert weights["section_term"] == 0.22


def test_chinese_utils_keep_text_compatibility_behavior():
    assert text_utils.normalize_query(" a  b ") == "a b"
    assert text_utils.dedupe_keep_order([" a ", "a", "", "b"]) == ["a", "b"]
