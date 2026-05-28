import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "services" / "rag-app" / "main.py"


def load_module():
    spec = importlib.util.spec_from_file_location("rag_reconcile_main", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Backfill SQLite control-plane and lexical tables from Milvus.")
    parser.add_argument("--prune-sqlite-orphans", action="store_true", help="Delete SQLite-only sources that no longer exist in Milvus.")
    parser.add_argument("--source", action="append", default=[], help="Reconcile only the given source. Repeatable.")
    args = parser.parse_args()

    module = load_module()
    report = module.reconcile_sqlite_with_milvus(
        prune_sqlite_orphans=args.prune_sqlite_orphans,
        sources=args.source or None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()