import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_INPUT = os.path.abspath(os.path.join(THIS_DIR, "..", "rag-app", "uploads", "training", "reranker_round1", "reranker_pairs.jsonl"))
DEFAULT_OUTPUT = os.path.join(THIS_DIR, "artifacts", "finetuned-reranker")
JSON_BEGIN = "===LEGAL_RERANK_TRAIN_JSON_BEGIN==="
JSON_END = "===LEGAL_RERANK_TRAIN_JSON_END==="


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the reranker on legal RAG pairs.")
    parser.add_argument("--inside-container", action="store_true")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "BAAI/bge-reranker-base"))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _inside_container() -> bool:
    return os.path.exists("/.dockerenv")


def _extract_embedded_json(output: str) -> Dict[str, Any]:
    start = output.find(JSON_BEGIN)
    end = output.find(JSON_END)
    if start < 0 or end <= start:
        raise RuntimeError("reranker training did not emit embedded JSON")
    return json.loads(output[start + len(JSON_BEGIN):end].strip())


def _resolve_model_name_or_path(model_name: str) -> str:
    value = str(model_name or "").strip()
    if not value:
        value = "BAAI/bge-reranker-base"
    if os.path.exists(value):
        return os.path.abspath(value)

    candidates: List[str] = []
    env_model = str(os.getenv("MODEL_NAME", "")).strip()
    if env_model:
        candidates.append(env_model)
    if value == "BAAI/bge-reranker-base":
        candidates.extend([
            "/models/BAAI/bge-reranker-base",
            os.path.abspath(os.path.join(THIS_DIR, "..", "..", "local-models", "BAAI", "bge-reranker-base")),
        ])
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return value


def _run_via_container(args: argparse.Namespace) -> None:
    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output_dir)
    container_script = "/app/train_reranker.py"
    container_input_dir = "/tmp/legal_rerank_training"
    container_input = os.path.join(container_input_dir, os.path.basename(input_path))
    container_output = os.path.join(container_input_dir, "output")

    subprocess.run(["docker", "exec", "rerank-service", "mkdir", "-p", container_input_dir], check=True)
    subprocess.run(["docker", "cp", __file__, f"rerank-service:{container_script}"], check=True)
    subprocess.run(["docker", "cp", input_path, f"rerank-service:{container_input}"], check=True)

    resolved_model = _resolve_model_name_or_path(args.model_name)
    cmd = [
        "docker", "exec", "rerank-service", "python", container_script,
        "--inside-container",
        "--input", container_input,
        "--output-dir", container_output,
        "--model-name", resolved_model,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--warmup-ratio", str(args.warmup_ratio),
        "--max-length", str(args.max_length),
        "--dev-ratio", str(args.dev_ratio),
        "--seed", str(args.seed),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr or result.stdout)
        raise SystemExit(result.returncode)
    summary = _extract_embedded_json(result.stdout)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    subprocess.run(["docker", "cp", f"rerank-service:{container_output}", output_dir], check=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _load_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _to_examples(rows: List[Dict[str, Any]], input_example_cls: Any) -> List[Any]:
    examples: List[Any] = []
    for row in rows:
        query = str(row.get("query") or "").strip()
        passage = str(row.get("passage") or "").strip()
        if not query or not passage:
            continue
        label = float(row.get("label", 0.0))
        examples.append(input_example_cls(texts=[query, passage], label=label))
    return examples


def main() -> None:
    args = _parse_args()
    if not args.inside_container and not _inside_container():
        _run_via_container(args)
        return

    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    rows = _load_rows(os.path.abspath(args.input))
    if not rows:
        raise SystemExit("no reranker pairs found")
    random.shuffle(rows)
    cut = int(len(rows) * max(0.0, min(args.dev_ratio, 0.4)))
    dev_rows = rows[:cut]
    train_rows = rows[cut:] if cut < len(rows) else rows
    train_examples = _to_examples(train_rows, InputExample)
    dev_examples = _to_examples(dev_rows, InputExample)
    if not train_examples:
        raise SystemExit("no valid training examples")

    resolved_model = _resolve_model_name_or_path(args.model_name)
    model = CrossEncoder(resolved_model, max_length=args.max_length)
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    warmup_steps = max(1, int(len(train_loader) * max(1, args.epochs) * args.warmup_ratio))

    os.makedirs(args.output_dir, exist_ok=True)
    model.fit(
        train_dataloader=train_loader,
        evaluator=None,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        optimizer_params={"lr": args.learning_rate},
        use_amp=False,
        show_progress_bar=True,
    )
    model.save(args.output_dir)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": os.path.abspath(args.input),
        "output_dir": os.path.abspath(args.output_dir),
        "model_name": resolved_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
    }
    with open(os.path.join(args.output_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(JSON_BEGIN)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(JSON_END)


if __name__ == "__main__":
    main()