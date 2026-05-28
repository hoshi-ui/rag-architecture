import argparse
import json
import os
import random
import time
from typing import Any, Dict, List


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_INPUT = os.path.abspath(os.path.join(THIS_DIR, "..", "rag-app", "uploads", "training", "reranker_round1", "embedding_triples.jsonl"))
DEFAULT_OUTPUT = os.path.join(THIS_DIR, "artifacts", "finetuned-embedding")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the embedding model on legal RAG triplets.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "BAAI/bge-m3"))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
        positive = str(row.get("positive") or "").strip()
        negative = str(row.get("hard_negative") or "").strip()
        if not query or not positive or not negative:
            continue
        examples.append(input_example_cls(texts=[query, positive, negative]))
    return examples


def main() -> None:
    args = _parse_args()
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    rows = _load_rows(os.path.abspath(args.input))
    if not rows:
        raise SystemExit("no embedding triplets found")
    random.shuffle(rows)
    examples = _to_examples(rows, InputExample)
    if not examples:
        raise SystemExit("no valid embedding examples")

    model = SentenceTransformer(args.model_name)
    train_loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.TripletLoss(model)
    warmup_steps = max(1, int(len(train_loader) * max(1, args.epochs) * args.warmup_ratio))

    os.makedirs(args.output_dir, exist_ok=True)
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        optimizer_params={"lr": args.learning_rate},
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        show_progress_bar=True,
    )

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": os.path.abspath(args.input),
        "output_dir": os.path.abspath(args.output_dir),
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "triplets": len(examples),
    }
    with open(os.path.join(args.output_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()