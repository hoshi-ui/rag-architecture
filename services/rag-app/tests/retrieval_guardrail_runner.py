import json
import os
import subprocess
import sys
import time
from typing import Any, Dict


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

CHINESE_REPORT = os.path.join(UPLOAD_DIR, "chinese_retrieval_report.json")
BASELINE_REPORT = os.path.join(UPLOAD_DIR, "real_regulation_baseline_report.json")
ABLATION_REPORT = os.path.join(UPLOAD_DIR, "live_ablation_report.json")
SUITE_REPORT = os.path.join(UPLOAD_DIR, "retrieval_guardrail_report.json")


def _run_script(script_name: str) -> None:
    script_path = os.path.join(THIS_DIR, script_name)
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr or result.stdout)
        raise SystemExit(result.returncode)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _focus_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    return (((report or {}).get("summary") or {}).get("focus_metrics") or {})


def _ablation_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    return ((((report or {}).get("summary") or {}).get("rerank_ablation") or {}).get("hybrid_fusion_rerank") or {})


def main() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    _run_script("chinese_retrieval_runner.py")
    _run_script("real_regulation_baseline.py")
    _run_script("live_ablation_runner.py")

    chinese_report = _load_json(CHINESE_REPORT)
    baseline_report = _load_json(BASELINE_REPORT)
    ablation_report = _load_json(ABLATION_REPORT)

    chinese_metrics = _focus_metrics(chinese_report)
    baseline_metrics = _focus_metrics(baseline_report)
    ablation_metrics = _ablation_metrics(ablation_report)

    summary = {
        "guardrails": {
            "chinese_retrieval": {
                "wrong_source_rate": chinese_metrics.get("wrong_source_rate", 0.0),
                "mis_refusal_rate": chinese_metrics.get("mis_refusal_rate", 0.0),
                "negative_clean_rate": chinese_metrics.get("negative_clean_rate", 0.0),
            },
            "real_baseline": {
                "wrong_source_rate": baseline_metrics.get("wrong_source_rate", 0.0),
                "mis_refusal_rate": baseline_metrics.get("mis_refusal_rate", 0.0),
                "negative_clean_rate": baseline_metrics.get("negative_clean_rate", 0.0),
            },
            "live_ablation_hybrid_fusion_rerank": {
                "wrong_source_rate": ablation_metrics.get("wrong_source_rate", 0.0),
                "mis_refusal_rate": ablation_metrics.get("mis_refusal_rate", 0.0),
                "negative_clean_rate": ablation_metrics.get("negative_clean_rate", 0.0),
                "positive_hit_rate_top3": ablation_metrics.get("positive_hit_rate_top3", 0.0),
            },
        },
        "pass": {
            "wrong_source_guardrail": all(metric.get("wrong_source_rate", 1.0) <= 0.0 for metric in (chinese_metrics, baseline_metrics, ablation_metrics)),
            "negative_clean_guardrail": all(metric.get("negative_clean_rate", 0.0) >= 1.0 for metric in (chinese_metrics, baseline_metrics, ablation_metrics)),
            "mis_refusal_tracking": True,
        },
        "failure_buckets": {
            "chinese": (((chinese_report.get("summary") or {}).get("focus_failures") or {})),
            "baseline": (((baseline_report.get("summary") or {}).get("focus_failures") or {})),
            "ablation_rerank_deltas": (((ablation_report.get("summary") or {}).get("rerank_deltas") or {})),
        },
    }
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reports": {
            "chinese_retrieval_report": CHINESE_REPORT,
            "real_baseline_report": BASELINE_REPORT,
            "live_ablation_report": ABLATION_REPORT,
        },
        "summary": summary,
    }
    with open(SUITE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()