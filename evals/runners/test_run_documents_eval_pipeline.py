import argparse

from evals.runners import run_documents_eval_pipeline as pipeline


def test_judge_config_falls_back_to_llm_api_env(monkeypatch):
    monkeypatch.delenv("EVAL_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_API_BASE", "http://judge.local/v1")

    args = argparse.Namespace(judge_base_url="")

    assert pipeline._configured_judge_base_url(args) == "http://judge.local/v1"
    assert pipeline._judge_chat_completions_url("http://judge.local/v1") == "http://judge.local/v1/chat/completions"


def test_normalize_judge_scores_accepts_wrapped_string_numbers():
    scores = pipeline._normalize_judge_scores(
        {
            "scores": {
                "faithfulness": "0.8",
                "answer_relevance": "1/1",
                "legal_correctness": "0.75",
                "score_0_5": "4.5",
            },
            "reason": "ok",
        }
    )

    assert scores["faithfulness"] == 0.8
    assert scores["answer_relevance"] == 1.0
    assert scores["legal_correctness"] == 0.75
    assert scores["score_0_5"] == 4.5
    assert scores["reason"] == "ok"
