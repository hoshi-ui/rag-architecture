from app.core.intent_classifier import normalize_payload
from app.core.legal_subjects import normalize_subject_terms


def test_subject_dictionary_expands_behavior_and_responsibility_entities():
    assert normalize_subject_terms(["养犬行为"]) == ["养犬行为", "养犬", "养犬人", "养犬单位"]
    assert normalize_subject_terms(["公安机关职责"]) == ["公安机关职责", "公安机关"]


def test_intent_payload_applies_subject_dictionary_constraints():
    payload = normalize_payload(
        {
            "target_subject": ["养犬行为"],
            "excluded_subject": ["公安机关职责"],
            "confidence": 0.95,
        }
    )

    assert payload["target_subject"] == ["养犬行为", "养犬", "养犬人", "养犬单位"]
    assert payload["excluded_subject"] == ["公安机关职责", "公安机关"]
