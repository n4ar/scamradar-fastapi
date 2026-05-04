import pytest
from app.scorer import compute_risk_score

def test_text_only_high_risk():
    result = compute_risk_score(nlp_result={"scam_prob": 0.9, "confidence": "high"})
    assert result["score"] == pytest.approx(0.9)
    assert result["risk_level"] == "สูง"
    assert any("NLP" in e for e in result["evidence"])

def test_text_only_low_risk():
    result = compute_risk_score(nlp_result={"scam_prob": 0.1, "confidence": "high"})
    assert result["score"] == pytest.approx(0.1)
    assert result["risk_level"] == "ต่ำ"

def test_url_and_text():
    result = compute_risk_score(
        nlp_result={"scam_prob": 0.8, "confidence": "high"},
        url_result={"malicious": 10, "total_engines": 90, "domain_age_days": 5, "impersonation": None},
    )
    # score = 0.8*0.5 + (10/90)*0.5 + 0.1 (age<30) = 0.4 + 0.056 + 0.1 = 0.556
    assert result["score"] > 0.5
    assert result["risk_level"] == "ปานกลาง"

def test_domain_impersonation_bonus():
    result = compute_risk_score(
        url_result={"malicious": 0, "total_engines": 90, "domain_age_days": 100, "impersonation": "กสิกรไทย"},
    )
    assert result["score"] >= 0.2
    assert any("แอบอ้าง" in e for e in result["evidence"])

def test_score_capped_at_1():
    result = compute_risk_score(
        nlp_result={"scam_prob": 1.0, "confidence": "high"},
        url_result={"malicious": 90, "total_engines": 90, "domain_age_days": 1, "impersonation": "ธนาคาร"},
    )
    assert result["score"] <= 1.0

def test_line_oa_risk():
    result = compute_risk_score(
        url_result={"malicious": 0, "total_engines": 0, "domain_age_days": 999, "is_line_oa": True},
    )
    assert result["score"] >= 0.3
    assert any("Line OA" in e for e in result["evidence"])
