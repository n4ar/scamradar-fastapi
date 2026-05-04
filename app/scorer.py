def compute_risk_score(nlp_result: dict = None, url_result: dict = None) -> dict:
    score = 0.0
    evidence = []
    has_url = url_result is not None

    if nlp_result:
        w = 0.5 if has_url else 1.0
        score += nlp_result["scam_prob"] * w
        if nlp_result["scam_prob"] > 0.6:
            evidence.append(f"NLP: ข้อความมีลักษณะ SMS scam (confidence: {nlp_result['confidence']})")

    if url_result:
        total = url_result["total_engines"] or 1
        vt_ratio = url_result["malicious"] / total
        score += vt_ratio * 0.5
        if url_result["malicious"] > 0:
            evidence.append(f"URL: VirusTotal พบ {url_result['malicious']}/{url_result['total_engines']} engines")
        if url_result.get("domain_age_days", 999) < 30:
            score += 0.1
            evidence.append(f"Domain: อายุเพียง {url_result['domain_age_days']} วัน (น่าสงสัย)")
        if url_result.get("impersonation"):
            score += 0.2
            evidence.append(f"Domain: แอบอ้างเป็น {url_result['impersonation']}")
        if url_result.get("is_line_oa"):
            score += 0.3
            evidence.append("URL: ลิงก์นำทางไปยังการเพิ่มเพื่อน Line OA (เสี่ยงสูง)")

    score = min(score, 1.0)
    risk_level = "สูง" if score >= 0.7 else "ปานกลาง" if score >= 0.4 else "ต่ำ"
    return {"score": score, "risk_level": risk_level, "evidence": evidence}
