_SCAM_KEYWORDS = [
    "แจกหนัก", "ได้จริง", "ถอน", "คลิกเลย", "รับสิทธิ์", "โบนัส", "ฟรี", "ด่วน", "เงินสด",
    "สล็อต", "บาคาร่า", "เครดิตฟรี", "เว็บตรง", "คาสิโน", "ฝาก-ถอน", "แจกฟรี", "โปรโมชั่น",
    "รับโปร", "สปิน", "แตกง่าย", "ยิงปลา", "พนัน"
]

def compute_risk_score(nlp_result: dict = None, url_result: dict = None, original_text: str = "") -> dict:
    score = 0.0
    evidence = []
    has_url = url_result is not None

    if nlp_result:
        w = 0.5 if has_url else 1.0
        score += nlp_result["scam_prob"] * w
        if nlp_result["scam_prob"] > 0.6:
            evidence.append(f"NLP: ข้อความมีลักษณะ SMS scam (confidence: {nlp_result['confidence']})")
            
    # Rule-based fallback for text
    keyword_hits = [k for k in _SCAM_KEYWORDS if k in original_text]
    if keyword_hits:
        keyword_score = min(len(keyword_hits) * 0.15, 0.4)
        score += keyword_score
        evidence.append(f"Keyword: พบคำที่มักใช้ในการหลอกลวง ({', '.join(keyword_hits)})")

    if url_result:
        total = url_result["total_engines"] or 1
        
        if url_result["malicious"] > 0:
            # If EVEN ONE engine flags it, it's highly suspicious
            score += 0.5 + (url_result["malicious"] / total) * 0.5
            evidence.append(f"URL: VirusTotal พบ {url_result['malicious']}/{url_result['total_engines']} engines (อันตราย)")
            
        if url_result.get("domain_age_days", 999) < 30:
            score += 0.1
            evidence.append(f"Domain: อายุเพียง {url_result['domain_age_days']} วัน (น่าสงสัย)")
            
        if url_result.get("is_suspicious_tld"):
            score += 0.3
            evidence.append("Domain: ใช้นามสกุลโดเมนที่มักใช้ในเว็บพนันหรือสแกม (เช่น .bet, .vip)")
            
        if url_result.get("impersonation"):
            score += 0.6  # Heavy penalty for impersonating a brand
            evidence.append(f"Domain: แอบอ้างเป็น {url_result['impersonation']} (เสี่ยงสูงมาก)")
            
        if url_result.get("is_line_oa"):
            score += 0.5  # Heavy penalty for short links to Line OA
            evidence.append("URL: ลิงก์นำทางไปยังการเพิ่มเพื่อน Line OA (เสี่ยงสูง)")

    # Ensure strong indicators push the risk to at least "Medium" or "High"
    if url_result:
        if url_result.get("impersonation") or url_result.get("malicious", 0) > 0:
            score = max(score, 0.8)
        elif url_result.get("is_line_oa") or url_result.get("is_suspicious_tld"):
            score = max(score, 0.5)
        
    score = min(score, 1.0)
    risk_level = "สูง" if score >= 0.7 else "ปานกลาง" if score >= 0.4 else "ต่ำ"
    return {"score": score, "risk_level": risk_level, "evidence": evidence}
