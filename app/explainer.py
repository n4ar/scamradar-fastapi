import os
import httpx

_TYPHOON_BASE = "https://api.opentyphoon.ai/v1/chat/completions"

_FALLBACK = {
    "สูง": "• ข้อความ/ลิงก์นี้มีลักษณะต้องสงสัยสูงมาก",
    "ปานกลาง": "• ข้อความ/ลิงก์นี้มีบางส่วนที่น่าสงสัย",
    "ต่ำ": "• ไม่พบรูปแบบ scam ที่ชัดเจน",
}


async def explain(score: float, risk_level: str, evidence: list[str], original_text: str) -> tuple[str, str]:
    api_key = os.environ.get("TYPHOON_API_KEY")
    if not api_key:
        return _FALLBACK.get(risk_level, "• วิเคราะห์แล้ว"), "โปรดระวังและตรวจสอบแหล่งที่มา"

    evidence_str = "\n".join(f"- {e}" for e in evidence) or "- ไม่พบหลักฐานเพิ่มเติม"
    prompt = f"""คุณเป็นผู้เชี่ยวชาญด้านความปลอดภัยไซเบอร์ของไทย

ข้อมูลจากระบบวิเคราะห์:
- คะแนนความเสี่ยง: {score:.0%} (ระดับ{risk_level})
- หลักฐานที่พบ:
{evidence_str}
- ข้อความต้นฉบับ: "{original_text[:200]}"

เขียนคำอธิบายสั้นๆ ให้ผู้ใช้ทั่วไปเข้าใจได้ (ไม่เกิน 3 bullet points ขึ้นต้นด้วย •) และคำแนะนำ 1 บรรทัด
ตอบเป็นภาษาไทยเท่านั้น ห้ามใช้ศัพท์เทคนิค ตอบในรูปแบบ:
BULLETS:
• ...
• ...
ADVICE:
..."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "typhoon-v2.5-30b-a3b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.3,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(_TYPHOON_BASE, json=payload, headers=headers)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return _parse_response(content)
    except httpx.HTTPStatusError as e:
        print(f"Typhoon API Error: {e.response.status_code} - {e.response.text}")
        return _FALLBACK.get(risk_level, "• ไม่สามารถวิเคราะห์เพิ่มเติมได้"), "โปรดระวังและตรวจสอบแหล่งที่มา"
    except Exception as e:
        print(f"Typhoon API Error: {e}")
        return _FALLBACK.get(risk_level, "• ไม่สามารถวิเคราะห์เพิ่มเติมได้"), "โปรดระวังและตรวจสอบแหล่งที่มา"


def _parse_response(content: str) -> tuple[str, str]:
    lines = content.splitlines()
    in_bullets = False
    in_advice = False
    bullet_lines = []
    advice_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("BULLETS:"):
            in_bullets, in_advice = True, False
            # If there's content on the same line, though not expected from the prompt
            remaining = stripped.replace("BULLETS:", "").strip()
            if remaining: bullet_lines.append(remaining)
        elif stripped.startswith("ADVICE:"):
            in_bullets, in_advice = False, True
            remaining = stripped.replace("ADVICE:", "").strip()
            if remaining: advice_lines.append(remaining)
        elif in_bullets and stripped:
            bullet_lines.append(stripped)
        elif in_advice and stripped:
            advice_lines.append(stripped)
    
    bullets = "\n".join(bullet_lines) or "• วิเคราะห์แล้ว"
    advice = " ".join(advice_lines) or "โปรดระวัง"
    return bullets, advice
