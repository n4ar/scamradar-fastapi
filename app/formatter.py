_ICONS = {"สูง": "🔴", "ปานกลาง": "🟡", "ต่ำ": "🟢"}

def format_response(score: float, risk_level: str, explanation: str, advice: str) -> str:
    icon = _ICONS.get(risk_level, "⚪")
    pct = int(round(score * 100))
    return (
        f"{icon} ความเสี่ยง{risk_level} ({pct}%)\n\n"
        f"⚠️ ที่พบ:\n{explanation}\n\n"
        f"📋 คำแนะนำ: {advice}"
    )
