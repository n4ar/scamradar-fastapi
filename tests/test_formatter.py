from app.formatter import format_response

def test_high_risk_format():
    msg = format_response(
        score=0.87,
        risk_level="สูง",
        explanation="• ข้อความแอบอ้างธนาคาร\n• ลิงก์ไปยัง domain ปลอม",
        advice="อย่ากดลิงก์ รายงาน ETDA 1212",
    )
    assert "🔴" in msg
    assert "87%" in msg
    assert "อย่ากดลิงก์" in msg

def test_medium_risk_format():
    msg = format_response(score=0.55, risk_level="ปานกลาง", explanation="• น่าสงสัย", advice="ระวัง")
    assert "🟡" in msg
    assert "55%" in msg

def test_low_risk_format():
    msg = format_response(score=0.1, risk_level="ต่ำ", explanation="• ปกติ", advice="ปลอดภัย")
    assert "🟢" in msg
    assert "10%" in msg
