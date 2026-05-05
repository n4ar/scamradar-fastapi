import json

# This structure matches the linebot v3 FlexMessage requirements
def build_vote_flex(msg_hash: str):
    return {
        "type": "flex",
        "altText": "ข้อความนี้เป็น scam จริงไหม?",
        "contents": {
            "type": "bubble",
            "size": "kilo",
            "body": {
                "type": "box",
                "layout": "vertical",
                "paddingAll": "md",
                "contents": [
                    {"type": "text", "text": "ช่วยปกป้องชุมชนด้วย 🙏", "weight": "bold", "size": "sm"},
                    {"type": "text", "text": "ข้อความนี้เป็น scam จริงไหม?", 
                     "size": "xs", "color": "#888888", "wrap": True},
                    {"type": "text", "text": "*ข้อมูลจะถูกจัดเก็บแบบไม่ระบุตัวตนและปกปิดข้อมูลส่วนบุคคล เพื่อนำไปพัฒนา AI ให้เก่งขึ้น",
                     "size": "xxs", "color": "#B0B0B0", "wrap": True, "margin": "sm"}
                ]
            },
            "footer": {
                "type": "box",
                "layout": "horizontal",
                "spacing": "sm",
                "contents": [
                    {
                        "type": "button", "style": "primary", "color": "#E53E3E", "height": "sm",
                        "action": {
                            "type": "postback",
                            "label": "ใช่ scam",
                            "data": f"action=vote&result=scam&msg_id={msg_hash}"
                        }
                    },
                    {
                        "type": "button", "style": "secondary", "height": "sm",
                        "action": {
                            "type": "postback",
                            "label": "ไม่ใช่",
                            "data": f"action=vote&result=safe&msg_id={msg_hash}"
                        }
                    }
                ]
            }
        }
    }
