import asyncio
import os
from dotenv import load_dotenv
import httpx
import base64

load_dotenv()

async def run_ocr(file_path):
    print(f"Testing {file_path}")
    api_key = os.environ.get("TYPHOON_API_KEY")
    with open(file_path, "rb") as f:
        image_bytes = f.read()
    
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": "typhoon-ocr-preview",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "สกัดข้อความทั้งหมดในภาพ ตอบเฉพาะข้อความที่เห็นในภาพเท่านั้น"},
            ],
        }],
        "max_tokens": 500,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post("https://api.opentyphoon.ai/v1/chat/completions", json=payload, headers=headers)
            print(f"Status: {r.status_code}")
            print(f"Response: {r.text[:500]}")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    await run_ocr('/Users/near/Downloads/S__11477016_0.jpg')
    await run_ocr('/Users/near/Downloads/S__11477017_0.jpg')

asyncio.run(main())
