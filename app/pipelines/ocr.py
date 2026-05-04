import asyncio
import base64
import hashlib
import os
import httpx

_TYPHOON_BASE = "https://api.opentyphoon.ai/v1/chat/completions"
_SEMAPHORE = asyncio.Semaphore(1)  # 2 req/s limit
_cache: dict[str, str] = {}  # sha256(image_bytes) → extracted_text


async def extract_text(image_bytes: bytes) -> str:
    img_hash = hashlib.sha256(image_bytes).hexdigest()
    if img_hash in _cache:
        return _cache[img_hash]

    async with _SEMAPHORE:
        text = await _call_typhoon_ocr(image_bytes)

    _cache[img_hash] = text
    return text


async def _call_typhoon_ocr(image_bytes: bytes) -> str:
    api_key = os.environ.get("TYPHOON_API_KEY")
    if not api_key:
        return ""

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
            r = await client.post(_TYPHOON_BASE, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""
