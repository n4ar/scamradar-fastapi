import asyncio
import os
from dotenv import load_dotenv
from app.pipelines.ocr import extract_text

load_dotenv()

async def main():
    with open('/Users/near/Downloads/S__11477016_0.jpg', "rb") as f:
        image_bytes = f.read()
    text = await extract_text(image_bytes)
    print(f"TEXT 1: {text}")

    with open('/Users/near/Downloads/S__11477017_0.jpg', "rb") as f:
        image_bytes = f.read()
    text = await extract_text(image_bytes)
    print(f"TEXT 2: {text}")

asyncio.run(main())
