import os
import asyncio
from dotenv import load_dotenv
from app.explainer import explain

load_dotenv()

async def run_test():
    api_key = os.environ.get("TYPHOON_API_KEY")
    if not api_key:
        print("ERROR: TYPHOON_API_KEY is not set.")
        return
    print("API Key found (length: {})".format(len(api_key)))
    
    score = 0.99
    risk_level = "สูง"
    evidence = ["NLP: ข้อความมีลักษณะ SMS scam (confidence: high)", "URL: ลิงก์นำทางไปยังการเพิ่มเพื่อน Line OA (เสี่ยงสูง)"]
    original_text = "รับโปรโมชั่นพิเศษคลิก https://line.me/R/ti/p/@705gscam"
    
    print("Calling Typhoon API...")
    try:
        bullets, advice = await explain(score, risk_level, evidence, original_text)
        print("--- Result ---")
        print(bullets)
        print(advice)
    except Exception as e:
        print("Exception:", e)

asyncio.run(run_test())
