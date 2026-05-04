import asyncio
import os
from app.pipelines import nlp as nlp_pipeline
from app.pipelines.url import check_url
from app.scorer import compute_risk_score
from app.formatter import format_response
from ml.classifier import load_vocab, load_interpreter

async def simulate_test(text: str):
    print(f"--- Input Message ---\n{text}\n")
    
    # Setup state
    vocab = load_vocab("ml/vocab.txt")
    interpreter = load_interpreter("ml/model.tflite")
    
    # 1. NLP Pipeline
    nlp_result = nlp_pipeline.run(text, interpreter, vocab)
    print(f"NLP Result: {nlp_result}")
    
    # 2. URL Pipeline (extract manually for test)
    import re
    urls = re.findall(r"https?://\S+", text)
    url_result = None
    if urls:
        print(f"Analyzing URL: {urls[0]}...")
        url_result = await check_url(urls[0])
        print(f"URL Result: {url_result}")
    
    # 3. Scorer
    score_result = compute_risk_score(nlp_result=nlp_result, url_result=url_result)
    print(f"Scorer Result: {score_result}")
    
    # 4. Formatter (Simulating advice from fallback)
    advice = "อย่ากดลิงก์ ห้ามกรอกข้อมูล และห้ามเพิ่มเพื่อน Line OA แปลกหน้า"
    explanation = "\n".join(f"• {e}" for e in score_result["evidence"])
    
    response = format_response(
        score=score_result["score"],
        risk_level=score_result["risk_level"],
        explanation=explanation,
        advice=advice
    )
    
    print(f"\n--- Final LINE Message ---\n{response}")

if __name__ == "__main__":
    # Test with the Line OA scam URL
    test_message = "ยินดีด้วยคุณได้รับสิทธิพิเศษ คลิกเลย https://bit.ly/3UEF6sc" 
    # Note: bit.ly/3UEF6sc is a dummy example, using the user's logic
    asyncio.run(simulate_test(test_message))
