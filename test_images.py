import asyncio
from app.pipelines import ocr as ocr_pipeline
from app.pipelines import nlp as nlp_pipeline
from app.pipelines.qr import decode_qr
from app.pipelines.url import check_url
from app.router import extract_urls
from ml.classifier import load_vocab, load_interpreter
from app.scorer import compute_risk_score

async def test_image(file_path):
    print(f"\n--- Testing Image: {file_path} ---")
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    vocab = load_vocab("ml/vocab.txt")
    interpreter = load_interpreter("ml/model.tflite")

    # 1. QR
    qr_url = decode_qr(image_bytes)
    if qr_url:
        print(f"Found QR URL: {qr_url}")
        url_result = await check_url(qr_url)
        print(f"QR URL Score: {url_result}")
        score_result = compute_risk_score(url_result=url_result)
        print(f"Final Score Result: {score_result}")
        return

    # 2. OCR
    text = await ocr_pipeline.extract_text(image_bytes)
    print(f"OCR Text: {text}")

    # 3. NLP
    nlp_result = nlp_pipeline.run(text, interpreter, vocab)
    print(f"NLP Score: {nlp_result}")

    # 4. URL
    urls = extract_urls(text)
    url_result = None
    if urls:
        print(f"Found URL from OCR: {urls[0]}")
        url_result = await check_url(urls[0])
        print(f"URL Result: {url_result}")

    # 5. Final
    score_result = compute_risk_score(nlp_result=nlp_result, url_result=url_result)
    print(f"Final Score Result: {score_result}")

async def main():
    await test_image('/Users/near/Downloads/S__11477016_0.jpg')
    await test_image('/Users/near/Downloads/S__11477017_0.jpg')

asyncio.run(main())
