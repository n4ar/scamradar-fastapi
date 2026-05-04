import os
import asyncio
from fastapi import Request, HTTPException
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi, MessagingApiBlob,
    ReplyMessageRequest, PushMessageRequest, TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent
from app.router import detect_type
from app.pipelines import nlp as nlp_pipeline
from app.scorer import compute_risk_score
from app.formatter import format_response
_token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
_secret = os.environ.get("LINE_CHANNEL_SECRET")

_handler = WebhookHandler(os.environ.get("LINE_CHANNEL_SECRET", "dummy"))

def get_config():
    return Configuration(access_token=os.environ.get("LINE_CHANNEL_ACCESS_TOKEN"))

async def handle_webhook(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = (await request.body()).decode()
    
    try:
        _handler.handle(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return "OK"


def _reply(reply_token: str, text: str):
    with ApiClient(get_config()) as client:
        MessagingApi(client).reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text=text)])
        )


def _push(user_id: str, text: str):
    with ApiClient(get_config()) as client:
        MessagingApi(client).push_message(
            PushMessageRequest(to=user_id, messages=[TextMessage(text=text)])
        )

async def _build_response_async(score_result: dict, original_text: str = "") -> str:
    from app.explainer import explain
    bullets, advice = await explain(
        score=score_result["score"],
        risk_level=score_result["risk_level"],
        evidence=score_result["evidence"],
        original_text=original_text,
    )
    return format_response(
        score=score_result["score"],
        risk_level=score_result["risk_level"],
        explanation=bullets,
        advice=advice,
    )


async def _process_text(text: str, state) -> str:
    nlp_result = nlp_pipeline.run(text, state.interpreter, state.vocab)
    score_result = compute_risk_score(nlp_result=nlp_result)
    return await _build_response_async(score_result, text)


async def _process_text_with_url(text: str, state) -> str:
    from app.router import extract_urls
    from app.pipelines.url import check_url

    urls = extract_urls(text)
    nlp_result = nlp_pipeline.run(text, state.interpreter, state.vocab)

    url_result = None
    if urls:
        url_result = await check_url(urls[0])  # check first URL only

    score_result = compute_risk_score(nlp_result=nlp_result, url_result=url_result)
    return await _build_response_async(score_result, text)


@_handler.add(MessageEvent, message=TextMessageContent)
def on_text(event: MessageEvent):
    from app.main import app as _app
    state = _app.state
    msg_type = detect_type(event)

    if msg_type == "unsupported":
        _reply(event.reply_token, "ขอโทษ ไม่รองรับรูปแบบนี้")
        return

    _reply(event.reply_token, "🔍 กำลังวิเคราะห์...")
    user_id = event.source.user_id

    async def _run():
        if msg_type == "text":
            result_text = await _process_text(event.message.text, state)
        elif msg_type in ("url", "text_with_url"):
            result_text = await _process_text_with_url(event.message.text, state)
        else:
            result_text = "ไม่รองรับรูปแบบนี้"
        
        _push(user_id, result_text)

    try:
        asyncio.create_task(_run())
    except Exception as e:
        print(f"Error processing text: {e}")


@_handler.add(MessageEvent, message=ImageMessageContent)
def on_image(event: MessageEvent):
    from app.main import app as _app
    from app.pipelines import ocr as ocr_pipeline
    from app.router import extract_urls
    from app.pipelines.url import check_url
    from app.pipelines.qr import decode_qr

    state = _app.state
    _reply(event.reply_token, "🔍 กำลังวิเคราะห์รูปภาพ...")
    user_id = event.source.user_id

    async def _run():
        with ApiClient(get_config()) as client:
            blob_api = MessagingApiBlob(client)
            image_bytes = blob_api.get_message_content(event.message.id)
        image_bytes = bytes(image_bytes)

        # Try QR decode first
        qr_url = decode_qr(image_bytes)
        if qr_url:
            url_result = await check_url(qr_url)
            score_result = compute_risk_score(url_result=url_result)
            result_text = await _build_response_async(score_result, f"QR Code: {qr_url}")
            _push(user_id, result_text)
            return

        # Fall back to OCR
        extracted_text = await ocr_pipeline.extract_text(image_bytes)
        if not extracted_text:
            _push(user_id, "❌ ไม่สามารถอ่านข้อความจากรูปภาพได้")
            return

        nlp_result = nlp_pipeline.run(extracted_text, state.interpreter, state.vocab)
        urls = extract_urls(extracted_text)
        url_result = await check_url(urls[0]) if urls else None
        score_result = compute_risk_score(nlp_result=nlp_result, url_result=url_result)
        result_text = await _build_response_async(score_result, extracted_text)
        _push(user_id, result_text)

    try:
        asyncio.create_task(_run())
    except Exception as e:
        print(f"Error processing image: {e}")
