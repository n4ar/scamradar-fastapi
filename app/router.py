import re
from linebot.v3.webhooks import TextMessageContent, ImageMessageContent

_URL_RE = re.compile(r"https?://\S+")

def detect_type(event) -> str:
    if isinstance(event.message, ImageMessageContent):
        return "image"
    if isinstance(event.message, TextMessageContent):
        text = event.message.text.strip()
        if _URL_RE.fullmatch(text):
            return "url"
        if _URL_RE.search(text):
            return "text_with_url"
        return "text"
    return "unsupported"

def extract_urls(text: str) -> list[str]:
    return _URL_RE.findall(text)
