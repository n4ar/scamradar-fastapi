import re
from linebot.v3.webhooks import TextMessageContent, ImageMessageContent

_URL_RE = re.compile(r"(?:https?://)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?::\d+)?(?:/[^\s]*)?")

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
    urls = _URL_RE.findall(text)
    clean_urls = []
    for u in urls:
        if not u.startswith("http"):
            clean_urls.append(f"https://{u}")
        else:
            clean_urls.append(u)
    return clean_urls
