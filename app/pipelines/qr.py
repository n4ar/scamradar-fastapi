from PIL import Image
import io
import logging

try:
    from pyzbar.pyzbar import decode
    _HAS_ZBAR = True
except ImportError as e:
    logging.warning(f"pyzbar not fully installed (missing system library): {e}")
    _HAS_ZBAR = False


def decode_qr(image_bytes: bytes) -> str | None:
    if not _HAS_ZBAR:
        return None
        
    try:
        img = Image.open(io.BytesIO(image_bytes))
        results = decode(img)
        for r in results:
            data = r.data.decode("utf-8", errors="ignore")
            if data.startswith("http"):
                return data
        return None
    except Exception:
        return None
