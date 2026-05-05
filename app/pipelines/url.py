import base64
import hashlib
import os
import time
from urllib.parse import urlparse
import httpx

_VT_BASE = "https://www.virustotal.com/api/v3"
_HEADERS = lambda: {"x-apikey": os.environ.get("VIRUSTOTAL_API_KEY", "")}

_LEGIT_DOMAINS = {
    "kbank": ("กสิกรไทย", "kbank.co.th"),
    "scb": ("ไทยพาณิชย์", "scb.co.th"),
    "ktb": ("กรุงไทย", "ktb.co.th"),
    "rd": ("กรมสรรพากร", "rd.go.th"),
    "police": ("ตำรวจ", "royalthaipolice.go.th"),
}

_cache: dict[str, dict] = {}  # url_hash → result


def extract_domain(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    return domain.removeprefix("www.")


_SUSPICIOUS_TLDS = [".bet", ".vip", ".top", ".cc", ".win", ".asia", ".xyz", ".club", ".me"]

def check_domain_impersonation(url: str) -> tuple[str | None, bool]:
    domain = extract_domain(url)
    for keyword, (brand, legit) in _LEGIT_DOMAINS.items():
        if keyword in domain and domain != legit:
            return brand, True
    return None, False

def check_suspicious_tld(url: str) -> bool:
    domain = extract_domain(url)
    return any(domain.endswith(tld) for tld in _SUSPICIOUS_TLDS)

async def check_url(url: str) -> dict:
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cached = _cache.get(url_hash)
    if cached and time.time() - cached["_ts"] < 3600:
        return {k: v for k, v in cached.items() if k != "_ts"}

    # Follow redirects to catch short URLs (e.g., bit.ly -> line.me)
    final_url = url
    is_line_oa = False
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            resp = await client.head(url)
            final_url = str(resp.url)
            # Detect Line OA: line.me/R/ti/p/@... or line.me/ti/p/@...
            if "line.me" in final_url and ("/ti/p/" in final_url or "/R/ti/p/" in final_url):
                is_line_oa = True
    except Exception:
        pass

    brand, impersonated = check_domain_impersonation(final_url)
    is_suspicious_tld = check_suspicious_tld(final_url)
    vt_result = await _virustotal_check(final_url)
    
    result = {
        "malicious": vt_result.get("malicious", 0),
        "total_engines": vt_result.get("total_engines", 0),
        "domain_age_days": vt_result.get("domain_age_days", 999),
        "impersonation": brand if impersonated else None,
        "is_line_oa": is_line_oa,
        "is_suspicious_tld": is_suspicious_tld,
        "final_url": final_url if final_url != url else None
    }
    _cache[url_hash] = {**result, "_ts": time.time()}
    return result


async def _virustotal_check(url: str) -> dict:
    api_key = os.environ.get("VIRUSTOTAL_API_KEY")
    if not api_key:
        return {}

    url_id = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{_VT_BASE}/urls/{url_id}", headers=_HEADERS())
            if r.status_code != 200:
                # If not found, we could submit it, but for a hackathon let's just return empty
                return {}
            stats = r.json()["data"]["attributes"]["last_analysis_stats"]
            total = sum(stats.values())
            domain = extract_domain(url)
            age_days = await _get_domain_age(client, domain)
            return {"malicious": stats.get("malicious", 0), "total_engines": total, "domain_age_days": age_days}
    except Exception:
        return {}


async def _get_domain_age(client: httpx.AsyncClient, domain: str) -> int:
    try:
        r = await client.get(f"{_VT_BASE}/domains/{domain}", headers=_HEADERS())
        if r.status_code != 200:
            return 999
        created = r.json()["data"]["attributes"].get("creation_date", 0)
        age_days = int((time.time() - created) / 86400)
        return max(age_days, 0)
    except Exception:
        return 999
