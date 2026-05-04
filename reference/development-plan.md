---
title: ScamRadar — Development Plan
type: project
tags: [cybersecurity, AI, NLP, LINE, hackathon, GRU, PyThaiNLP, FastAPI]
created: 2026-05-04
updated: 2026-05-04
---

# ScamRadar — Development Plan

> ดู overview ของโปรเจคได้ที่ [[projects/scamradar/scamradar]]
> **Hack Day:** 30 พ.ค. – 1 มิ.ย. 2569 (44 ชั่วโมง)

---

## 0. Architecture Overview

```
INPUT (ผ่าน LINE)
  ├── text (copy-paste SMS)   ──────────────────────────┐
  └── image (screenshot SMS)  → [typhoon-ocr] → text ──→ [BiGRU Classifier]
                                                              │ scam_prob (0–1)
                               URL ──→ [VirusTotal API] ──→  │ threat_score
                               QR  ──→ [pyzbar decode]  ──→  │ decoded_url
                                                         ┌────▼─────────────┐
                                                         │ Ensemble Scorer  │
                                                         │ weighted sum      │
                                                         └────┬─────────────┘
                                                              │ {score, evidence}
                                                    ┌─────────▼──────────────────┐
                                                    │ Typhoon v2.1-12b-instruct  │
                                                    │ รับ structured evidence    │
                                                    │ → อธิบายเหตุผลภาษาไทย     │
                                                    └─────────┬──────────────────┘
                                                              │
                                                    LINE reply → ผู้ใช้
```

### Role ของแต่ละ Component

| Component          | Role                     | ทำไมเลือกนี้                                          |
| ------------------ | ------------------------ | ----------------------------------------------------- |
| **BiGRU**          | Thai SMS scam classifier | เร็ว (<50ms), RAM < 5MB, self-host ได้, real ML model |
| **typhoon-ocr**    | แปลง screenshot → text   | รองรับไทย+อังกฤษ, ฟรี                                 |
| **VirusTotal API** | URL reputation           | 90+ AV engines, ground truth, ไม่ใช่ LLM              |
| **pyzbar**         | QR decode → URL          | lightweight, ไม่ต้อง AP                               |
| **Typhoon v2.1**   | อธิบายเหตุผลภาษาไทย      | รับ structured evidence เท่านั้น, ไม่ classify เอง    |

---

## 1. Phase 1 — BiGRU Classifier (ก่อน Hack Day)

### 1.1 Dataset

| ไฟล์ | Path |
|------|------|
| Training data | `raw/Fraud-SMS-Detection-Application-main/Model/Thai-language-model/Dataset/Final-Data/final_data_th.csv` |
| Format | CSV: `text` (comma-sep tokens), `label` (0=ham, 1=fraud) |
| Size | 1,330 samples (665 fraud / 665 ham) — balanced |
| Token stats | min=3, max=101, avg=19.8 tokens/message |
| Vocab size | 2,345 unique tokens |

> **หมายเหตุ:** text ใน dataset tokenized ด้วย PyThaiNLP แล้ว (comma-separated) — ใช้ตรงๆ ได้เลย ไม่ต้อง re-tokenize

### 1.2 GRU Architecture

```python
# Hyperparameters
VOCAB_SIZE    = 2345 + 2   # +2 for <PAD>=0, <UNK>=1
EMBED_DIM     = 64
GRU_UNITS     = 64
MAX_LEN       = 64          # cover 99% of messages (max=101, avg=19.8)
DROPOUT       = 0.3
BATCH_SIZE    = 32
EPOCHS        = 30
LR            = 1e-3

# Model
Input(MAX_LEN)
  → Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)
  → Bidirectional(GRU(GRU_UNITS, return_sequences=False))
  → Dropout(0.3)
  → Dense(32, activation='relu')
  → Dropout(0.2)
  → Dense(1, activation='sigmoid')   # output: scam_prob
```

### 1.3 Training Script

สร้างไว้ที่ `scamradar/ml/train_gru.py`

```python
# สิ่งที่ script ต้องทำ:
# 1. โหลด final_data_th.csv
# 2. สร้าง vocab จาก text column (token → index)
# 3. encode + pad sequences → (N, MAX_LEN)
# 4. split 80/10/10 (train/val/test) stratified
# 5. train BiGRU model
# 6. evaluate บน test set
# 7. save:
#    - scamradar/ml/model.keras
#    - scamradar/ml/vocab.json  (token → index mapping)
#    - scamradar/ml/config.json (MAX_LEN, threshold)
```

### 1.4 Expected Output

| Metric | Target | Beebuzz baseline |
|--------|--------|-----------------|
| Val Accuracy | ≥ 94% | 95–97% (LSTM/GRU) |
| Val F1 | ≥ 0.93 | — |
| Inference latency | < 50ms (CPU) | — |
| Model size | < 5MB | — |

### 1.5 Threshold Tuning

หลัง train เสร็จ — plot precision-recall curve แล้วเลือก threshold:
- Default: 0.5
- ถ้า recall สำคัญกว่า (missed scam แย่กว่า false alarm) → ลด threshold ลง ~0.35–0.40
- บันทึก threshold ที่เลือกลง `config.json`

### 1.6 Inference Function

```python
def predict_scam(text: str, model, vocab, config) -> dict:
    """
    Input:  raw Thai text (ยังไม่ tokenize)
    Output: {"scam_prob": float, "label": "fraud"|"ham", "confidence": "high"|"medium"|"low"}
    """
    # 1. tokenize ด้วย PyThaiNLP (newmm engine)
    tokens = word_tokenize(text, engine='newmm')
    # 2. encode (UNK=1 ถ้าไม่อยู่ใน vocab)
    encoded = [vocab.get(t, 1) for t in tokens]
    # 3. pad/truncate → MAX_LEN
    padded = pad_sequences([encoded], maxlen=config['MAX_LEN'], padding='post')
    # 4. predict
    prob = float(model.predict(padded)[0][0])
    label = "fraud" if prob >= config['threshold'] else "ham"
    confidence = "high" if abs(prob - 0.5) > 0.3 else "medium" if abs(prob - 0.5) > 0.15 else "low"
    return {"scam_prob": prob, "label": label, "confidence": confidence}
```

---

## 2. Phase 2 — Backend (FastAPI)

### 2.0 LINE SDK v3 — Imports & Patterns

> ⚠️ **ใช้ `linebot.v3` เท่านั้น** — v2 deprecated, import path เปลี่ยนทั้งหมด

```python
# ✅ v3 (ถูกต้อง)
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient,
    MessagingApi,       # reply, push text
    MessagingApiBlob,   # ดาวน์โหลด image bytes ← แยก class จาก MessagingApi
    ReplyMessageRequest, PushMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent,   # ตรวจ event.message type
)

# ❌ v2 (เก่า — ห้ามใช้)
# from linebot import LineBotApi, WebhookHandler
```

**FastAPI webhook endpoint:**
```python
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

@app.post("/callback")
async def callback(request: Request):
    signature = request.headers["X-Line-Signature"]
    body = (await request.body()).decode()
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400)
    return "OK"

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text(event):
    # process text...

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event):
    # ดาวน์โหลด image ด้วย MessagingApiBlob
    with ApiClient(configuration) as api_client:
        blob_api = MessagingApiBlob(api_client)
        image_bytes = blob_api.get_message_content(event.message.id)
        # → bytearray → ส่งให้ typhoon-ocr
```

**Reply + Push:**
```python
# Reply (ตอบ event ที่รับมา — มี reply_token)
with ApiClient(configuration) as api_client:
    MessagingApi(api_client).reply_message_with_http_info(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text="กำลังวิเคราะห์...")]
        )
    )

# Push (ส่งหลัง async process เสร็จ — ใช้ user_id)
with ApiClient(configuration) as api_client:
    MessagingApi(api_client).push_message(
        PushMessageRequest(
            to=event.source.user_id,
            messages=[TextMessage(text=result_text)]
        )
    )
```

> **Pattern สำคัญ:** Reply "กำลังวิเคราะห์..." ก่อน (ใช้ reply_token) → process async → Push ผลจริง (ใช้ user_id) — ป้องกัน LINE webhook timeout 30s

---

### 2.1 Project Structure

```
scamradar/
├── ml/
│   ├── train_gru.py          ← training script
│   ├── model.keras           ← trained model
│   ├── vocab.json            ← token → index
│   └── config.json           ← MAX_LEN, threshold
├── app/
│   ├── main.py               ← FastAPI entry point + startup model load
│   ├── line_handler.py       ← LINE webhook handler (linebot.v3)
│   ├── router.py             ← message type detection
│   ├── pipelines/
│   │   ├── nlp.py            ← BiGRU classify
│   │   ├── ocr.py            ← typhoon-ocr call
│   │   ├── url.py            ← VirusTotal + domain analysis
│   │   └── qr.py             ← pyzbar decode
│   ├── scorer.py             ← ensemble risk score
│   ├── explainer.py          ← Typhoon v2.1 structured prompt
│   └── formatter.py          ← LINE message formatting
├── requirements.txt
├── Dockerfile
└── railway.toml
```

### 2.2 Message Router Logic

```python
# ตรวจ type จาก event.message class (v3 pattern)
from linebot.v3.webhooks import TextMessageContent, ImageMessageContent

def detect_input_type(event) -> str:
    if isinstance(event.message, ImageMessageContent):
        return "image"          # → blob_api.get_message_content() → OCR
    if isinstance(event.message, TextMessageContent):
        text = event.message.text
        if re.match(r'https?://', text.strip()):
            return "url"
        if re.search(r'https?://', text):
            return "text_with_url"   # → NLP + URL pipeline
        return "text"               # → NLP pipeline only
    return "unsupported"
```

### 2.3 Ensemble Scorer

```python
def compute_risk_score(nlp_result=None, url_result=None) -> dict:
    """
    Weighted ensemble ของ signals ทั้งหมด
    """
    score = 0.0
    evidence = []

    if nlp_result:
        # weight: 0.5 ถ้ามี URL result ด้วย, 1.0 ถ้า text only
        w = 0.5 if url_result else 1.0
        score += nlp_result['scam_prob'] * w
        if nlp_result['scam_prob'] > 0.7:
            evidence.append(f"NLP: ข้อความมีลักษณะ SMS scam (confidence {nlp_result['confidence']})")

    if url_result:
        vt_ratio = url_result['malicious'] / url_result['total_engines']  # e.g. 12/90
        score += vt_ratio * 0.5
        if url_result['malicious'] > 0:
            evidence.append(f"URL: VirusTotal ตรวจพบ {url_result['malicious']}/{url_result['total_engines']} engines")
        if url_result['domain_age_days'] < 30:
            score += 0.1
            evidence.append(f"Domain: อายุแค่ {url_result['domain_age_days']} วัน (น่าสงสัย)")

    score = min(score, 1.0)
    risk_level = "สูง" if score >= 0.7 else "ปานกลาง" if score >= 0.4 else "ต่ำ"
    return {"score": score, "risk_level": risk_level, "evidence": evidence}
```

### 2.4 Typhoon v2.1 Explainer Prompt

```python
EXPLAIN_PROMPT = """คุณเป็นผู้เชี่ยวชาญด้านความปลอดภัยไซเบอร์ของไทย

ข้อมูลจากระบบวิเคราะห์:
- คะแนนความเสี่ยง: {score:.0%} (ระดับ{risk_level})
- หลักฐานที่พบ: {evidence_list}
- ข้อความต้นฉบับ: "{original_text}"

เขียนคำอธิบายสั้นๆ ให้ผู้ใช้ทั่วไปเข้าใจได้ (ไม่เกิน 3 bullet points) และคำแนะนำ 1 บรรทัด
ตอบเป็นภาษาไทยเท่านั้น ห้ามใช้ศัพท์เทคนิคที่ยาก"""

# หมายเหตุ: Typhoon รับ structured evidence ไม่ใช่ข้อความดิบ
# → ไม่ใช่ LLM wrapper
```

### 2.5 LINE Response Format

```
🔴 ความเสี่ยงสูง (87%)

⚠️ ที่พบ:
• ข้อความแอบอ้างเป็นธนาคารกสิกรไทย
• ลิงก์ไปยัง kbank-th.xyz (ไม่ใช่ kbank.co.th)
• VirusTotal ตรวจพบ 12/90 engines

📋 คำแนะนำ: อย่ากดลิงก์ ห้ามกรอก OTP
รายงาน → ETDA 1212 / โทร 1441
```

---

## 3. Phase 3 — OCR Pipeline (typhoon-ocr)

### 3.1 API Call

```python
# Typhoon OCR 1.5 (2B)
# Rate limit: 2 req/s / 20 req/min
# Endpoint: https://api.opentyphoon.ai/v1/chat/completions

async def extract_text_from_image(image_bytes: bytes) -> str:
    base64_img = base64.b64encode(image_bytes).decode()
    payload = {
        "model": "typhoon-ocr-preview",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                {"type": "text", "text": "สกัดข้อความทั้งหมดในภาพ ตอบเฉพาะข้อความที่เห็นในภาพเท่านั้น"}
            ]
        }],
        "max_tokens": 500
    }
    # → extracted text → ส่งต่อให้ NLP pipeline
```

### 3.2 Rate Limit Mitigation

- Cache OCR result ด้วย SHA256 hash ของ image bytes
- ถ้า hash ซ้ำ → ใช้ cached result ทันที
- Queue ถ้า request เข้าพร้อมกัน (asyncio.Semaphore)

---

## 4. Phase 4 — URL Pipeline (VirusTotal)

### 4.1 VirusTotal API v3

```python
# Free tier: 4 req/min
# Cache สำคัญมาก

async def check_url(url: str) -> dict:
    # 1. encode URL เป็น base64 (VirusTotal requirement)
    url_id = base64.urlsafe_b64encode(url.encode()).decode().rstrip('=')
    # 2. GET /urls/{url_id}
    # 3. parse: malicious, suspicious, harmless, undetected counts
    # 4. GET /domains/{domain} → creation_date → domain_age_days
    return {
        "malicious": int,
        "total_engines": int,
        "domain_age_days": int,
        "final_url": str   # หลัง redirect
    }
```

### 4.2 Domain Pattern Rules (no API needed)

```python
THAI_BRAND_DOMAINS = {
    "กสิกรไทย": ["kbank.co.th"],
    "ไทยพาณิชย์": ["scb.co.th"],
    "กรุงไทย": ["ktb.co.th"],
    "กรมสรรพากร": ["rd.go.th"],
    "ตำรวจ": ["royalthaipolice.go.th"],
}

def check_domain_impersonation(url: str) -> bool:
    """ตรวจว่า URL แอบอ้างเป็น brand ไทยไหม"""
    domain = extract_domain(url)
    for brand, legit_domains in THAI_BRAND_DOMAINS.items():
        if brand_keyword_in_domain(domain) and domain not in legit_domains:
            return True, brand
    return False, None
```

---

## 5. Phase 5 — Deployment (Railway)

### 5.1 Requirements

```txt
# requirements.txt
fastapi==0.111.0
uvicorn[standard]==0.30.0
line-bot-sdk==3.11.0
tensorflow==2.16.1       # หรือ keras-only ถ้าต้องการ lighter
pythainlp==5.0.4
httpx==0.27.0
pyzbar==0.1.9
Pillow==10.3.0
python-dotenv==1.0.1
```

### 5.2 Environment Variables

```env
LINE_CHANNEL_SECRET=xxx
LINE_CHANNEL_ACCESS_TOKEN=xxx
VIRUSTOTAL_API_KEY=xxx
TYPHOON_API_KEY=xxx         # สำหรับ OCR + explainer
```

### 5.3 Startup

```python
# main.py — โหลด model ตอน startup (1 ครั้ง)
@app.on_event("startup")
async def load_model():
    app.state.model = keras.models.load_model("ml/model.keras")
    app.state.vocab  = json.load(open("ml/vocab.json"))
    app.state.config = json.load(open("ml/config.json"))
    # → inference ไม่มี cold start
```

### 5.4 Response Time Target

| Step | Target latency |
|------|---------------|
| LINE webhook receive | 0ms |
| Message routing | <5ms |
| BiGRU inference (CPU) | <50ms |
| VirusTotal API | ~300ms (cached: 0ms) |
| typhoon-ocr (image) | ~2s |
| Typhoon v2.1 explain | ~1s |
| **Total (text only)** | **< 500ms** |
| **Total (image+URL)** | **< 4s** |

LINE webhook timeout = 30s → ปลอดภัย

---

## 6. Hack Day Timeline (44 ชั่วโมง)

### Hour 0–4: Model Training ✅ (ทำก่อน Hack Day ได้เลย)
- [ ] เขียน `ml/train_gru.py`
- [ ] Train + evaluate บน Beebuzz dataset
- [ ] Save `model.keras` + `vocab.json` + `config.json`
- [ ] ทดสอบ inference function

### Hour 0–8: Foundation
- [ ] Setup LINE Official Account + Messaging API
- [ ] FastAPI skeleton + Railway deploy + HTTPS verify
- [ ] LINE webhook basic (receive + echo)
- [ ] Load BiGRU at startup → `/health` endpoint

### Hour 8–16: Core Pipelines
- [ ] NLP pipeline (`pipelines/nlp.py`) — text → BiGRU → scam_prob
- [ ] URL pipeline (`pipelines/url.py`) — VirusTotal + domain impersonation
- [ ] Message Router — auto-detect input type
- [ ] Ensemble Scorer (`scorer.py`)

### Hour 16–24: OCR + QR + Explainer
- [ ] OCR pipeline (`pipelines/ocr.py`) — image → typhoon-ocr → text → NLP
- [ ] QR pipeline (`pipelines/qr.py`) — pyzbar → URL → URL pipeline
- [ ] Typhoon explainer (`explainer.py`) — structured prompt → Thai explanation
- [ ] Response formatter — emoji + readable format

### Hour 24–32: Reliability
- [ ] Caching layer — OCR hash cache, VirusTotal URL cache
- [ ] Rate limit guards (asyncio.Semaphore)
- [ ] Async "กำลังวิเคราะห์..." reply ก่อน (ป้องกัน timeout)
- [ ] Error handling ทุก pipeline + fallback messages

### Hour 32–40: Demo Prep
- [ ] ทดสอบ 4 formats ครบ (text / image / URL / QR)
- [ ] เตรียม test cases จริง (screenshot scam จริง)
- [ ] สร้าง demo script สำหรับ pitch
- [ ] Slide ประกอบ

### Hour 40–44: Buffer
- [ ] Bug fixes
- [ ] Final deploy check
- [ ] Rehearse pitch

---

## 7. ความเสี่ยงและ Mitigation

| ความเสี่ยง | โอกาส | Impact | Mitigation |
|-----------|-------|--------|-----------|
| BiGRU accuracy ต่ำกว่าที่คาด | ต่ำ | สูง | Beebuzz baseline GRU ~95% — ถ้าต่ำให้ tune threshold |
| typhoon-ocr rate limit (20/min) | ปานกลาง | กลาง | Image hash cache + asyncio.Semaphore(1) |
| VirusTotal rate limit (4/min) | สูง | กลาง | URL cache + domain pattern rules เป็น fallback |
| Railway memory เกิน | ต่ำ | สูง | BiGRU < 5MB — Railway 512MB พอสบาย (TF อาจใหญ่ → ใช้ tflite) |
| LINE webhook timeout | ต่ำ | สูง | Reply "กำลังวิเคราะห์..." ก่อน → process async |
| TensorFlow บน Railway ใหญ่เกิน | ปานกลาง | กลาง | ใช้ `tflite-runtime` แทน TF full (< 10MB) |

---

## 8. การทดสอบ (Test Cases)

### Text Scam (ตัวอย่างจริง)
```
"ยินดีด้วย! บัญชีกสิกรของคุณได้รับโบนัส 5,000 บาท กดรับที่นี่: http://kbank-bonus.xyz/claim"
```
Expected: 🔴 สูง

### Ham (SMS ปกติ)
```
"สินค้าของคุณได้รับการจัดส่งแล้ว คาดว่าถึงภายใน 2-3 วันทำการ ติดตามพัสดุ: TH123456789"
```
Expected: 🟢 ต่ำ

### URL Phishing
```
http://scb-reward.xyz/login
```
Expected: 🔴 สูง (domain impersonation + VirusTotal)

### Screenshot SMS
```
[รูปภาพ SMS: "ด่วน! รหัส OTP ของคุณคือ 123456 ห้ามบอกใคร หากไม่ใช่คุณโทร 02-xxx-xxxx"]
```
Expected: OCR → extract → 🟡 ปานกลาง–🔴 สูง

---

## 9. สิ่งที่ต้องทำก่อน Hack Day

| Task | Deadline | Status |
|------|----------|--------|
| Train BiGRU + save artifacts | ก่อน 30 พ.ค. | ⬜ TODO |
| สมัคร LINE Developer account | ก่อน 30 พ.ค. | ⬜ TODO |
| สมัคร VirusTotal API key | ก่อน 30 พ.ค. | ⬜ TODO |
| สมัคร Typhoon API key | ก่อน 30 พ.ค. | ⬜ TODO |
| สมัคร Railway account | ก่อน 30 พ.ค. | ⬜ TODO |
| เก็บ test cases จริง (screenshot scam) | ก่อน 30 พ.ค. | ⬜ TODO |

---

## Links

- [[projects/scamradar/scamradar]] — overview
- [[competitions/leagues-of-code-hackathon-2]] — competition details
- [[research/ETDA-Annual-Report-2025]] — สถิติ scam ไทย
