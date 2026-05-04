# ScamRadar — Design Spec

**Date:** 2026-05-04  
**Hackathon:** Leagues of Code AI & Cybersecurity Hackathon #2  
**Hack Day:** 30 พ.ค. – 1 มิ.ย. 2569 (44h)  
**Build Approach:** Phased — ML core first, LINE bot second, APIs third

---

## 1. Project Structure

```
scamradar/
├── ml/
│   ├── model.tflite          ← thgru_grid.tflite (pre-trained, no retraining needed)
│   ├── vocab.txt             ← thai_text_classification_vocab.txt
│   └── classifier.py         ← inference wrapper (PyThaiNLP → tflite → scam_prob)
├── app/
│   ├── main.py               ← FastAPI entry + startup model load
│   ├── line_handler.py       ← LINE webhook (linebot.v3)
│   ├── router.py             ← detect input type
│   ├── pipelines/
│   │   ├── nlp.py            ← uses ml/classifier.py
│   │   ├── url.py            ← VirusTotal + domain impersonation rules
│   │   ├── ocr.py            ← typhoon-ocr API
│   │   └── qr.py             ← pyzbar decode
│   ├── scorer.py             ← weighted ensemble
│   ├── explainer.py          ← Typhoon v2.1 structured prompt
│   └── formatter.py          ← LINE message formatting
├── requirements.txt
├── Dockerfile
└── railway.toml
```

`ml/classifier.py` is a standalone module — no FastAPI dependency, independently testable, loaded once at startup and passed as a dependency.

---

## 2. Data Flow

```
LINE User → webhook POST /callback
                │
         [line_handler.py]
                │
    reply "กำลังวิเคราะห์..." ← reply_token (prevents 30s timeout)
                │
         [router.py] detect type
          ├── text only      → nlp.py
          ├── text + URL     → nlp.py + url.py (parallel)
          ├── URL only       → url.py
          ├── image          → ocr.py → text → nlp.py + url.py
          └── QR (image)     → qr.py → URL → url.py
                │
         [scorer.py] weighted ensemble
          • text only:  score = nlp_prob
          • url:        score = nlp_prob×0.5 + vt_ratio×0.5
          • domain hit: score += 0.2 (impersonation bonus)
                │
         [explainer.py] Typhoon v2.1
          input: {score, evidence[]} → Thai explanation (3 bullets max)
          fallback: rule-based Thai message if Typhoon API fails
                │
         [formatter.py] → push_message with user_id
```

**Critical path:** OCR (~2s) + Typhoon (~1s) = ~3s total — well within LINE's 30s webhook timeout.

---

## 3. ML Classifier

**Model:** `thgru_grid.tflite` (GRU, trained on Beebuzz Thai SMS dataset)  
**Vocab:** `thai_text_classification_vocab.txt` (2,136 tokens)  
**Max length:** 109 tokens (from notebook: `max_len = 109`)  
**Special tokens:** `<PAD>=0, <START>=1, <UNKNOWN>=2, <UNUSED>=3`

```python
# ml/classifier.py

MAX_LEN = 109
PAD_IDX, START_IDX, UNK_IDX = 0, 1, 2

def load_vocab(path: str) -> dict:
    vocab = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().rsplit(' ', 1)
            if len(parts) == 2:
                vocab[parts[0]] = int(parts[1])
    return vocab

def predict(text: str, interpreter, vocab: dict) -> dict:
    tokens = word_tokenize(text, engine='newmm')          # PyThaiNLP
    encoded = [START_IDX] + [vocab.get(t, UNK_IDX) for t in tokens]
    encoded = encoded[:MAX_LEN]
    padded = encoded + [PAD_IDX] * (MAX_LEN - len(encoded))
    inp = np.array([padded], dtype=np.float32)
    interpreter.set_tensor(input_idx, inp)
    interpreter.invoke()
    prob = float(interpreter.get_tensor(output_idx)[0][0])
    return {
        "scam_prob": prob,
        "label": "fraud" if prob >= 0.5 else "ham",
        "confidence": "high" if abs(prob-0.5) > 0.3 else "medium" if abs(prob-0.5) > 0.15 else "low"
    }
```

**Index offset note:** `vocab.txt` indices already match `tokenizer.word_index` values (special tokens were prepended with correct offsets during training). No additional `+3` needed at inference time.

---

## 4. External APIs & Caching

### VirusTotal (`url.py`)
- Encode URL → base64 (urlsafe, strip `=`) → `GET /urls/{id}`
- Parse: `malicious`, `total = sum(stats.values())`
- `GET /domains/{domain}` → `creation_date` → `domain_age_days`
- Cache: `dict[url_hash → result]`, TTL = 1 hour
- Rate limit: 4 req/min free tier

### Thai Brand Impersonation Rules (no API required)
```python
LEGIT_DOMAINS = {
    "kbank": "kbank.co.th", "scb": "scb.co.th",
    "ktb":   "ktb.co.th",   "rd":  "rd.go.th",
    "police": "royalthaipolice.go.th",
}
# brand keyword in domain but not matching legit → score += 0.2
```

### typhoon-ocr (`ocr.py`)
- `POST` image as base64 → `typhoon-ocr-preview` model
- Prompt: `"สกัดข้อความทั้งหมดในภาพ ตอบเฉพาะข้อความที่เห็นในภาพเท่านั้น"`
- Cache: `dict[sha256(image_bytes) → extracted_text]`
- Rate limit guard: `asyncio.Semaphore(1)` (2 req/s limit)

### Typhoon v2.1 Explainer (`explainer.py`)
- Input: `{score, risk_level, evidence[], original_text}`
- Output: 3-bullet Thai explanation for general users
- Fallback: rule-based Thai message derived from `risk_level` if API fails or times out

---

## 5. Deployment

**Platform:** Railway (free tier, HTTPS included)

```toml
# railway.toml
[build]
builder = "dockerfile"

[deploy]
startCommand = "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
```

**requirements.txt**
```
fastapi==0.111.0
uvicorn[standard]==0.30.0
line-bot-sdk==3.11.0
tflite-runtime==2.14.0
pythainlp==5.0.4
httpx==0.27.0
pyzbar==0.1.9
Pillow==10.3.0
python-dotenv==1.0.1
numpy==1.26.4
```

`tflite-runtime` instead of full TensorFlow saves ~400MB on Railway image size.

**Environment variables**
```
LINE_CHANNEL_SECRET
LINE_CHANNEL_ACCESS_TOKEN
VIRUSTOTAL_API_KEY
TYPHOON_API_KEY
```

**Startup**
```python
@app.on_event("startup")
async def load_model():
    app.state.vocab       = load_vocab("ml/vocab.txt")
    app.state.interpreter = load_tflite("ml/model.tflite")
```

`/health` returns `{"status": "ok", "model": "loaded"}` for Railway health check.

---

## 6. Build Timeline (Phased — Approach A)

### Day 1 — ML Core
- Copy `thgru_grid.tflite` + `thai_text_classification_vocab.txt` → `ml/`
- Write `ml/classifier.py` + test inference with dev-plan test cases
- Verify accuracy on `reference/Model/.../Dataset/Test-Data/exported_test_data.csv`

### Day 2 — LINE Bot + NLP Pipeline
- Register LINE Developer + Railway accounts
- FastAPI skeleton + Railway deploy + HTTPS verify
- LINE webhook: receive + reply "กำลังวิเคราะห์..."
- Wire: text → `nlp.py` → `scorer.py` → `formatter.py` → push result
- **Demo checkpoint:** send Thai scam text → bot replies with risk score

### Day 3 — URL + Domain Rules
- Register VirusTotal API key
- `url.py` + domain impersonation rules + URL cache
- Router detects URL-in-text → runs NLP + URL pipelines in parallel
- **Demo checkpoint:** send URL → VirusTotal result

### Day 4 — OCR + QR
- Register Typhoon API key
- `ocr.py` + image hash cache + semaphore
- `qr.py` (pyzbar) → URL pipeline
- **Demo checkpoint:** send screenshot → OCR → classify

### Day 5+ — Explainer + Polish
- `explainer.py` Typhoon v2.1 + fallback
- Error handling for every pipeline
- Rate limit guards
- Prepare test cases for Hack Day demo

---

## 7. Test Cases

| Input | Expected |
|-------|---------|
| `"ยินดีด้วย! บัญชีกสิกรของคุณได้รับโบนัส 5,000 บาท กดรับที่นี่: http://kbank-bonus.xyz/claim"` | 🔴 สูง |
| `"สินค้าของคุณได้รับการจัดส่งแล้ว คาดว่าถึงภายใน 2-3 วันทำการ"` | 🟢 ต่ำ |
| `http://scb-reward.xyz/login` | 🔴 สูง (domain impersonation + VirusTotal) |
| Screenshot SMS scam | OCR → 🟡–🔴 |

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| tflite accuracy ต่ำกว่าคาด | ตรวจสอบบน exported_test_data.csv ก่อน Day 2 |
| VirusTotal 4 req/min | URL cache + domain rules เป็น fallback |
| typhoon-ocr 20 req/min | Image hash cache + asyncio.Semaphore(1) |
| Typhoon API ล้มเหลว | rule-based fallback message |
| LINE 30s webhook timeout | Reply "กำลังวิเคราะห์..." → process async → push result |
| Railway memory | tflite-runtime <10MB (vs TF full ~400MB) |
