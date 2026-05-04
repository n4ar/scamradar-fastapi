# ScamRadar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LINE bot that classifies Thai scam messages (text/image/URL/QR) using a pre-trained GRU tflite model and external APIs, deployed on Railway.

**Architecture:** Phased — ML classifier first (Day 1), LINE bot + NLP pipeline (Day 2), URL pipeline (Day 3), OCR + QR (Day 4), Typhoon explainer (Day 5). Each phase produces a working demo checkpoint. FastAPI backend on Railway, LINE SDK v3, tflite-runtime for model inference.

**Tech Stack:** Python 3.11, FastAPI, linebot.v3, tflite-runtime (+ tensorflow fallback), PyThaiNLP, httpx, pyzbar, Pillow, Railway

---

## File Map

| File | Responsibility |
|------|---------------|
| `ml/model.tflite` | Pre-trained GRU model (copied from reference) |
| `ml/vocab.txt` | Token→index mapping (copied from reference) |
| `ml/classifier.py` | Load vocab/model, tokenize raw Thai text, return scam_prob |
| `app/main.py` | FastAPI app, startup model load, `/health` |
| `app/line_handler.py` | LINE webhook receive, reply "กำลังวิเคราะห์...", dispatch async |
| `app/router.py` | Detect message type (text/url/image/qr) from LINE event |
| `app/pipelines/nlp.py` | Wrap `ml/classifier.py` for pipeline use |
| `app/pipelines/url.py` | VirusTotal API + domain impersonation rules |
| `app/pipelines/ocr.py` | typhoon-ocr API call + image hash cache |
| `app/pipelines/qr.py` | pyzbar QR decode → URL |
| `app/scorer.py` | Weighted ensemble of nlp + url signals → risk score |
| `app/explainer.py` | Typhoon v2.1 structured prompt + rule-based fallback |
| `app/formatter.py` | Build LINE text message from score + explanation |
| `tests/test_classifier.py` | Unit tests for ml/classifier.py |
| `tests/test_scorer.py` | Unit tests for scorer.py |
| `tests/test_url.py` | Unit tests for url.py domain rules (no API calls) |
| `tests/test_formatter.py` | Unit tests for formatter.py |
| `requirements.txt` | Dependencies |
| `Dockerfile` | Railway build |
| `railway.toml` | Railway deploy config |
| `.env.example` | Required env var template |

---

## Task 1: Project scaffold + ML artifacts

**Files:**
- Create: `ml/model.tflite` (copy)
- Create: `ml/vocab.txt` (copy)
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `tests/__init__.py`

- [ ] **Step 1: Copy ML artifacts**

```bash
mkdir -p ml tests
cp "reference/Model/Thai-language-model/GridSearch-model/thgru_grid.tflite" ml/model.tflite
cp "reference/Model/Thai-language-model/Dataset/Final-Data/thai_text_classification_vocab.txt" ml/vocab.txt
```

- [ ] **Step 2: Verify files exist and vocab format**

```bash
ls -lh ml/
head -6 ml/vocab.txt
```

Expected output:
```
<PAD> 0
<START> 1
<UNKNOWN> 2
<UNUSED> 3
เลข 4
ลิงก์ 5
```

- [ ] **Step 3: Create requirements.txt**

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
pytest==8.2.0
pytest-asyncio==0.23.7
```

- [ ] **Step 4: Create .env.example**

```
LINE_CHANNEL_SECRET=your_secret_here
LINE_CHANNEL_ACCESS_TOKEN=your_token_here
VIRUSTOTAL_API_KEY=your_key_here
TYPHOON_API_KEY=your_key_here
```

- [ ] **Step 5: Install deps**

```bash
pip install -r requirements.txt
```

Expected: installs without errors. If `tflite-runtime` fails on Mac, install `tensorflow==2.16.1` instead for local dev.

- [ ] **Step 6: Create tests/__init__.py**

```bash
touch tests/__init__.py
```

- [ ] **Step 7: Commit**

```bash
git init
git add ml/model.tflite ml/vocab.txt requirements.txt .env.example tests/__init__.py
git commit -m "chore: add ML artifacts and project scaffold"
```

---

## Task 2: ML Classifier

**Files:**
- Create: `ml/__init__.py`
- Create: `ml/classifier.py`
- Create: `tests/test_classifier.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_classifier.py`:

```python
import pytest
import numpy as np
from ml.classifier import load_vocab, load_interpreter, predict

VOCAB_PATH = "ml/vocab.txt"
MODEL_PATH = "ml/model.tflite"

@pytest.fixture(scope="module")
def classifier():
    vocab = load_vocab(VOCAB_PATH)
    interpreter = load_interpreter(MODEL_PATH)
    return vocab, interpreter

def test_vocab_loads(classifier):
    vocab, _ = classifier
    assert "<PAD>" in vocab
    assert vocab["<PAD>"] == 0
    assert "<START>" in vocab
    assert vocab["<START>"] == 1
    assert "<UNKNOWN>" in vocab
    assert vocab["<UNKNOWN>"] == 2
    assert len(vocab) > 100

def test_predict_returns_expected_keys(classifier):
    vocab, interpreter = classifier
    result = predict("ทดสอบ", interpreter, vocab)
    assert "scam_prob" in result
    assert "label" in result
    assert "confidence" in result
    assert 0.0 <= result["scam_prob"] <= 1.0
    assert result["label"] in ("fraud", "ham")
    assert result["confidence"] in ("high", "medium", "low")

def test_predict_scam_text(classifier):
    vocab, interpreter = classifier
    text = "ยินดีด้วย บัญชีกสิกรของคุณได้รับโบนัส กดรับที่ลิงก์"
    result = predict(text, interpreter, vocab)
    assert result["scam_prob"] > 0.5
    assert result["label"] == "fraud"

def test_predict_ham_text(classifier):
    vocab, interpreter = classifier
    text = "สินค้าของคุณจัดส่งแล้ว ติดตามพัสดุได้ที่ไปรษณีย์ไทย"
    result = predict(text, interpreter, vocab)
    assert result["scam_prob"] < 0.5
    assert result["label"] == "ham"

def test_predict_empty_text(classifier):
    vocab, interpreter = classifier
    result = predict("", interpreter, vocab)
    assert "scam_prob" in result  # should not crash
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_classifier.py -v
```

Expected: `ModuleNotFoundError: No module named 'ml.classifier'`

- [ ] **Step 3: Create ml/__init__.py**

```bash
touch ml/__init__.py
```

- [ ] **Step 4: Write ml/classifier.py**

```python
import numpy as np
from pythainlp.tokenize import word_tokenize

try:
    import tflite_runtime.interpreter as tflite
    _Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    _Interpreter = tf.lite.Interpreter

MAX_LEN = 109
PAD_IDX = 0
START_IDX = 1
UNK_IDX = 2


def load_vocab(path: str) -> dict:
    vocab = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                vocab[parts[0]] = int(parts[1])
    return vocab


def load_interpreter(path: str):
    interpreter = _Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


def predict(text: str, interpreter, vocab: dict) -> dict:
    tokens = word_tokenize(text, engine="newmm") if text.strip() else []
    encoded = [START_IDX] + [vocab.get(t, UNK_IDX) for t in tokens]
    encoded = encoded[:MAX_LEN]
    padded = encoded + [PAD_IDX] * (MAX_LEN - len(encoded))

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    inp = np.array([padded], dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    prob = float(interpreter.get_tensor(output_details[0]["index"])[0][0])

    label = "fraud" if prob >= 0.5 else "ham"
    gap = abs(prob - 0.5)
    confidence = "high" if gap > 0.3 else "medium" if gap > 0.15 else "low"
    return {"scam_prob": prob, "label": label, "confidence": confidence}
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_classifier.py -v
```

Expected: all 5 tests PASS. If `test_predict_scam_text` or `test_predict_ham_text` fails, the model may classify differently — adjust assertion threshold (e.g. `> 0.4`) rather than the logic.

- [ ] **Step 6: Commit**

```bash
git add ml/__init__.py ml/classifier.py tests/test_classifier.py
git commit -m "feat: ML classifier with tflite GRU inference"
```

---

## Task 3: FastAPI skeleton + health endpoint

**Files:**
- Create: `app/__init__.py`
- Create: `app/main.py`

- [ ] **Step 1: Create app/__init__.py**

```bash
mkdir -p app/pipelines
touch app/__init__.py app/pipelines/__init__.py
```

- [ ] **Step 2: Write app/main.py**

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from ml.classifier import load_vocab, load_interpreter

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vocab = load_vocab("ml/vocab.txt")
    app.state.interpreter = load_interpreter("ml/model.tflite")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok", "model": "loaded"}
```

- [ ] **Step 3: Run locally and verify**

```bash
uvicorn app.main:app --reload --port 8000
```

In a second terminal:
```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok","model":"loaded"}`

Stop the server (Ctrl+C).

- [ ] **Step 4: Commit**

```bash
git add app/__init__.py app/pipelines/__init__.py app/main.py
git commit -m "feat: FastAPI app with health endpoint and model startup load"
```

---

## Task 4: Deployment config (Railway)

**Files:**
- Create: `Dockerfile`
- Create: `railway.toml`

- [ ] **Step 1: Write Dockerfile**

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y libzbar0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write railway.toml**

```toml
[build]
builder = "dockerfile"

[deploy]
startCommand = "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 30
```

- [ ] **Step 3: Register accounts (manual — do before Day 2)**

- Go to https://developers.line.biz → create Provider → create Messaging API channel
- Note: `Channel Secret` and `Channel Access Token`
- Go to https://railway.app → create project → link GitHub repo
- Copy `.env.example` → `.env` and fill in LINE credentials

- [ ] **Step 4: Commit**

```bash
git add Dockerfile railway.toml
git commit -m "chore: Railway deployment config"
```

---

## Task 5: LINE webhook + echo (Day 2 start)

**Files:**
- Create: `app/line_handler.py`
- Modify: `app/main.py`

- [ ] **Step 1: Write app/line_handler.py**

```python
import os
import asyncio
from fastapi import Request, HTTPException
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent

_config = Configuration(access_token=os.environ["LINE_CHANNEL_ACCESS_TOKEN"])
_handler = WebhookHandler(os.environ["LINE_CHANNEL_SECRET"])


async def handle_webhook(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = (await request.body()).decode()
    try:
        _handler.handle(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return "OK"


def _reply(reply_token: str, text: str):
    with ApiClient(_config) as client:
        MessagingApi(client).reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[TextMessage(text=text)],
            )
        )


@_handler.add(MessageEvent, message=TextMessageContent)
def on_text(event: MessageEvent):
    _reply(event.reply_token, f"Echo: {event.message.text}")


@_handler.add(MessageEvent, message=ImageMessageContent)
def on_image(event: MessageEvent):
    _reply(event.reply_token, "ได้รับรูปภาพแล้ว (ยังไม่ได้ implement)")
```

- [ ] **Step 2: Register webhook in main.py**

Replace `app/main.py` content:

```python
import os
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from ml.classifier import load_vocab, load_interpreter
from app.line_handler import handle_webhook

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vocab = load_vocab("ml/vocab.txt")
    app.state.interpreter = load_interpreter("ml/model.tflite")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok", "model": "loaded"}

@app.post("/callback")
async def callback(request: Request):
    return await handle_webhook(request)
```

- [ ] **Step 3: Test locally with ngrok**

```bash
# terminal 1
uvicorn app.main:app --reload --port 8000

# terminal 2
ngrok http 8000
```

Set ngrok HTTPS URL + `/callback` as webhook URL in LINE Developer Console.
Send any message to the LINE bot → expect echo reply.

- [ ] **Step 4: Commit**

```bash
git add app/line_handler.py app/main.py
git commit -m "feat: LINE webhook with echo handler"
```

---

## Task 6: Scorer + Formatter

**Files:**
- Create: `app/scorer.py`
- Create: `app/formatter.py`
- Create: `tests/test_scorer.py`
- Create: `tests/test_formatter.py`

- [ ] **Step 1: Write failing tests for scorer**

Create `tests/test_scorer.py`:

```python
from app.scorer import compute_risk_score

def test_text_only_high_risk():
    result = compute_risk_score(nlp_result={"scam_prob": 0.9, "confidence": "high"})
    assert result["score"] == pytest.approx(0.9)
    assert result["risk_level"] == "สูง"
    assert any("NLP" in e for e in result["evidence"])

def test_text_only_low_risk():
    result = compute_risk_score(nlp_result={"scam_prob": 0.1, "confidence": "high"})
    assert result["score"] == pytest.approx(0.1)
    assert result["risk_level"] == "ต่ำ"

def test_url_and_text():
    result = compute_risk_score(
        nlp_result={"scam_prob": 0.8, "confidence": "high"},
        url_result={"malicious": 10, "total_engines": 90, "domain_age_days": 5, "impersonation": None},
    )
    # score = 0.8*0.5 + (10/90)*0.5 + 0.1 (age<30) = 0.4 + 0.056 + 0.1 = 0.556
    assert result["score"] > 0.5
    assert result["risk_level"] == "ปานกลาง"

def test_domain_impersonation_bonus():
    result = compute_risk_score(
        url_result={"malicious": 0, "total_engines": 90, "domain_age_days": 100, "impersonation": "กสิกรไทย"},
    )
    assert result["score"] >= 0.2
    assert any("แอบอ้าง" in e for e in result["evidence"])

def test_score_capped_at_1():
    result = compute_risk_score(
        nlp_result={"scam_prob": 1.0, "confidence": "high"},
        url_result={"malicious": 90, "total_engines": 90, "domain_age_days": 1, "impersonation": "ธนาคาร"},
    )
    assert result["score"] <= 1.0

import pytest
```

- [ ] **Step 2: Write failing tests for formatter**

Create `tests/test_formatter.py`:

```python
from app.formatter import format_response

def test_high_risk_format():
    msg = format_response(
        score=0.87,
        risk_level="สูง",
        explanation="• ข้อความแอบอ้างธนาคาร\n• ลิงก์ไปยัง domain ปลอม",
        advice="อย่ากดลิงก์ รายงาน ETDA 1212",
    )
    assert "🔴" in msg
    assert "87%" in msg
    assert "อย่ากดลิงก์" in msg

def test_medium_risk_format():
    msg = format_response(score=0.55, risk_level="ปานกลาง", explanation="• น่าสงสัย", advice="ระวัง")
    assert "🟡" in msg
    assert "55%" in msg

def test_low_risk_format():
    msg = format_response(score=0.1, risk_level="ต่ำ", explanation="• ปกติ", advice="ปลอดภัย")
    assert "🟢" in msg
    assert "10%" in msg
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_scorer.py tests/test_formatter.py -v
```

Expected: `ImportError`

- [ ] **Step 4: Write app/scorer.py**

```python
def compute_risk_score(nlp_result: dict = None, url_result: dict = None) -> dict:
    score = 0.0
    evidence = []
    has_url = url_result is not None

    if nlp_result:
        w = 0.5 if has_url else 1.0
        score += nlp_result["scam_prob"] * w
        if nlp_result["scam_prob"] > 0.6:
            evidence.append(f"NLP: ข้อความมีลักษณะ SMS scam (confidence: {nlp_result['confidence']})")

    if url_result:
        total = url_result["total_engines"] or 1
        vt_ratio = url_result["malicious"] / total
        score += vt_ratio * 0.5
        if url_result["malicious"] > 0:
            evidence.append(f"URL: VirusTotal พบ {url_result['malicious']}/{url_result['total_engines']} engines")
        if url_result["domain_age_days"] < 30:
            score += 0.1
            evidence.append(f"Domain: อายุเพียง {url_result['domain_age_days']} วัน (น่าสงสัย)")
        if url_result.get("impersonation"):
            score += 0.2
            evidence.append(f"Domain: แอบอ้างเป็น {url_result['impersonation']}")

    score = min(score, 1.0)
    risk_level = "สูง" if score >= 0.7 else "ปานกลาง" if score >= 0.4 else "ต่ำ"
    return {"score": score, "risk_level": risk_level, "evidence": evidence}
```

- [ ] **Step 5: Write app/formatter.py**

```python
_ICONS = {"สูง": "🔴", "ปานกลาง": "🟡", "ต่ำ": "🟢"}

def format_response(score: float, risk_level: str, explanation: str, advice: str) -> str:
    icon = _ICONS.get(risk_level, "⚪")
    pct = int(round(score * 100))
    return (
        f"{icon} ความเสี่ยง{risk_level} ({pct}%)\n\n"
        f"⚠️ ที่พบ:\n{explanation}\n\n"
        f"📋 คำแนะนำ: {advice}"
    )
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_scorer.py tests/test_formatter.py -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add app/scorer.py app/formatter.py tests/test_scorer.py tests/test_formatter.py
git commit -m "feat: risk scorer and LINE message formatter"
```

---

## Task 7: NLP pipeline + wire text flow end-to-end

**Files:**
- Create: `app/pipelines/nlp.py`
- Create: `app/router.py`
- Modify: `app/line_handler.py`

- [ ] **Step 1: Write app/pipelines/nlp.py**

```python
from ml.classifier import predict

def run(text: str, interpreter, vocab: dict) -> dict:
    return predict(text, interpreter, vocab)
```

- [ ] **Step 2: Write app/router.py**

```python
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
```

- [ ] **Step 3: Rewrite app/line_handler.py with full text pipeline**

```python
import os
import asyncio
from fastapi import Request, HTTPException
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, PushMessageRequest, TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent
from app.router import detect_type
from app.pipelines import nlp as nlp_pipeline
from app.scorer import compute_risk_score
from app.formatter import format_response

_config = Configuration(access_token=os.environ["LINE_CHANNEL_ACCESS_TOKEN"])
_handler = WebhookHandler(os.environ["LINE_CHANNEL_SECRET"])

_FALLBACK_ADVICE = {
    "สูง": "อย่ากดลิงก์ ห้ามกรอกข้อมูล รายงาน ETDA 1212 หรือโทร 1441",
    "ปานกลาง": "ระวัง ตรวจสอบแหล่งที่มาก่อนดำเนินการ",
    "ต่ำ": "ดูเหมือนปกติ แต่ยังควรระวังข้อความที่ไม่คาดคิด",
}


async def handle_webhook(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = (await request.body()).decode()
    try:
        _handler.handle(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return "OK"


def _reply(reply_token: str, text: str):
    with ApiClient(_config) as client:
        MessagingApi(client).reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text=text)])
        )


def _push(user_id: str, text: str):
    with ApiClient(_config) as client:
        MessagingApi(client).push_message(
            PushMessageRequest(to=user_id, messages=[TextMessage(text=text)])
        )


def _build_response(score_result: dict, explanation: str = None) -> str:
    advice = _FALLBACK_ADVICE.get(score_result["risk_level"], "")
    expl = explanation or "\n".join(f"• {e}" for e in score_result["evidence"]) or "• ไม่พบรูปแบบที่น่าสงสัย"
    return format_response(
        score=score_result["score"],
        risk_level=score_result["risk_level"],
        explanation=expl,
        advice=advice,
    )


def _process_text(text: str, state) -> str:
    nlp_result = nlp_pipeline.run(text, state.interpreter, state.vocab)
    score_result = compute_risk_score(nlp_result=nlp_result)
    return _build_response(score_result)


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

    if msg_type == "text":
        result_text = _process_text(event.message.text, state)
    else:
        result_text = "🚧 URL analysis coming soon"

    _push(user_id, result_text)


@_handler.add(MessageEvent, message=ImageMessageContent)
def on_image(event: MessageEvent):
    _reply(event.reply_token, "🔍 กำลังวิเคราะห์รูปภาพ...")
    _push(event.source.user_id, "🚧 Image analysis coming soon")
```

- [ ] **Step 4: Test end-to-end with ngrok**

```bash
uvicorn app.main:app --reload --port 8000
```

Send a scam-like Thai text to LINE bot:
```
ยินดีด้วย บัญชีกสิกรของคุณได้รับโบนัส กดรับที่ลิงก์
```
Expected: bot replies "🔍 กำลังวิเคราะห์..." then pushes risk score result.

- [ ] **Step 5: Commit**

```bash
git add app/pipelines/nlp.py app/router.py app/line_handler.py
git commit -m "feat: NLP pipeline wired into LINE bot — text classify demo working"
```

---

## Task 8: URL pipeline (Day 3)

**Files:**
- Create: `app/pipelines/url.py`
- Create: `tests/test_url.py`
- Modify: `app/line_handler.py`

- [ ] **Step 1: Write failing tests for domain rules (no API)**

Create `tests/test_url.py`:

```python
from app.pipelines.url import check_domain_impersonation, extract_domain

def test_extract_domain():
    assert extract_domain("http://kbank-th.xyz/login") == "kbank-th.xyz"
    assert extract_domain("https://www.scb-reward.xyz/claim") == "scb-reward.xyz"

def test_impersonation_detected():
    brand, detected = check_domain_impersonation("http://kbank-bonus.xyz/claim")
    assert detected is True
    assert brand == "กสิกรไทย"

def test_legit_domain_not_flagged():
    brand, detected = check_domain_impersonation("https://kbank.co.th/login")
    assert detected is False

def test_scb_impersonation():
    brand, detected = check_domain_impersonation("http://scb-reward.xyz")
    assert detected is True
    assert brand == "ไทยพาณิชย์"

def test_unrelated_domain_not_flagged():
    brand, detected = check_domain_impersonation("https://example.com")
    assert detected is False
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_url.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write app/pipelines/url.py**

```python
import base64
import hashlib
import os
import time
from urllib.parse import urlparse
import httpx

_VT_BASE = "https://www.virustotal.com/api/v3"
_HEADERS = lambda: {"x-apikey": os.environ["VIRUSTOTAL_API_KEY"]}

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


def check_domain_impersonation(url: str) -> tuple[str | None, bool]:
    domain = extract_domain(url)
    for keyword, (brand, legit) in _LEGIT_DOMAINS.items():
        if keyword in domain and domain != legit:
            return brand, True
    return None, False


async def check_url(url: str) -> dict:
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cached = _cache.get(url_hash)
    if cached and time.time() - cached["_ts"] < 3600:
        return {k: v for k, v in cached.items() if k != "_ts"}

    brand, impersonated = check_domain_impersonation(url)
    vt_result = await _virustotal_check(url)
    result = {
        "malicious": vt_result.get("malicious", 0),
        "total_engines": vt_result.get("total_engines", 0),
        "domain_age_days": vt_result.get("domain_age_days", 999),
        "impersonation": brand if impersonated else None,
    }
    _cache[url_hash] = {**result, "_ts": time.time()}
    return result


async def _virustotal_check(url: str) -> dict:
    url_id = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{_VT_BASE}/urls/{url_id}", headers=_HEADERS())
            if r.status_code != 200:
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
```

- [ ] **Step 4: Run domain rule tests**

```bash
pytest tests/test_url.py -v
```

Expected: all 5 tests PASS (domain rules don't need API key).

- [ ] **Step 5: Update line_handler.py to handle URLs**

In `on_text` handler, replace the `else` branch:

```python
    if msg_type == "text":
        result_text = _process_text(event.message.text, state)
    elif msg_type in ("url", "text_with_url"):
        result_text = await _process_text_with_url(event.message.text, state)
    else:
        result_text = "🚧 Format not yet supported"
```

Add `_process_text_with_url` function in `line_handler.py`:

```python
async def _process_text_with_url(text: str, state) -> str:
    from app.router import extract_urls
    from app.pipelines.url import check_url

    urls = extract_urls(text)
    nlp_result = nlp_pipeline.run(text, state.interpreter, state.vocab)

    url_result = None
    if urls:
        url_result = await check_url(urls[0])  # check first URL only

    score_result = compute_risk_score(nlp_result=nlp_result, url_result=url_result)
    return _build_response(score_result)
```

Also update `on_text` to be async:

```python
@_handler.add(MessageEvent, message=TextMessageContent)
def on_text(event: MessageEvent):
    from app.main import app as _app
    import asyncio
    state = _app.state
    msg_type = detect_type(event)

    if msg_type == "unsupported":
        _reply(event.reply_token, "ขอโทษ ไม่รองรับรูปแบบนี้")
        return

    _reply(event.reply_token, "🔍 กำลังวิเคราะห์...")
    user_id = event.source.user_id

    async def _run():
        if msg_type == "text":
            return _process_text(event.message.text, state)
        elif msg_type in ("url", "text_with_url"):
            return await _process_text_with_url(event.message.text, state)
        return "ไม่รองรับรูปแบบนี้"

    result_text = asyncio.run(_run())
    _push(user_id, result_text)
```

- [ ] **Step 6: Test with a phishing URL (requires VT API key in .env)**

Send to LINE bot: `http://scb-reward.xyz/login`  
Expected: bot returns risk score with domain impersonation evidence.

- [ ] **Step 7: Commit**

```bash
git add app/pipelines/url.py tests/test_url.py app/line_handler.py
git commit -m "feat: URL pipeline with VirusTotal and domain impersonation rules"
```

---

## Task 9: OCR pipeline (Day 4)

**Files:**
- Create: `app/pipelines/ocr.py`
- Modify: `app/line_handler.py`

- [ ] **Step 1: Write app/pipelines/ocr.py**

```python
import asyncio
import base64
import hashlib
import os
import httpx

_TYPHOON_BASE = "https://api.opentyphoon.ai/v1/chat/completions"
_SEMAPHORE = asyncio.Semaphore(1)  # 2 req/s limit
_cache: dict[str, str] = {}  # sha256(image_bytes) → extracted_text


async def extract_text(image_bytes: bytes) -> str:
    img_hash = hashlib.sha256(image_bytes).hexdigest()
    if img_hash in _cache:
        return _cache[img_hash]

    async with _SEMAPHORE:
        text = await _call_typhoon_ocr(image_bytes)

    _cache[img_hash] = text
    return text


async def _call_typhoon_ocr(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": "typhoon-ocr-preview",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "สกัดข้อความทั้งหมดในภาพ ตอบเฉพาะข้อความที่เห็นในภาพเท่านั้น"},
            ],
        }],
        "max_tokens": 500,
    }
    headers = {
        "Authorization": f"Bearer {os.environ['TYPHOON_API_KEY']}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(_TYPHOON_BASE, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""
```

- [ ] **Step 2: Update on_image in line_handler.py**

Replace the `on_image` handler:

```python
@_handler.add(MessageEvent, message=ImageMessageContent)
def on_image(event: MessageEvent):
    from app.main import app as _app
    from linebot.v3.messaging import MessagingApiBlob
    from app.pipelines import ocr as ocr_pipeline
    from app.router import extract_urls
    from app.pipelines.url import check_url
    import asyncio

    state = _app.state
    _reply(event.reply_token, "🔍 กำลังวิเคราะห์รูปภาพ...")
    user_id = event.source.user_id

    async def _run():
        with ApiClient(_config) as client:
            blob_api = MessagingApiBlob(client)
            image_bytes = blob_api.get_message_content(event.message.id)

        extracted_text = await ocr_pipeline.extract_text(bytes(image_bytes))
        if not extracted_text:
            return "❌ ไม่สามารถอ่านข้อความจากรูปภาพได้"

        nlp_result = nlp_pipeline.run(extracted_text, state.interpreter, state.vocab)
        urls = extract_urls(extracted_text)
        url_result = await check_url(urls[0]) if urls else None
        score_result = compute_risk_score(nlp_result=nlp_result, url_result=url_result)
        return _build_response(score_result)

    result_text = asyncio.run(_run())
    _push(user_id, result_text)
```

- [ ] **Step 3: Test with a screenshot (requires TYPHOON_API_KEY in .env)**

Send a screenshot of a Thai scam SMS to the bot.  
Expected: bot OCRs → classifies → returns risk score.

- [ ] **Step 4: Commit**

```bash
git add app/pipelines/ocr.py app/line_handler.py
git commit -m "feat: OCR pipeline via typhoon-ocr for screenshot analysis"
```

---

## Task 10: QR pipeline

**Files:**
- Create: `app/pipelines/qr.py`
- Modify: `app/line_handler.py` (on_image)

- [ ] **Step 1: Write app/pipelines/qr.py**

```python
from PIL import Image
from pyzbar.pyzbar import decode
import io


def decode_qr(image_bytes: bytes) -> str | None:
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
```

- [ ] **Step 2: Update on_image to try QR first**

In `on_image._run()`, add QR check before OCR:

```python
    async def _run():
        with ApiClient(_config) as client:
            blob_api = MessagingApiBlob(client)
            image_bytes = blob_api.get_message_content(event.message.id)
        image_bytes = bytes(image_bytes)

        # Try QR decode first
        from app.pipelines.qr import decode_qr
        from app.pipelines.url import check_url
        qr_url = decode_qr(image_bytes)
        if qr_url:
            url_result = await check_url(qr_url)
            score_result = compute_risk_score(url_result=url_result)
            return _build_response(score_result)

        # Fall back to OCR
        extracted_text = await ocr_pipeline.extract_text(image_bytes)
        if not extracted_text:
            return "❌ ไม่สามารถอ่านข้อความจากรูปภาพได้"

        nlp_result = nlp_pipeline.run(extracted_text, state.interpreter, state.vocab)
        urls = extract_urls(extracted_text)
        url_result = await check_url(urls[0]) if urls else None
        score_result = compute_risk_score(nlp_result=nlp_result, url_result=url_result)
        return _build_response(score_result)
```

- [ ] **Step 3: Test with QR image**

Generate a QR code for `http://scb-reward.xyz` and send to bot.  
Expected: bot decodes QR → scans URL → returns risk score (no OCR needed).

- [ ] **Step 4: Commit**

```bash
git add app/pipelines/qr.py app/line_handler.py
git commit -m "feat: QR decode pipeline via pyzbar"
```

---

## Task 11: Typhoon v2.1 Explainer (Day 5)

**Files:**
- Create: `app/explainer.py`
- Modify: `app/line_handler.py`

- [ ] **Step 1: Write app/explainer.py**

```python
import os
import httpx

_TYPHOON_BASE = "https://api.opentyphoon.ai/v1/chat/completions"

_FALLBACK = {
    "สูง": "• ข้อความ/ลิงก์นี้มีลักษณะต้องสงสัยสูงมาก",
    "ปานกลาง": "• ข้อความ/ลิงก์นี้มีบางส่วนที่น่าสงสัย",
    "ต่ำ": "• ไม่พบรูปแบบ scam ที่ชัดเจน",
}


async def explain(score: float, risk_level: str, evidence: list[str], original_text: str) -> str:
    evidence_str = "\n".join(f"- {e}" for e in evidence) or "- ไม่พบหลักฐานเพิ่มเติม"
    prompt = f"""คุณเป็นผู้เชี่ยวชาญด้านความปลอดภัยไซเบอร์ของไทย

ข้อมูลจากระบบวิเคราะห์:
- คะแนนความเสี่ยง: {score:.0%} (ระดับ{risk_level})
- หลักฐานที่พบ:
{evidence_str}
- ข้อความต้นฉบับ: "{original_text[:200]}"

เขียนคำอธิบายสั้นๆ ให้ผู้ใช้ทั่วไปเข้าใจได้ (ไม่เกิน 3 bullet points ขึ้นต้นด้วย •) และคำแนะนำ 1 บรรทัด
ตอบเป็นภาษาไทยเท่านั้น ห้ามใช้ศัพท์เทคนิค ตอบในรูปแบบ:
BULLETS:
• ...
• ...
ADVICE:
..."""

    headers = {
        "Authorization": f"Bearer {os.environ['TYPHOON_API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "typhoon-v2.1-12b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.3,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(_TYPHOON_BASE, json=payload, headers=headers)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return _parse_response(content)
    except Exception:
        return _FALLBACK.get(risk_level, "• ไม่สามารถวิเคราะห์เพิ่มเติมได้"), "โปรดระวังและตรวจสอบแหล่งที่มา"


def _parse_response(content: str) -> tuple[str, str]:
    bullets, advice = "", ""
    lines = content.splitlines()
    in_bullets = False
    in_advice = False
    bullet_lines = []
    advice_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == "BULLETS:":
            in_bullets, in_advice = True, False
        elif stripped == "ADVICE:":
            in_bullets, in_advice = False, True
        elif in_bullets and stripped:
            bullet_lines.append(stripped)
        elif in_advice and stripped:
            advice_lines.append(stripped)
    bullets = "\n".join(bullet_lines) or "• วิเคราะห์แล้ว"
    advice = " ".join(advice_lines) or "โปรดระวัง"
    return bullets, advice
```

- [ ] **Step 2: Update _build_response in line_handler.py to use explainer**

Replace `_build_response` with:

```python
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
```

Update all `_process_text`, `_process_text_with_url`, and `on_image._run()` to call `await _build_response_async(score_result, original_text)` instead of `_build_response(score_result)`.

- [ ] **Step 3: Test full pipeline**

Send to bot: `"ยินดีด้วย บัญชีกสิกรของคุณได้รับโบนัส 5,000 บาท กดรับที่ http://kbank-bonus.xyz/claim"`

Expected: 🔴 risk score + Thai explanation with 3 bullets + advice.

- [ ] **Step 4: Commit**

```bash
git add app/explainer.py app/line_handler.py
git commit -m "feat: Typhoon v2.1 Thai explanation with rule-based fallback"
```

---

## Task 12: Deploy to Railway

- [ ] **Step 1: Push to GitHub**

```bash
git remote add origin https://github.com/<your-username>/scamradar.git
git push -u origin main
```

- [ ] **Step 2: Link to Railway**

- Railway dashboard → New Project → Deploy from GitHub repo → select `scamradar`
- Add environment variables in Railway dashboard (from `.env`)

- [ ] **Step 3: Set LINE webhook URL**

After Railway deploy completes, set webhook URL in LINE Developer Console:
```
https://<railway-app>.railway.app/callback
```

- [ ] **Step 4: Verify /health**

```bash
curl https://<railway-app>.railway.app/health
```

Expected: `{"status":"ok","model":"loaded"}`

- [ ] **Step 5: End-to-end smoke test**

Send all 4 format types to the deployed bot:
1. Thai scam text → 🔴
2. Phishing URL → 🔴
3. Normal SMS text → 🟢
4. Screenshot of scam → 🔴

- [ ] **Step 6: Commit final state**

```bash
git add .
git commit -m "chore: production deploy verified on Railway"
```

---

## Test Cases Reference

| Input | Expected label | Expected risk |
|-------|---------------|--------------|
| `"ยินดีด้วย บัญชีกสิกรได้รับโบนัส กดรับลิงก์"` | fraud | 🔴 สูง |
| `"สินค้าจัดส่งแล้ว ติดตามพัสดุที่ไปรษณีย์ไทย"` | ham | 🟢 ต่ำ |
| `http://scb-reward.xyz/login` | — | 🔴 สูง (domain impersonation) |
| Screenshot SMS scam | fraud (via OCR) | 🔴–🟡 |
| QR → `http://kbank-bonus.xyz` | — | 🔴 สูง |
