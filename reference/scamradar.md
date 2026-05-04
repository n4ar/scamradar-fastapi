---
title: ScamRadar — AI Multimodal Scam Detection บน LINE
type: idea
tags: [cybersecurity, AI, NLP, LINE, hackathon, Thai-scam, multimodal, OCR]
created: 2026-05-03
updated: 2026-05-04
---

# ScamRadar — AI Multimodal Scam Detection บน LINE

> **การแข่งขันเป้าหมาย:** [[competitions/leagues-of-code-hackathon-2]] (Leagues of Code AI & Cybersecurity Hackathon ครั้งที่ 2)
> **Deadline Proposal:** 4–9 พ.ค. 2569 | **Hack Day:** 30 พ.ค. – 1 มิ.ย. 2569 (44 ชม.)

---

## 1. ปัญหาและความสำคัญ

### สถิติปี 2568 (ไทย)
- คนไทยถูกโจมตีจาก scam **173 ล้านครั้ง** (สายโทรศัพท์ + SMS)
- ความเสียหายรวม **23,000 ล้านบาท**
- ผู้เสียหายกว่า **405,000 ราย** (กลุ่มอายุ 20–49 ปี)
- SMS หลอกลวง: **134 ล้านข้อความ/ปี** (กระทรวง DE, 2569)

### ช่องว่างที่ยังไม่มีใครแก้
ภัย scam ในไทยปี 2569 ไม่ได้มาแค่ SMS — มาทาง **LINE** เป็นหลัก ทั้งในรูปแบบ:
- ข้อความหลอกแอบอ้าง (ธนาคาร, กรมสรรพากร, ตำรวจ)
- ลิงก์ phishing ที่ดูเหมือน URL จริง
- QR Code ที่นำไปยังเว็บปลอม
- **Screenshot** ที่ผู้ใช้รับมาแล้วไม่แน่ใจว่า scam หรือไม่

| เครื่องมือที่มีอยู่ | ข้อจำกัด |
|--------------------|---------|
| **Beebuzz** (KMUTT, 2566) | Android เท่านั้น, SMS เท่านั้น, ต้อง grant SMS permission |
| **ETDA 1212** | มนุษย์ดำเนินการ, ไม่ real-time, ไม่มี API |
| **Bitdefender Scamio** | ไม่รองรับภาษาไทย, ไม่เข้าใจบริบทไทย |
| **VirusTotal (manual)** | URL เท่านั้น, ต้องทำเอง, ไม่มี UI |

**ไม่มีเครื่องมือไทยใดที่:** วิเคราะห์ทุก format ได้ในที่เดียว + ใช้งานผ่าน LINE ที่มีอยู่แล้วโดยไม่ต้อง install อะไรเพิ่ม

---

## 2. Solution — ScamRadar LINE Bot

### Concept
**ScamRadar** คือ LINE Official Account ที่ใช้ AI วิเคราะห์ภัยออนไลน์แบบ **multimodal** ผู้ใช้เพียง forward หรือส่ง screenshot มาให้ bot แล้วจะได้รับผลวิเคราะห์ภายใน 3 วินาที

```
ผู้ใช้พบข้อความ/ภาพน่าสงสัย
        ↓
forward หรือ screenshot → @ScamRadar
        ↓
AI วิเคราะห์ < 3 วินาที
        ↓
Risk Score + เหตุผล + คำแนะนำ (ภาษาไทย)
```

### 4 Input Formats ที่รองรับ

| Format | ตัวอย่าง | การประมวลผล |
|--------|---------|------------|
| **ข้อความ** | "ยินดีด้วย! คุณได้รับรางวัล..." | WangchanBERTa classify |
| **Screenshot** | รูปภาพ LINE message | typhoon-ocr OCR → classify |
| **URL/ลิงก์** | `http://kbank-th.xyz/login` | VirusTotal API + domain analysis |
| **QR Code** | รูปภาพ QR | pyzbar decode → URL → VirusTotal |

### Output ที่ส่งกลับ
```
🔴 ความเสี่ยงสูง (87%)

⚠️ เหตุผล:
• ข้อความแอบอ้างเป็น "กสิกรไทย" แต่ลิงก์ไปยัง kbank-th.xyz
• 도메น ไม่ใช่ kbank.co.th จริง
• VirusTotal: 12/90 engine ตรวจพบ phishing

✅ คำแนะนำ:
อย่ากดลิงก์ ห้ามกรอกข้อมูล
รายงานไปที่ ETDA 1212 หรือโทร 1441
```

---

## 3. สถาปัตยกรรมระบบ

### 3.1 Architecture Decision — ไม่ใช่ LLM Wrapper

> **หลักการ:** WangchanBERTa = **classifier** (ML จริง), Typhoon v2.1 = **explainer เท่านั้น**

ผู้ใช้ส่ง **screenshot SMS** หรือ **copy-paste ข้อความ SMS** ผ่าน LINE — content จริงๆ คือ SMS scam text ซึ่งตรง domain ของ Beebuzz dataset (1,310 Thai SMS) พอดี → ไม่มี domain shift ปัญหา

Typhoon v2.1 รับ **structured evidence จาก pipeline** (risk score + URL result + keyword pattern) → อธิบายเหตุผลเป็นภาษาไทย — ไม่ได้ judge เองจากข้อความดิบ

```
INPUT (จาก LINE)
  ├── text (SMS copy-paste)  ──────────────────┐
  └── image (SMS screenshot) → typhoon-ocr ───→ WangchanBERTa (fine-tuned)
                                                       │ scam_prob
                                               URL ─→ VirusTotal
                                                       │ threat_score
                                               ┌───────▼────────┐
                                               │ Ensemble Scorer │
                                               │ weighted score  │
                                               └───────┬────────┘
                                                       │ {score, evidence}
                                               ┌───────▼────────────────────┐
                                               │ Typhoon v2.1-12b-instruct  │
                                               │ รับ structured evidence    │
                                               │ → อธิบายเหตุผลภาษาไทย     │
                                               └───────┬────────────────────┘
                                                       │
                                               LINE reply → ผู้ใช้
```

### 3.2 System Architecture (Full)

```
┌─────────────────────────────────────┐
│         LINE User (iOS/Android)     │
└────────────────┬────────────────────┘
                 │ Webhook (HTTPS)
┌────────────────▼────────────────────┐
│      LINE Messaging API             │
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│      ScamRadar Backend (FastAPI)    │
│  ┌─────────────────────────────┐   │
│  │ Message Router              │   │
│  │  ├── text? → NLP Pipeline  │   │
│  │  ├── image? → OCR Pipeline │   │
│  │  ├── url? → URL Pipeline   │   │
│  │  └── qr? → QR Pipeline     │   │
│  └────────────┬────────────────┘   │
│               │                     │
│  ┌────────────▼────────────────┐   │
│  │ Ensemble Scorer             │   │
│  │  NLP score + URL score      │   │
│  │  → weighted risk score      │   │
│  └────────────┬────────────────┘   │
│               │                     │
│  ┌────────────▼────────────────┐   │
│  │ Typhoon v2.1 Explainer      │   │
│  │  รับ structured evidence    │   │
│  │  → Thai explanation         │   │
│  └─────────────────────────────┘   │
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│   External APIs                     │
│   • VirusTotal (URL reputation)     │
│   • HuggingFace (WangchanBERTa)     │
│   • Typhoon API (explainer)         │
└─────────────────────────────────────┘
```

---

## 4. Tech Stack

### AI/ML Layer
| Component | Tool | เหตุผล |
|-----------|------|--------|
| Thai text classification | `WangchanBERTa` fine-tuned (airesearch/wangchanberta-base-att-spm-uncased) | BERT ภาษาไทย, fine-tune บน Beebuzz SMS dataset (1,310 samples, 98% val acc), **เป็น real ML classifier ไม่ใช่ LLM** |
| Thai OCR | `typhoon-ocr` (Typhoon OCR 1.5, 2B) | ฟรี, 2 req/s / 20 req/min, รองรับไทย+อังกฤษ — แปลง screenshot → text ก่อนส่ง classify |
| URL reputation | VirusTotal API v3 | 90+ antivirus engines, free tier |
| QR decode | `pyzbar` + `Pillow` | lightweight, ไม่ต้อง API |
| Thai explanation | `typhoon-v2.1-12b-instruct` (Typhoon API) | ฟรี, 5 req/s / 200 req/min, Thai-first LLM — **รับ structured evidence จาก pipeline เท่านั้น ไม่ classify เอง** |

### Backend
| Component        | Tool                                |
| ---------------- | ----------------------------------- |
| Framework        | FastAPI (Python)                    |
| LINE integration | `line-bot-sdk-python`               |
| Async            | `asyncio` + `httpx`                 |
| Model serving    | HuggingFace Inference API (free)    |
| Deploy           | Railway (free tier, HTTPS included) |

### Dataset สำหรับ Fine-tune
- **Thai SMS Spam (Beebuzz/KMUTT)** — 1,310 samples (655 fraud / 655 ham), pre-tokenized ด้วย PyThaiNLP → **ใช้ตรงๆ ได้เลย ไม่ต้อง augment** เพราะ input จริงของ ScamRadar คือ SMS text (screenshot หรือ copy-paste จาก SMS) ตรง domain พอดี
- **Manual collection** — screenshot scam จาก Facebook group "แจ้งเตือนภัยออนไลน์" (เพิ่ม coverage ถ้ามีเวลา)
- **PhishTank + VirusTotal** — labeled phishing URLs (สำหรับ URL pipeline)

---

## 5. ความแตกต่างจาก Prior Art

### vs Beebuzz (KMUTT)
| ด้าน | Beebuzz | ScamRadar |
|------|---------|-----------|
| Platform | Android app เท่านั้น | LINE = iOS + Android + Desktop |
| Input | SMS text | text + screenshot + URL + QR |
| Installation | ต้อง install + grant SMS permission | ไม่ต้อง install อะไร |
| Language model | GRU/LSTM (2023) | WangchanBERTa (BERT-based) |
| URL check | ❌ | ✅ VirusTotal integration |
| Screenshot OCR | ❌ | ✅ typhoon-ocr |
| Explanation | Risk score อย่างเดียว | อธิบายเหตุผลภาษาไทย |

### vs Bitdefender Scamio
| ด้าน | Scamio | ScamRadar |
|------|--------|-----------|
| ภาษาไทย | ❌ | ✅ |
| บริบทไทย (ธนาคารไทย, กรม...) | ❌ | ✅ |
| Platform | Web/WhatsApp | LINE (แพลตฟอร์มหลักคนไทย) |
| OCR screenshot | ❌ | ✅ |

---

## 6. แผน 44 ชั่วโมง (Hack Day)

### Hour 0–8: Foundation
- [ ] Setup LINE Official Account + Messaging API webhook
- [ ] FastAPI skeleton + Railway deploy (HTTPS)
- [ ] Basic text → WangchanBERTa pipeline (HuggingFace Inference API)
- [ ] VirusTotal URL check integration

### Hour 8–20: Core Features
- [ ] Image receive → typhoon-ocr OCR → classify pipeline
- [ ] QR decode → URL extract → scan pipeline
- [ ] Message Router (auto-detect input type)
- [ ] Ensemble scorer (weight NLP + URL scores)

### Hour 20–32: Polish
- [ ] LLM explainer (Typhoon v2.1-12b-instruct) → Thai explanation
- [ ] Response formatting (สวยงาม, emoji, readable)
- [ ] Error handling + fallback messages
- [ ] Rate limiting (ป้องกัน API abuse)

### Hour 32–40: Demo Prep
- [ ] เตรียม test cases ครบ 4 format (text/image/URL/QR)
- [ ] สร้าง demo script สำหรับ judge
- [ ] ทำ slide ประกอบ pitch

### Hour 40–44: Buffer
- [ ] Bug fixes
- [ ] Deploy check
- [ ] Rehearse pitch

---

## 7. MVP Scope (รอบ Proposal)

ถ้าต้องทำ demo สำหรับ Proposal video (3 นาที):

**Minimum viable demo:**
1. ส่งข้อความ scam ภาษาไทยให้ bot → ได้ risk score + explanation ✅
2. ส่ง URL phishing → ได้ VirusTotal result ✅
3. ส่ง screenshot SMS scam → OCR → classify ✅

**สิ่งที่ทำได้ใน 1–2 วัน (ก่อน deadline 4 พ.ค.):**
- LINE bot basic + text classify → ทำได้แน่นอน
- URL check → ทำได้แน่นอน
- OCR pipeline → ทำได้ถ้ามีเวลา

---

## 8. Impact & Scalability

### ระยะสั้น (Hack Day prototype)
- ทำงานได้จริง รองรับ text + URL + screenshot + QR

### ระยะกลาง
- Community reporting layer — aggregate anonymous reports
- Trending scam alert — แจ้งเตือน pattern ใหม่ที่กำลังระบาด
- Dashboard สำหรับ ETDA/DSI ดู threat intelligence

### ระยะยาว
- เชื่อมต่อ ETDA 1212 / DSI → auto-escalate เคส
- เชื่อมต่อ NCSEC (National Cyber Security Committee) database
- Expand ไป WhatsApp, Telegram

### Target Users
- **Primary:** ผู้สูงอายุ (เหยื่อหลัก) ที่ใช้ LINE เป็นแพลตฟอร์มหลัก
- **Secondary:** คนทั่วไปที่ต้องการ second opinion ก่อนกดลิงก์
- **Tertiary:** หน่วยงานรัฐ (ใช้ threat intelligence ที่ aggregate ได้)

---

## 9. ร่าง Proposal (2–3 หน้า)

### หน้า 1: ปัญหา + Solution Overview
*(ดูจาก section 1–2 ข้างบน)*

### หน้า 2: Technical Architecture + Differentiation
*(ดูจาก section 3–5 ข้างบน)*

### หน้า 3: Impact + Hack Day Plan
*(ดูจาก section 6–8 ข้างบน)*

---

## 10. ร่าง Video Pitch Script (3 นาที)

**[0:00–0:20] Hook — ปัญหา**
> "ปี 2568 คนไทยโดน scam 173 ล้านครั้ง เสียเงิน 23,000 ล้านบาท — และส่วนใหญ่มาผ่าน LINE"

**[0:20–0:50] Gap — ทำไมของเดิมไม่พอ**
> "Beebuzz อ่านได้แค่ SMS บน Android. ETDA 1212 ต้องรอมนุษย์ตอบ. ไม่มีเครื่องมือไทยใดที่วิเคราะห์ screenshot LINE ได้"

**[0:50–1:50] Demo — ScamRadar ทำงาน**
> Live demo: ส่ง screenshot scam → bot ตอบใน 3 วินาที
> Live demo: ส่ง URL → ได้ VirusTotal result
> Live demo: ส่ง QR → decode → scan → alert

**[1:50–2:20] Tech — ทำด้วยอะไร**
> WangchanBERTa + typhoon-ocr + typhoon-v2.1-12b-instruct + VirusTotal API + LINE Messaging API

**[2:20–2:50] Impact**
> "ใช้ได้ทันที ไม่ต้อง install อะไร ทุก platform ทุก format"

**[2:50–3:00] Call to action**
> "ScamRadar — เพราะภัยออนไลน์ไม่รอ"

---

## 11. ความเสี่ยงและ Mitigation

| ความเสี่ยง | โอกาส | Mitigation |
|-----------|-------|-----------|
| HuggingFace Inference API ช้า | สูง | ใช้ pipeline local ใน Railway ถ้าช้าเกิน |
| typhoon-ocr rate limit (20 req/min) | ปานกลาง | queue + cache ผล OCR ที่เคยทำแล้ว |
| VirusTotal rate limit (free: 4 req/min) | สูง | cache ผล URL ที่เคยตรวจแล้ว + queue |
| LINE bot webhook timeout (30s) | ปานกลาง | async processing + "กำลังวิเคราะห์..." reply ก่อน |
| Dataset ภาษาไทยน้อย | สูง | augment ด้วย paraphrase + ขอ dataset จาก ETDA/KMUTT |

---

## 12. Resources & References

- **Beebuzz source code:** KMUTT capstone — ขอดู model architecture เป็น baseline
- **WangchanBERTa:** `airesearch/wangchanberta-base-att-spm-uncased` (HuggingFace)
- **typhoon-ocr:** Typhoon OCR 1.5 (2B) — ฟรี, 2 req/s / 20 req/min (2025-11-14)
- **typhoon-v2.1-12b-instruct:** Typhoon API — ฟรี, 5 req/s / 200 req/min, Thai-first LLM (previous flagship)
- **VirusTotal API:** https://developers.virustotal.com/reference/overview
- **LINE Messaging API:** https://developers.line.biz/en/docs/messaging-api/
- **สถิติ scam ไทย:** กระทรวง DE, 2569 + ETDA Annual Report 2568 ([[research/ETDA-Annual-Report-2025]])
- **ETDA 1212:** https://www.etda.or.th/en/Our-Service/Online-Consumer-Protection.aspx

---

## Links
- [[competitions/leagues-of-code-hackathon-2]]
- [[research/ETDA-Annual-Report-2025]]
- [[patterns/rapid-prototyping]]
- [[patterns/winning-pattern-thailand]]
