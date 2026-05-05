from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from ml.classifier import load_vocab, load_interpreter
from app.line_handler import handle_webhook

@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.vote_db import init_db
    await init_db()
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
