import asyncio
from app.pipelines.nlp import run as run_nlp
from ml.classifier import load_vocab, load_interpreter

text1 = "ยินดีด้วย บัญชีกสิกรของคุณได้รับโบนัส กดรับที่ลิงก์"

vocab = load_vocab("ml/vocab.txt")
interpreter = load_interpreter("ml/model.tflite")

r1 = run_nlp(text1, interpreter, vocab)
print(f"NLP Scam: {r1}")

