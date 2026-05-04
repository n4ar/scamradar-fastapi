import json
import numpy as np
from keras.models import load_model as _load_keras
from pythainlp.tokenize import word_tokenize

PAD_IDX = 0
START_IDX = 1
UNK_IDX = 2


def load_vocab(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model(path: str):
    return _load_keras(path)


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def predict(text: str, model, vocab: dict, config: dict) -> dict:
    max_len = config["max_len"]
    threshold = config.get("threshold", 0.5)

    tokens = word_tokenize(text, engine="newmm") if text.strip() else []
    encoded = [START_IDX] + [vocab.get(t, UNK_IDX) for t in tokens]
    if len(encoded) >= max_len:
        encoded = encoded[-max_len:]  # truncate from start (pre-truncate)
    padded = [PAD_IDX] * (max_len - len(encoded)) + encoded  # pre-padding

    inp = np.array([padded], dtype=np.int32)
    prob = float(model.predict(inp, verbose=0)[0][0])

    label = "fraud" if prob >= threshold else "ham"
    gap = abs(prob - threshold)
    confidence = "high" if gap > 0.3 else "medium" if gap > 0.15 else "low"
    return {"scam_prob": prob, "label": label, "confidence": confidence}
