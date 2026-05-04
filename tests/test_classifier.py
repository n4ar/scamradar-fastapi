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
