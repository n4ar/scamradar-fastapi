import pytest
from ml.classifier import load_vocab, load_model, load_config, predict

VOCAB_PATH = "ml/vocab.json"
MODEL_PATH = "ml/model.keras"
CONFIG_PATH = "ml/config.json"


@pytest.fixture(scope="module")
def classifier():
    vocab = load_vocab(VOCAB_PATH)
    model = load_model(MODEL_PATH)
    config = load_config(CONFIG_PATH)
    return vocab, model, config


def test_vocab_loads(classifier):
    vocab, _, _ = classifier
    assert "<PAD>" in vocab
    assert vocab["<PAD>"] == 0
    assert "<START>" in vocab
    assert vocab["<START>"] == 1
    assert "<UNKNOWN>" in vocab
    assert vocab["<UNKNOWN>"] == 2
    assert len(vocab) > 100


def test_config_loads(classifier):
    _, _, config = classifier
    assert "max_len" in config
    assert "threshold" in config
    assert config["max_len"] > 0
    assert 0 < config["threshold"] < 1


def test_predict_returns_expected_keys(classifier):
    vocab, model, config = classifier
    result = predict("ทดสอบ", model, vocab, config)
    assert "scam_prob" in result
    assert "label" in result
    assert "confidence" in result
    assert 0.0 <= result["scam_prob"] <= 1.0
    assert result["label"] in ("fraud", "ham")
    assert result["confidence"] in ("high", "medium", "low")


def test_predict_scam_text(classifier):
    vocab, model, config = classifier
    text = "ยินดีด้วย บัญชีกสิกรของคุณได้รับโบนัส กดรับที่ลิงก์"
    result = predict(text, model, vocab, config)
    assert result["scam_prob"] > 0.5
    assert result["label"] == "fraud"


def test_predict_ham_text(classifier):
    vocab, model, config = classifier
    text = "สวัสดีครับ ขอนัดหมายวันพรุ่งนี้เวลาสิบโมง"
    result = predict(text, model, vocab, config)
    assert result["scam_prob"] < 0.5
    assert result["label"] == "ham"


def test_predict_empty_text(classifier):
    vocab, model, config = classifier
    result = predict("", model, vocab, config)
    assert "scam_prob" in result
