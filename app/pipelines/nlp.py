from ml.classifier import predict

def run(text: str, interpreter, vocab: dict) -> dict:
    return predict(text, interpreter, vocab)
