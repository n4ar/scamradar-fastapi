import numpy as np
from pythainlp.tokenize import word_tokenize

try:
    import tflite_runtime.interpreter as tflite
    _Interpreter = tflite.Interpreter
except ImportError:
    try:
        import tensorflow as tf
        _Interpreter = tf.lite.Interpreter
    except ImportError:
        # Fallback for environments without tflite-runtime or tensorflow
        class _Interpreter:
            def __init__(self, model_path): pass
            def allocate_tensors(self): pass
            def get_input_details(self): return [{"index": 0}]
            def get_output_details(self): return [{"index": 0}]
            def set_tensor(self, index, data): pass
            def invoke(self): pass
            def get_tensor(self, index): return np.array([[0.0]])

MAX_LEN = 110
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
    padded = [PAD_IDX] * (MAX_LEN - len(encoded)) + encoded

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
