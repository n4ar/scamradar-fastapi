"""Train GRU classifier on Thai SMS scam dataset (Beebuzz/KMUTT)."""
import csv
import json
import numpy as np
from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_PATH = Path("reference/Model/Thai-language-model/Dataset/Final-Data/final_data_th.csv")
ML_DIR = Path("ml")

# Special token indices (must match vocab.txt)
PAD_IDX = 0
START_IDX = 1
UNK_IDX = 2

EPOCHS = 30
BATCH_SIZE = 64
EMBED_DIM = 64
GRU_UNITS = 64
DROPOUT = 0.3
THRESHOLD = 0.5


def load_dataset():
    texts, labels = [], []
    with open(DATA_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            proc = row.get("processed_sentence", "").strip()
            label = int(row["label"])
            if proc:
                texts.append(proc)
                labels.append(label)
    return texts, labels


def build_vocab(texts: list[str]) -> dict:
    """Build vocab from space-tokenized processed_sentences."""
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    # Special tokens first (same scheme as original notebook)
    vocab = {"<PAD>": PAD_IDX, "<START>": START_IDX, "<UNKNOWN>": UNK_IDX, "<UNUSED>": 3}
    for i, (word, _) in enumerate(counter.most_common()):
        vocab[word] = i + 4  # offset by 4 special tokens
    return vocab


def encode(texts: list[str], vocab: dict, max_len: int) -> np.ndarray:
    """Encode texts → padded sequences (pre-padding, matching keras pad_sequences default)."""
    seqs = []
    for text in texts:
        tokens = text.split()
        encoded = [START_IDX] + [vocab.get(t, UNK_IDX) for t in tokens]
        encoded = encoded[:max_len]
        # pre-padding
        padded = [PAD_IDX] * (max_len - len(encoded)) + encoded
        seqs.append(padded)
    return np.array(seqs, dtype=np.int32)


def build_model(vocab_size: int, max_len: int) -> Sequential:
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, mask_zero=True),
        Bidirectional(GRU(GRU_UNITS, return_sequences=True)),
        Dropout(DROPOUT),
        GRU(GRU_UNITS // 2),
        Dropout(DROPOUT),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    print("Loading dataset...")
    texts, labels = load_dataset()
    print(f"  {len(texts)} samples | {sum(labels)} fraud / {len(labels)-sum(labels)} ham")

    # Compute max_len from data (cover 99th percentile)
    lengths = [len(t.split()) + 1 for t in texts]  # +1 for START token
    max_len = int(np.percentile(lengths, 99))
    print(f"  max_len (99th percentile): {max_len}")

    print("Building vocab...")
    # Build vocab from all data (same as notebook — fit before split)
    vocab = build_vocab(texts)
    vocab_size = len(vocab)
    print(f"  vocab size: {vocab_size}")

    print("Encoding sequences...")
    X = encode(texts, vocab, max_len)
    y = np.array(labels, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    print(f"  train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    print("Building model...")
    model = build_model(vocab_size, max_len)
    model.summary()

    callbacks = [EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)]

    print("Training...")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    print("\nEvaluating on test set...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc:.4f} | loss: {loss:.4f}")

    y_pred = (model.predict(X_test) > THRESHOLD).astype(int).flatten()
    print(classification_report(y_test, y_pred, target_names=["ham", "fraud"]))

    print("Saving artifacts...")
    ML_DIR.mkdir(exist_ok=True)
    model.save(ML_DIR / "model.keras")
    print(f"  model → {ML_DIR / 'model.keras'}")

    with open(ML_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  vocab → {ML_DIR / 'vocab.json'}")

    config = {"max_len": max_len, "threshold": THRESHOLD}
    with open(ML_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  config → {ML_DIR / 'config.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
