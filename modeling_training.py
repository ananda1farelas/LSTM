"""
Orang 2 — Tokenisasi, Pembangunan Model, dan Training Stacked LSTM
Input : processed/grab_reviews_clean.csv (hasil preprocessing Orang 1)
Output: artifacts/model_stacked_lstm.keras
        artifacts/tokenizer.json
        artifacts/X_test.npy, artifacts/y_test.npy
        artifacts/data_test.csv
        artifacts/history.csv
        artifacts/training_accuracy.png
        artifacts/training_loss.png
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ============================================================
# 1. KONFIGURASI
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed" / "grab_reviews_clean.csv"
OUTPUT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MAX_WORDS = 20_000
MAX_LEN = 100
EMBEDDING_DIM = 128
EPOCHS = 15
BATCH_SIZE = 64
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15  # proporsi dari keseluruhan dataset
USE_CLASS_WEIGHT = True

LABEL_NAMES = {0: "Negatif", 1: "Netral", 2: "Positif"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def main():
    set_seed(RANDOM_SEED)

    # --------------------------------------------------------
    # 2. LOAD DAN VALIDASI DATASET
    # --------------------------------------------------------
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}\n"
        )

    df = pd.read_csv(DATA_PATH)
    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing_columns)}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1, 2}):
        raise ValueError("Label harus berupa 0 (Negatif), 1 (Netral), atau 2 (Positif).")

    if set(df["label"].unique()) != {0, 1, 2}:
        raise ValueError("Dataset harus memiliki ketiga kelas: 0, 1, dan 2.")

    texts = df["text"].tolist()
    labels = df["label"].to_numpy()

    print(f"\nJumlah data yang digunakan: {len(df):,}")
    print("Distribusi label:")
    for label, name in LABEL_NAMES.items():
        print(f"  {name:8s} ({label}): {(labels == label).sum():,}")

    # --------------------------------------------------------
    # 3. PEMBAGIAN TRAINING, VALIDATION, DAN TESTING
    # --------------------------------------------------------
    # Sisihkan data test dulu: 15%.
    x_train_val, x_test, y_train_val, y_test, idx_train_val, idx_test = (
        train_test_split(
            texts,
            labels,
            np.arange(len(texts)),
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=labels,
        )
    )

    # Validation dibuat 15% dari keseluruhan data.
    # Karena 15% diambil dari sisa 85%, proporsinya = 0.15 / 0.85.
    val_ratio_from_train_val = VALIDATION_SIZE / (1 - TEST_SIZE)

    x_train, x_val, y_train, y_val, idx_train, idx_val = train_test_split(
        x_train_val,
        y_train_val,
        idx_train_val,
        test_size=val_ratio_from_train_val,
        random_state=RANDOM_SEED,
        stratify=y_train_val,
    )

    print("\nPembagian dataset:")
    print(f"  Training   : {len(x_train):,} ({len(x_train)/len(df)*100:.1f}%)")
    print(f"  Validation : {len(x_val):,} ({len(x_val)/len(df)*100:.1f}%)")
    print(f"  Testing    : {len(x_test):,} ({len(x_test)/len(df)*100:.1f}%)")

    # --------------------------------------------------------
    # 4. TOKENISASI
    # --------------------------------------------------------
    # Tokenizer hanya di-fit pada teks training untuk mencegah
    # informasi kosakata dari validation/test ikut terbaca.
    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>",
        filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
        lower=True,
        split=" ",
    )
    tokenizer.fit_on_texts(x_train)

    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    print(f"\nJumlah kata unik pada word_index: {len(tokenizer.word_index):,}")
    print(f"Vocabulary size yang digunakan : {vocab_size:,}")

    # Mengubah teks menjadi sequence angka.
    train_sequences = tokenizer.texts_to_sequences(x_train)
    val_sequences = tokenizer.texts_to_sequences(x_val)
    test_sequences = tokenizer.texts_to_sequences(x_test)

    # Padding/truncating dibuat konsisten untuk seluruh subset.
    X_train = pad_sequences(
        train_sequences, maxlen=MAX_LEN, padding="post", truncating="post"
    )
    X_val = pad_sequences(
        val_sequences, maxlen=MAX_LEN, padding="post", truncating="post"
    )
    X_test = pad_sequences(
        test_sequences, maxlen=MAX_LEN, padding="post", truncating="post"
    )

    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    print("\nBentuk data setelah padding:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}, y_val  : {y_val.shape}")
    print(f"  X_test : {X_test.shape}, y_test : {y_test.shape}")

    # Simpan tokenizer untuk Orang 3.
    tokenizer_path = OUTPUT_DIR / "tokenizer.json"
    with open(tokenizer_path, "w", encoding="utf-8") as file:
        file.write(tokenizer.to_json())

    # Simpan data testing untuk Orang 3.
    np.save(OUTPUT_DIR / "X_test.npy", X_test)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    test_output = pd.DataFrame({
        "text": x_test,
        "label": y_test,
        "label_name": [LABEL_NAMES[int(label)] for label in y_test],
    })
    test_output.to_csv(OUTPUT_DIR / "data_test.csv", index=False)

    # Simpan indeks pembagian untuk dokumentasi/reproduksibilitas.
    split_info = {
        "random_seed": RANDOM_SEED,
        "total_data": len(df),
        "training_data": len(X_train),
        "validation_data": len(X_val),
        "testing_data": len(X_test),
        "max_words": MAX_WORDS,
        "vocab_size": vocab_size,
        "max_len": MAX_LEN,
        "embedding_dim": EMBEDDING_DIM,
        "epochs_max": EPOCHS,
        "batch_size": BATCH_SIZE,
        "label_mapping": LABEL_NAMES,
    }
    with open(OUTPUT_DIR / "training_config.json", "w", encoding="utf-8") as file:
        json.dump(split_info, file, indent=4, ensure_ascii=False)

    # --------------------------------------------------------
    # 5. PEMBANGUNAN ARSITEKTUR STACKED LSTM
    # --------------------------------------------------------
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nArsitektur model:")
    model.summary()

    # --------------------------------------------------------
    # 6. PENANGANAN KETIDAKSEIMBANGAN KELAS
    # --------------------------------------------------------
    # Bobot dihitung hanya dari label training
    class_weights = None
    if USE_CLASS_WEIGHT:
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        class_weights = {
            int(class_id): float(weight)
            for class_id, weight in zip(classes, weights)
        }
        print("\nClass weight (balanced) dari data training:")
        print(class_weights)

    # --------------------------------------------------------
    # 7. TRAINING MODEL
    # --------------------------------------------------------
    model_path = OUTPUT_DIR / "model_stacked_lstm.keras"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\nMulai proses training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Memuat model terbaik berdasarkan validation loss, lalu simpan ulang.
    best_model = tf.keras.models.load_model(model_path)
    best_model.save(model_path)

    # --------------------------------------------------------
    # 8. SIMPAN RIWAYAT TRAINING
    # --------------------------------------------------------
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(OUTPUT_DIR / "history.csv", index=False)

    # Grafik accuracy.
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training dan Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_accuracy.png", dpi=150)
    plt.close()

    # Grafik loss.
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training dan Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
    plt.close()

    # --------------------------------------------------------
    # 9. RINGKASAN HASIL DAN FILE UNTUK ORANG 3
    # --------------------------------------------------------
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    val_loss, val_accuracy = best_model.evaluate(X_val, y_val, verbose=0)

    result_summary = {
        "best_epoch_based_on_val_loss": best_epoch,
        "validation_loss": float(val_loss),
        "validation_accuracy": float(val_accuracy),
        "model_path": str(model_path.relative_to(BASE_DIR)),
        "tokenizer_path": str(tokenizer_path.relative_to(BASE_DIR)),
        "testing_features": "artifacts/X_test.npy",
        "testing_labels": "artifacts/y_test.npy",
        "testing_csv": "artifacts/data_test.csv",
    }
    with open(OUTPUT_DIR / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(result_summary, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("TRAINING SELESAI")
    print("=" * 65)
    print(f"Best epoch       : {best_epoch}")
    print(f"Validation loss  : {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    

if __name__ == "__main__":
    main()
