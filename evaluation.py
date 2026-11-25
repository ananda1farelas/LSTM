"""
Orang 3 — Evaluasi Model Stacked LSTM untuk Klasifikasi Sentimen Ulasan Grab

Letakkan file ini di folder yang sama dengan:
- modeling_training.py
- preprocessing.py
- artifacts/model_stacked_lstm.keras
- artifacts/tokenizer.json
- artifacts/X_test.npy
- artifacts/y_test.npy
- artifacts/data_test.csv
- artifacts/training_config.json
- artifacts/history.csv

Jalankan:
    python evaluation_orang3.py

Output disimpan di folder artifacts/evaluation/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# TensorFlow diimpor di dalam main agar pesan error dependensi lebih jelas.

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR = ARTIFACT_DIR / "evaluation"

MODEL_PATH = ARTIFACT_DIR / "model_stacked_lstm.keras"
TOKENIZER_PATH = ARTIFACT_DIR / "tokenizer.json"
X_TEST_PATH = ARTIFACT_DIR / "X_test.npy"
Y_TEST_PATH = ARTIFACT_DIR / "y_test.npy"
TEST_CSV_PATH = ARTIFACT_DIR / "data_test.csv"
CONFIG_PATH = ARTIFACT_DIR / "training_config.json"
HISTORY_PATH = ARTIFACT_DIR / "history.csv"

LABELS = [0, 1, 2]
LABEL_NAMES = ["Negatif", "Netral", "Positif"]
LABEL_MAPPING = {0: "Negatif", 1: "Netral", 2: "Positif"}

# Contoh ulasan baru untuk demonstrasi prediksi.
# Boleh diganti dengan ulasan buatan sendiri, tetapi jangan gunakan data test.
NEW_REVIEWS = [
    "Aplikasi Grab sangat membantu, driver ramah dan perjalanan nyaman.",
    "Pesanan saya terlambat, driver sulit dihubungi dan pelayanan buruk.",
    "Aplikasinya biasa saja, kadang lancar kadang lambat.",
    "Promo menarik dan proses pemesanan mudah sekali.",
    "Saldo terpotong tetapi pesanan dibatalkan, sangat mengecewakan.",
]

# Normalisasi ini disamakan dengan kamus pada preprocessing.py milik Orang 1.
SLANG_DICT = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "g": "tidak", "tdk": "tidak",
    "nggak": "tidak", "engga": "tidak", "enggak": "tidak",
    "bgt": "banget", "bngt": "banget", "mantul": "mantap betul",
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "krn": "karena", "karna": "karena",
    "jd": "jadi", "jg": "juga", "aja": "saja", "udh": "sudah", "udah": "sudah",
    "sdh": "sudah", "blm": "belum", "blum": "belum", "trs": "terus", "trus": "terus",
    "dr": "dari", "cm": "cuma", "cuma": "hanya", "org": "orang", "skrg": "sekarang", "bs": "bisa",
    "pdhl": "padahal", "pdahal": "padahal", "pake": "pakai", "pk": "pakai",
    "gimana": "bagaimana", "gmn": "bagaimana", "knp": "kenapa", "kenapa": "mengapa",
    "sy": "saya", "sya": "saya", "gw": "saya", "gue": "saya", "ane": "saya",
    "lu": "kamu", "lo": "kamu", "dpt": "dapat", "tp": "tapi", "tapi": "tetapi",
    "bnyk": "banyak", "byk": "banyak", "apk": "aplikasi", "app": "aplikasi",
    "driver": "pengemudi", "telat": "terlambat", "telattt": "terlambat",
    "parah": "sangat buruk", "oke": "baik", "ok": "baik",
}


def clean_text(text: str) -> str:
    """Meniru fungsi clean_text() pada preprocessing.py."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s!?.,]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_slang(text: str) -> str:
    """Meniru normalisasi slang pada preprocessing.py."""
    return " ".join(SLANG_DICT.get(word, word) for word in text.split())


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"File wajib tidak ditemukan: {path}\n"
            "Pastikan file evaluation_orang3.py berada di folder LSTM dan ZIP sudah diekstrak."
        )


def save_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(LABEL_NAMES)),
        yticks=np.arange(len(LABEL_NAMES)),
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        xlabel="Label Prediksi",
        ylabel="Label Aktual",
        title="Confusion Matrix — Evaluasi Test Set",
    )
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2 if cm.size and cm.max() > 0 else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col, row, f"{cm[row, col]:,}",
                ha="center", va="center",
                color="white" if cm[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_training_diagnostic(output_path: Path, y_test: np.ndarray) -> None:
    """Meringkas indikasi overfitting/underfitting dari history Orang 2."""
    lines = ["ANALISIS SINGKAT TRAINING DAN VALIDATION", "=" * 48]

    if not HISTORY_PATH.exists():
        lines.append("history.csv tidak ditemukan; analisis kurva training dilewati.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    history = pd.read_csv(HISTORY_PATH)
    required = {"epoch", "accuracy", "loss", "val_accuracy", "val_loss"}
    if not required.issubset(history.columns) or history.empty:
        lines.append("Kolom history.csv tidak lengkap atau file kosong.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    best_loss_row = history.loc[history["val_loss"].idxmin()]
    best_acc_row = history.loc[history["val_accuracy"].idxmax()]
    last_row = history.iloc[-1]
    majority_baseline = float(pd.Series(y_test).value_counts(normalize=True).max())
    three_class_random_loss = float(np.log(3))

    lines.extend([
        f"Jumlah epoch yang tercatat: {len(history)}",
        f"Epoch dengan validation loss terendah: {int(best_loss_row['epoch'])}",
        f"Validation loss terendah: {best_loss_row['val_loss']:.4f}",
        f"Training accuracy pada epoch tersebut: {best_loss_row['accuracy']:.4f}",
        f"Validation accuracy pada epoch tersebut: {best_loss_row['val_accuracy']:.4f}",
        f"Validation accuracy tertinggi: {best_acc_row['val_accuracy']:.4f} (epoch {int(best_acc_row['epoch'])})",
        f"Training accuracy pada epoch terakhir: {last_row['accuracy']:.4f}",
        f"Validation accuracy pada epoch terakhir: {last_row['val_accuracy']:.4f}",
        f"Training loss pada epoch terakhir: {last_row['loss']:.4f}",
        f"Validation loss pada epoch terakhir: {last_row['val_loss']:.4f}",
        f"Baseline accuracy jika selalu memilih kelas mayoritas pada test set: {majority_baseline:.4f}",
        f"Acuan loss untuk prediksi acak 3 kelas (ln 3): {three_class_random_loss:.4f}",
        "",
    ])

    train_acc = float(last_row["accuracy"])
    val_acc = float(last_row["val_accuracy"])
    train_loss = float(last_row["loss"])
    val_loss = float(last_row["val_loss"])

    if train_acc - val_acc >= 0.10 and val_loss > train_loss:
        diagnosis = (
            "Ada indikasi overfitting: performa training lebih tinggi daripada validation "
            "dan validation loss lebih besar. Periksa gap antarkurva dan pertimbangkan "
            "regularisasi, dropout, atau early stopping."
        )
    elif train_acc < 0.50 and val_acc < 0.50 and abs(train_loss - three_class_random_loss) < 0.15:
        diagnosis = (
            "Kurva belum menunjukkan pembelajaran yang kuat (indikasi underfitting atau "
            "training belum konvergen). Loss masih dekat dengan acuan prediksi acak 3 kelas. "
            "Perlu memeriksa kembali proses training, distribusi kelas, class_weight, "
            "tokenisasi, dan kesesuaian label."
        )
    else:
        diagnosis = (
            "Belum cukup bukti untuk menyimpulkan overfitting atau underfitting hanya dari "
            "ringkasan ini. Periksa grafik training_accuracy.png dan training_loss.png "
            "bersama metrik test set."
        )

    lines.extend(["Interpretasi awal:", diagnosis,
                  "Catatan: diagnosis ini bersifat indikatif, bukan kesimpulan final."])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Pastikan semua artefak Orang 2 tersedia.
    for path in [MODEL_PATH, TOKENIZER_PATH, X_TEST_PATH, Y_TEST_PATH, TEST_CSV_PATH, CONFIG_PATH]:
        require_file(path)

    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
    except ImportError as exc:
        raise ImportError(
            "TensorFlow belum terpasang. Jalankan: pip install tensorflow scikit-learn pandas numpy matplotlib"
        ) from exc

    # 2. Muat model, data testing, konfigurasi, dan tokenizer.
    print("Memuat model dan data testing...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH).astype(int)
    test_df = pd.read_csv(TEST_CSV_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as file:
        tokenizer = tokenizer_from_json(file.read())

    max_len = int(config.get("max_len", X_test.shape[1]))
    if X_test.ndim != 2 or X_test.shape[1] != max_len:
        raise ValueError(f"Bentuk X_test {X_test.shape} tidak sesuai max_len={max_len}.")
    if len(X_test) != len(y_test) or len(y_test) != len(test_df):
        raise ValueError("Jumlah baris X_test, y_test, dan data_test.csv tidak sama.")
    if "label" not in test_df.columns or "text" not in test_df.columns:
        raise ValueError("data_test.csv harus memiliki kolom 'text' dan 'label'.")
    if not np.array_equal(y_test, test_df["label"].to_numpy(dtype=int)):
        raise ValueError(
            "Label pada y_test.npy tidak sama urutannya dengan label di data_test.csv. "
            "Jangan lanjutkan agar hasil evaluasi tidak salah pasangan."
        )
    if not set(np.unique(y_test)).issubset(set(LABELS)):
        raise ValueError("Ditemukan label di luar kelas 0, 1, dan 2.")

    # 3. Prediksi test set yang tidak digunakan untuk training.
    print(f"Mengevaluasi {len(X_test):,} data test...")
    probabilities = model.predict(X_test, batch_size=256, verbose=1)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(LABELS):
        raise ValueError(
            f"Output model seharusnya memiliki 3 kelas, tetapi bentuknya {probabilities.shape}."
        )

    y_pred = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)

    # 4. Metrik evaluasi.
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, labels=LABELS, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, labels=LABELS, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, labels=LABELS, average="macro", zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, labels=LABELS, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, labels=LABELS, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, labels=LABELS, average="weighted", zero_division=0)

    report_dict = classification_report(
        y_test, y_pred, labels=LABELS, target_names=LABEL_NAMES,
        output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        y_test, y_pred, labels=LABELS, target_names=LABEL_NAMES,
        digits=4, zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    # 5. Simpan hasil prediksi setiap data testing.
    result_df = test_df.copy()
    result_df["actual_label"] = y_test
    result_df["actual_sentiment"] = [LABEL_MAPPING[int(v)] for v in y_test]
    result_df["predicted_label"] = y_pred
    result_df["predicted_sentiment"] = [LABEL_MAPPING[int(v)] for v in y_pred]
    result_df["confidence"] = confidence
    result_df["is_correct"] = y_test == y_pred
    for idx, name in enumerate(["probability_negative", "probability_neutral", "probability_positive"]):
        result_df[name] = probabilities[:, idx]
    result_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    # 6. Simpan baris yang salah diprediksi untuk analisis error.
    result_df.loc[~result_df["is_correct"]].to_csv(
        OUTPUT_DIR / "misclassified_reviews.csv", index=False, encoding="utf-8-sig"
    )

    # 7. Simpan classification report dan confusion matrix.
    pd.DataFrame(report_dict).T.to_csv(
        OUTPUT_DIR / "classification_report.csv", encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(
        cm, index=[f"Aktual_{name}" for name in LABEL_NAMES],
        columns=[f"Prediksi_{name}" for name in LABEL_NAMES],
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv", encoding="utf-8-sig")
    save_confusion_matrix(cm, OUTPUT_DIR / "confusion_matrix.png")

    metrics = {
        "test_size": int(len(y_test)),
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "actual_distribution": {
            LABEL_MAPPING[int(k)]: int(v)
            for k, v in zip(*np.unique(y_test, return_counts=True))
        },
        "predicted_distribution": {
            LABEL_MAPPING[int(k)]: int(v)
            for k, v in zip(*np.unique(y_pred, return_counts=True))
        },
        "misclassified_count": int((y_test != y_pred).sum()),
    }
    with open(OUTPUT_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    # 8. Prediksi ulasan baru menggunakan tokenizer dan preprocessing yang sama.
    cleaned_reviews = [normalize_slang(clean_text(review)) for review in NEW_REVIEWS]
    new_sequences = tokenizer.texts_to_sequences(cleaned_reviews)
    X_new = pad_sequences(new_sequences, maxlen=max_len, padding="post", truncating="post")
    new_probabilities = model.predict(X_new, verbose=0)
    new_predictions = np.argmax(new_probabilities, axis=1)

    new_review_df = pd.DataFrame({
        "review_asli": NEW_REVIEWS,
        "review_setelah_preprocessing": cleaned_reviews,
        "predicted_label": new_predictions,
        "predicted_sentiment": [LABEL_MAPPING[int(v)] for v in new_predictions],
        "confidence": np.max(new_probabilities, axis=1),
        "probability_negative": new_probabilities[:, 0],
        "probability_neutral": new_probabilities[:, 1],
        "probability_positive": new_probabilities[:, 2],
    })
    new_review_df.to_csv(
        OUTPUT_DIR / "new_review_predictions.csv", index=False, encoding="utf-8-sig"
    )

    # 9. Buat diagnosis awal dari history Orang 2.
    write_training_diagnostic(OUTPUT_DIR / "training_diagnostic.txt", y_test)

    # 10. Cetak ringkasan ke terminal.
    print("\n========== HASIL EVALUASI TEST SET ==========")
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision (macro) : {precision_macro:.4f}")
    print(f"Recall (macro)    : {recall_macro:.4f}")
    print(f"F1-score (macro)  : {f1_macro:.4f}")
    print("\nClassification Report:")
    print(report_text)
    print(f"Semua output tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
