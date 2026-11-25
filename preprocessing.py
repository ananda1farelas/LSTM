"""
Orang 1 — Data Preparation dan Exploratory Data Analysis (EDA)
Dataset: Grab App Reviews - Indonesia (Google Play Store)
Sumber  : https://www.kaggle.com/datasets/pandaa12/grab-app-reviews-indonesia-google-play-store

Tahapan:
1. Load dataset
2. Cek struktur, jumlah data, kolom, tipe data
3. Data cleaning (duplikat, kosong, tidak relevan, simbol)
4. Labeling sentimen dari rating bintang
5. Normalisasi kata tidak baku / slang bahasa Indonesia
6. EDA (distribusi kelas, panjang teks, wordcloud)
7. Simpan dataset bersih untuk Orang 2
"""

import re
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "raw" / "ulasan_com.grabtaxi.passenger.csv"
PROCESSED_DIR = BASE_DIR / "processed"
PLOTS_DIR = BASE_DIR / "eda_plots"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
print("=" * 70)
print("1. LOAD DATASET")
print("=" * 70)

df = pd.read_csv(RAW_PATH)
print(f"Jumlah baris awal : {len(df)}")
print(f"Kolom             : {list(df.columns)}")
print(df.dtypes)
print(df.head(3))

# ---------------------------------------------------------------------------
# 2. Struktur dataset
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. STRUKTUR & MISSING VALUE")
print("=" * 70)
print(df.isna().sum())
print(f"\nJumlah duplikat (baris penuh)   : {df.duplicated().sum()}")
print(f"Jumlah duplikat (kolom Ulasan)   : {df.duplicated(subset=['Ulasan']).sum()}")
print(f"\nDistribusi Rating:\n{df['Rating'].value_counts().sort_index()}")

# ---------------------------------------------------------------------------
# 3. Data cleaning
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. DATA CLEANING")
print("=" * 70)

df = df[["Ulasan", "Rating"]].copy()
df.columns = ["text", "rating"]

# 3a. Hapus baris dengan teks kosong / NaN
before = len(df)
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0]
df = df.dropna(subset=["text", "rating"])
print(f"Hapus teks kosong/NaN : {before - len(df)} baris dibuang")

# 3b. Hapus duplikat berdasarkan teks ulasan
before = len(df)
df = df.drop_duplicates(subset=["text"], keep="first")
print(f"Hapus duplikat        : {before - len(df)} baris dibuang")

# 3c. Buang ulasan yang terlalu pendek / tidak relevan (misal cuma angka/simbol)
def is_relevant(text: str) -> bool:
    letters = re.findall(r"[a-zA-Z]", text)
    return len(letters) >= 3  # minimal ada 3 huruf, bukan cuma simbol/angka

before = len(df)
df = df[df["text"].apply(is_relevant)]
print(f"Hapus tidak relevan    : {before - len(df)} baris dibuang")


def clean_text(text: str) -> str:
    """Bersihkan simbol/karakter tak perlu tanpa menghilangkan info sentimen."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # hapus URL
    text = re.sub(r"@\w+", " ", text)                       # hapus mention
    text = re.sub(r"[^a-z0-9\s!?.,]", " ", text)             # buang simbol aneh/emoji,
    # simpan ! ? . , karena bisa membawa nuansa sentimen (misal "bagus!!!")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)               # kurangi huruf berulang: "bagusss" -> "baguss"
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["text_clean"] = df["text"].apply(clean_text)

# buang lagi yang jadi kosong setelah cleaning
before = len(df)
df = df[df["text_clean"].str.len() > 0]
print(f"Hapus kosong setelah cleaning simbol : {before - len(df)} baris dibuang")

print(f"\nTotal baris setelah cleaning: {len(df)}")

# ---------------------------------------------------------------------------
# 4. Labeling sentimen dari rating bintang
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. LABELING SENTIMEN")
print("=" * 70)


def rating_to_label(rating: int) -> int:
    if rating <= 2:
        return 0  # Negatif
    elif rating == 3:
        return 1  # Netral
    else:
        return 2  # Positif


df["label"] = df["rating"].apply(rating_to_label)
label_names = {0: "Negatif", 1: "Netral", 2: "Positif"}
df["label_name"] = df["label"].map(label_names)

print(df["label_name"].value_counts())

# ---------------------------------------------------------------------------
# 5. Normalisasi kata tidak baku / slang Bahasa Indonesia
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. NORMALISASI SLANG")
print("=" * 70)

SLANG_DICT = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "g": "tidak", "tdk": "tidak",
    "nggak": "tidak", "engga": "tidak", "enggak": "tidak",
    "bgt": "banget", "bngt": "banget",
    "mantul": "mantap betul",
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "krn": "karena", "karna": "karena",
    "jd": "jadi", "jg": "juga", "aja": "saja", "udh": "sudah", "udah": "sudah",
    "sdh": "sudah", "blm": "belum", "blum": "belum",
    "trs": "terus", "trus": "terus", "dr": "dari",
    "cm": "cuma", "cuma": "hanya",
    "org": "orang", "skrg": "sekarang", "bs": "bisa", "pdhl": "padahal",
    "pdahal": "padahal", "pake": "pakai", "pk": "pakai",
    "gimana": "bagaimana", "gmn": "bagaimana", "knp": "kenapa", "kenapa": "mengapa",
    "sy": "saya", "sya": "saya", "gw": "saya", "gue": "saya", "ane": "saya",
    "lu": "kamu", "lo": "kamu",
    "dr": "dari", "dpt": "dapat", "tp": "tapi", "tapi": "tetapi",
    "bnyk": "banyak", "byk": "banyak",
    "apk": "aplikasi", "app": "aplikasi",
    "driver": "pengemudi",
    "telat": "terlambat", "telattt": "terlambat",
    "parah": "sangat buruk",
    "oke": "baik", "ok": "baik",
}


def normalize_slang(text: str) -> str:
    words = text.split()
    normalized = [SLANG_DICT.get(w, w) for w in words]
    return " ".join(normalized)


df["text_normalized"] = df["text_clean"].apply(normalize_slang)

print("Contoh sebelum/sesudah normalisasi:")
for i in range(3):
    print(f"  before : {df['text_clean'].iloc[i]}")
    print(f"  after  : {df['text_normalized'].iloc[i]}")
    print()

# ---------------------------------------------------------------------------
# 6. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("6. EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# 6a. Distribusi kelas sentimen
plt.figure(figsize=(7, 5))
order = ["Negatif", "Netral", "Positif"]
ax = sns.countplot(data=df, x="label_name", order=order, hue="label_name",
                    palette=["#e74c3c", "#f1c40f", "#2ecc71"], legend=False)
plt.title("Distribusi Kelas Sentimen Ulasan Grab")
plt.xlabel("Sentimen")
plt.ylabel("Jumlah Ulasan")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "distribusi_sentimen.png", dpi=150)
plt.close()
print("Saved: distribusi_sentimen.png")

# 6b. Panjang teks ulasan
df["text_length"] = df["text_normalized"].apply(lambda t: len(t.split()))

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="text_length", hue="label_name", bins=40, kde=True,
             palette=["#e74c3c", "#f1c40f", "#2ecc71"])
plt.title("Distribusi Panjang Teks Ulasan (jumlah kata) per Sentimen")
plt.xlabel("Jumlah kata")
plt.ylabel("Frekuensi")
plt.xlim(0, df["text_length"].quantile(0.99))
plt.tight_layout()
plt.savefig(PLOTS_DIR / "distribusi_panjang_teks.png", dpi=150)
plt.close()
print("Saved: distribusi_panjang_teks.png")

print("\nStatistik panjang teks (kata) per kelas:")
print(df.groupby("label_name")["text_length"].describe()[["count", "mean", "50%", "max"]])

# 6c. WordCloud per kelas sentimen
STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "pada",
    "tidak", "saya", "aplikasi", "nya", "ada", "sudah", "akan", "juga", "saja",
    "bisa", "karena", "tapi", "tetapi", "atau", "jadi", "jika", "kalau", "lagi",
    "banget", "sangat", "lebih", "sekali", "kok", "gak", "ga", "nih", "deh",
    "the", "a", "to", "is", "in", "grab", "grabfood",
}

for lbl in order:
    text_corpus = " ".join(df.loc[df["label_name"] == lbl, "text_normalized"])
    if not text_corpus.strip():
        continue
    wc = WordCloud(
        width=900, height=500, background_color="white",
        stopwords=STOPWORDS_ID, colormap="viridis", max_words=100,
    ).generate(text_corpus)

    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud Sentimen {lbl}", fontsize=14)
    plt.tight_layout()
    fname = f"wordcloud_{lbl.lower()}.png"
    plt.savefig(PLOTS_DIR / fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")

# ---------------------------------------------------------------------------
# 7. Simpan dataset bersih
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("7. SIMPAN DATASET BERSIH")
print("=" * 70)

final_df = df[["text_normalized", "label", "label_name", "rating"]].rename(
    columns={"text_normalized": "text"}
)
final_df = final_df.reset_index(drop=True)

out_path = PROCESSED_DIR / "grab_reviews_clean.csv"
final_df.to_csv(out_path, index=False)
print(f"Dataset bersih disimpan di: {out_path}")
print(f"Total baris final: {len(final_df)}")
print(final_df["label_name"].value_counts())

# simpan ringkasan statistik buat laporan
summary = {
    "total_baris_awal": 100000,
    "total_baris_setelah_cleaning": len(final_df),
    "distribusi_label": final_df["label_name"].value_counts().to_dict(),
    "rata_rata_panjang_teks_kata": round(df["text_length"].mean(), 2),
    "kolom_output": list(final_df.columns),
}
with open(PROCESSED_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\nRingkasan EDA disimpan di: {PROCESSED_DIR / 'eda_summary.json'}")

print("\nSELESAI. File siap diserahkan ke Orang 2:")
print(f"  -> {out_path}")