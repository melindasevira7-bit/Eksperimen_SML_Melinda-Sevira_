"""
automate_Melinda-Sevira.py
==========================
Script otomatisasi preprocessing dataset hate speech & abusive language
(Indonesian Twitter Text) untuk klasifikasi teks Bahasa Indonesia.

Cara penggunaan:
    python automate_Melinda-Sevira.py

Output:
    smsa_preprocessing/smsa_preprocessed.csv
"""

import pandas as pd
import numpy as np
import re
import string
import os
import argparse
import logging

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ─── Download NLTK Resources ──────────────────────────────────────────────────
def download_nltk_resources():
    """Download resource NLTK yang diperlukan."""
    logger.info("Mengunduh resource NLTK...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    logger.info("Resource NLTK berhasil diunduh.")


# ─── Load Dataset ─────────────────────────────────────────────────────────────
def load_data(raw_dir: str = '../smsa_raw', save_raw: bool = True) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV lokal.

    Args:
        raw_dir: Direktori yang berisi file data_raw.csv
        save_raw: Jika True, simpan salinan raw dataset.

    Returns:
        DataFrame hasil loading.
    """
    raw_path = os.path.join(raw_dir, 'data.csv')
    logger.info(f"Memuat dataset dari: {raw_path}")

    df = pd.read_csv(raw_path)
    logger.info(f"Dataset berhasil dimuat. Shape: {df.shape}")
    logger.info(f"Kolom: {list(df.columns)}")
    logger.info(f"Distribusi label:\n{df.iloc[:, -1].value_counts().to_string()}")

    return df


# ─── Text Cleaning ────────────────────────────────────────────────────────────
def build_stopwords() -> set:
    """Membangun set stopwords Bahasa Indonesia + kata custom."""
    stop_words = set(stopwords.words('indonesian'))
    custom_stopwords = {
        'yg', 'nya', 'ini', 'itu', 'dan', 'di', 'ke', 'dari',
        'utk', 'tdk', 'gak', 'ga', 'ya', 'aja', 'jg', 'sy',
        'dgn', 'dg', 'tp', 'tapi', 'sdh', 'udah', 'dah', 'jd',
        'rt', 'amp', 'via'
    }
    stop_words.update(custom_stopwords)
    return stop_words


def clean_text(text: str) -> str:
    """Membersihkan teks dari noise."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_remove_stopwords(text: str, stop_words: set) -> str:
    """Tokenisasi teks dan hapus stopwords."""
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)


def preprocess_text(text: str, stop_words: set) -> str:
    """Pipeline preprocessing lengkap untuk satu teks."""
    text = clean_text(text)
    text = tokenize_and_remove_stopwords(text, stop_words)
    return text


# ─── Full Preprocessing Pipeline ─────────────────────────────────────────────
def run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Menjalankan full preprocessing pipeline pada DataFrame."""
    logger.info("Memulai preprocessing...")

    # Identifikasi kolom teks dan label
    text_col = 'Tweet' if 'Tweet' in df.columns else df.columns[0]
    label_col = 'HS' if 'HS' in df.columns else df.columns[-1]

    logger.info(f"Kolom teks: {text_col}, Kolom label: {label_col}")

    # Build stopwords
    stop_words = build_stopwords()
    logger.info(f"Total stopwords: {len(stop_words)}")

    # Clean & preprocess teks
    logger.info("Melakukan cleaning dan tokenisasi teks...")
    df['text_clean'] = df[text_col].apply(lambda x: preprocess_text(x, stop_words))

    # Rename kolom untuk konsistensi
    df = df.rename(columns={text_col: 'text', label_col: 'label'})

    # Hapus baris dengan teks kosong
    before = len(df)
    df = df[df['text_clean'].str.strip() != '']
    df = df.dropna(subset=['text_clean'])
    after = len(df)
    logger.info(f"Baris dihapus (teks kosong): {before - after}")

    # Label encoding
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'].astype(str))
    logger.info(f"Label classes: {list(le.classes_)}")
    logger.info(f"Preprocessing selesai. Total data: {len(df)}")

    return df


# ─── Save Output ──────────────────────────────────────────────────────────────
def save_preprocessed(df: pd.DataFrame, output_dir: str = 'smsa_preprocessing'):
    """Menyimpan hasil preprocessing ke CSV."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'smsa_preprocessed.csv')

    cols = ['text', 'text_clean', 'label', 'label_encoded']
    available_cols = [c for c in cols if c in df.columns]
    df[available_cols].to_csv(output_path, index=False)

    logger.info(f"Dataset preprocessing disimpan ke: {output_path}")
    logger.info(f"Shape: {df[available_cols].shape}")
    logger.info(f"Distribusi label:\n{df['label'].value_counts().to_string()}")

    return output_path


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Automate preprocessing dataset')
    parser.add_argument('--output_dir', type=str, default='smsa_preprocessing',
                        help='Direktori output hasil preprocessing')
    parser.add_argument('--raw_dir', type=str, default='../smsa_raw',
                        help='Direktori yang berisi raw dataset')
    parser.add_argument('--no_save_raw', action='store_true',
                        help='Tidak menyimpan raw dataset')
    args = parser.parse_args()

    download_nltk_resources()
    df = load_data(raw_dir=args.raw_dir, save_raw=not args.no_save_raw)
    df_processed = run_preprocessing(df)
    output_path = save_preprocessed(df_processed, output_dir=args.output_dir)

    logger.info("=" * 50)
    logger.info("PREPROCESSING SELESAI!")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 50)

    return df_processed


if __name__ == '__main__':
    main()
