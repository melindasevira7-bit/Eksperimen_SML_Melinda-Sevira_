"""
automate_Melinda-Sevira.py
==========================
Script otomatisasi preprocessing dataset SmSA (Shopee Multi-domain Sentiment Analysis)
untuk klasifikasi sentimen Bahasa Indonesia.

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
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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
def load_data(save_raw: bool = True, raw_dir: str = '../smsa_raw') -> pd.DataFrame:
    """
    Memuat dataset SmSA dari HuggingFace dan menggabungkan semua split.

    Args:
        save_raw: Jika True, simpan raw dataset ke CSV.
        raw_dir: Direktori penyimpanan raw dataset.

    Returns:
        DataFrame gabungan semua split.
    """
    logger.info("Memuat dataset SmSA dari HuggingFace...")
    dataset = load_dataset('indonlu', 'smsa')

    df_train = pd.DataFrame(dataset['train'])
    df_val   = pd.DataFrame(dataset['validation'])
    df_test  = pd.DataFrame(dataset['test'])

    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    logger.info(f"Dataset berhasil dimuat. Total baris: {len(df)}")

    if save_raw:
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, 'smsa_raw.csv')
        df.to_csv(raw_path, index=False)
        logger.info(f"Raw dataset disimpan ke: {raw_path}")

    return df


# ─── Text Cleaning ────────────────────────────────────────────────────────────
def build_stopwords() -> set:
    """Membangun set stopwords Bahasa Indonesia + kata custom."""
    stop_words = set(stopwords.words('indonesian'))
    custom_stopwords = {
        'yg', 'nya', 'ini', 'itu', 'dan', 'di', 'ke', 'dari',
        'utk', 'tdk', 'gak', 'ga', 'ya', 'aja', 'jg', 'sy',
        'dgn', 'dg', 'tp', 'tapi', 'sdh', 'udah', 'dah', 'jd'
    }
    stop_words.update(custom_stopwords)
    return stop_words


def clean_text(text: str) -> str:
    """
    Membersihkan teks dari noise.

    Steps:
        1. Lowercase
        2. Hapus URL
        3. Hapus mention & hashtag
        4. Hapus angka
        5. Hapus tanda baca
        6. Hapus whitespace berlebih
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_remove_stopwords(text: str, stop_words: set) -> str:
    """
    Tokenisasi teks dan hapus stopwords.

    Args:
        text: Teks yang sudah di-clean.
        stop_words: Set stopwords.

    Returns:
        Teks yang sudah ditokenisasi dan bebas stopwords.
    """
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
    """
    Menjalankan full preprocessing pipeline pada DataFrame.

    Args:
        df: DataFrame raw dengan kolom 'text' dan 'label'.

    Returns:
        DataFrame hasil preprocessing.
    """
    logger.info("Memulai preprocessing...")

    # Mapping label
    label_map = {0: 'positif', 1: 'netral', 2: 'negatif'}
    df['sentiment'] = df['label'].map(label_map)

    # Build stopwords
    stop_words = build_stopwords()
    logger.info(f"Total stopwords: {len(stop_words)}")

    # Clean & preprocess teks
    logger.info("Melakukan cleaning dan tokenisasi teks...")
    df['text_clean'] = df['text'].apply(lambda x: preprocess_text(x, stop_words))

    # Hapus baris dengan teks kosong
    before = len(df)
    df = df[df['text_clean'].str.strip() != '']
    df = df.dropna(subset=['text_clean'])
    after = len(df)
    logger.info(f"Baris dihapus (teks kosong): {before - after}")

    # Label encoding
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['sentiment'])
    logger.info(f"Label classes: {list(le.classes_)}")

    logger.info(f"Preprocessing selesai. Total data: {len(df)}")
    return df


# ─── Save Output ──────────────────────────────────────────────────────────────
def save_preprocessed(df: pd.DataFrame, output_dir: str = 'smsa_preprocessing'):
    """
    Menyimpan hasil preprocessing ke CSV.

    Args:
        df: DataFrame hasil preprocessing.
        output_dir: Direktori output.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'smsa_preprocessed.csv')

    cols = ['text', 'text_clean', 'sentiment', 'label_encoded']
    df[cols].to_csv(output_path, index=False)
    logger.info(f"Dataset preprocessing disimpan ke: {output_path}")

    # Tampilkan statistik
    logger.info(f"Shape: {df[cols].shape}")
    logger.info(f"Distribusi label:\n{df['sentiment'].value_counts().to_string()}")

    return output_path


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Automate preprocessing SmSA dataset')
    parser.add_argument('--output_dir', type=str, default='smsa_preprocessing',
                        help='Direktori output hasil preprocessing')
    parser.add_argument('--raw_dir', type=str, default='../smsa_raw',
                        help='Direktori penyimpanan raw dataset')
    parser.add_argument('--no_save_raw', action='store_true',
                        help='Tidak menyimpan raw dataset')
    args = parser.parse_args()

    # Step 1: Download NLTK resources
    download_nltk_resources()

    # Step 2: Load data
    df = load_data(save_raw=not args.no_save_raw, raw_dir=args.raw_dir)

    # Step 3: Preprocessing
    df_processed = run_preprocessing(df)

    # Step 4: Save
    output_path = save_preprocessed(df_processed, output_dir=args.output_dir)

    logger.info("=" * 50)
    logger.info("PREPROCESSING SELESAI!")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 50)

    return df_processed


if __name__ == '__main__':
    main()
