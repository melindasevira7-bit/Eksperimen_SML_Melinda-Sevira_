# Eksperimen_SML_Melinda-Sevira

Repository ini berisi eksperimen dan otomatisasi preprocessing dataset SmSA (Shopee Multi-domain Sentiment Analysis) untuk tugas klasifikasi sentimen Bahasa Indonesia.

## Struktur Repository

```
Eksperimen_SML_Melinda-Sevira/
├── .github/
│   └── workflows/
│       └── preprocessing.yml       # GitHub Actions workflow
├── smsa_raw/                        # Raw dataset (di-generate otomatis)
│   └── smsa_raw.csv
├── preprocessing/
│   ├── Eksperimen_Melinda-Sevira.ipynb   # Notebook eksperimen
│   ├── automate_Melinda-Sevira.py        # Script otomatisasi
│   └── smsa_preprocessing/               # Output preprocessing
│       └── smsa_preprocessed.csv
└── requirements.txt
```

## Dataset

**SmSA - Shopee Multi-domain Sentiment Analysis** dari [IndoNLU](https://huggingface.co/datasets/indonlu)

- **Task:** Klasifikasi Sentimen (Positif, Negatif, Netral)
- **Bahasa:** Indonesia
- **Sumber:** Review produk Shopee

## Tahapan Preprocessing

1. **Data Loading** — Memuat dataset dari HuggingFace
2. **EDA** — Analisis distribusi label, panjang teks, frekuensi kata
3. **Text Cleaning** — Lowercase, hapus URL/mention/angka/tanda baca
4. **Tokenisasi** — Memecah teks menjadi token
5. **Stopword Removal** — Hapus kata tidak bermakna (Bahasa Indonesia)
6. **Label Encoding** — Konversi label ke numerik
7. **Penyimpanan** — Output disimpan ke `smsa_preprocessing/smsa_preprocessed.csv`

## Cara Menjalankan

### Manual (Notebook)
Buka dan jalankan `preprocessing/Eksperimen_Melinda-Sevira.ipynb`

### Otomatis (Script)
```bash
pip install -r requirements.txt
cd preprocessing
python automate_Melinda-Sevira.py
```

### GitHub Actions
Workflow akan otomatis berjalan ketika:
- Ada push ke branch `main` yang mengubah file di `smsa_raw/` atau `automate_Melinda-Sevira.py`
- Trigger manual melalui tab **Actions** di GitHub
