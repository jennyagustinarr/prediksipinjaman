# Loan Approval Prediction Dashboard

Dashboard ini dibuat untuk praktikum machine learning dengan tema hijau dan navigasi sidebar seperti contoh dashboard SVM. Model utama yang digunakan adalah Random Forest Classifier untuk memprediksi status pengajuan pinjaman, yaitu `Approved` atau `Rejected`.

## Struktur Project

```text
loan-approval-prediction/
├── app.py
├── loan_approval_dataset.csv
├── random_forest_loan_model.pkl
├── model_evaluation_results.csv
├── feature_importance_results.csv
├── requirements.txt
├── README.md
├── notebook/
│   └── loan_approval_training.ipynb
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## Fitur Dashboard

1. **Ringkasan**
   - Total data
   - Jumlah status Approved
   - Jumlah status Rejected
   - Akurasi Random Forest
   - Grafik distribusi status pinjaman
   - Grafik perbandingan evaluasi model

2. **Prediksi**
   - Form input data pemohon
   - Input profil pemohon, pendapatan, pinjaman, CIBIL score, dan aset
   - Hasil prediksi Approved atau Rejected
   - Probabilitas Approved dan Rejected
   - Catatan interpretasi sederhana dari hasil prediksi

3. **Visualisasi**
   - Korelasi fitur numerik terhadap status Approved
   - Feature importance Random Forest
   - Outlier per fitur numerik
   - Distribusi CIBIL score
   - Scatter plot CIBIL score terhadap jumlah pinjaman

4. **Evaluasi Model**
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion matrix
   - Classification report
   - Perbandingan model berdasarkan file evaluasi

5. **Dataset**
   - Preview 15 baris pertama dataset
   - Informasi jumlah data dan jumlah kolom

## Cara Menjalankan

Masuk ke folder project:

```bash
cd loan-approval-prediction
```

Buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan virtual environment di Windows Command Prompt:

```bash
.venv\Scripts\activate.bat
```

Atau aktifkan di PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install semua library:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
python app.py
```

Buka browser:

```text
http://127.0.0.1:5000
```

## Catatan

Jika file `random_forest_loan_model.pkl` tidak kompatibel dengan versi `scikit-learn` di laptop, sistem akan melatih ulang model dari `loan_approval_dataset.csv`, lalu menyimpan ulang model ke file pkl.
