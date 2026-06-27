# ⚡ EnergyIQ — Smart Building Energy Dashboard

Dashboard interaktif untuk eksplorasi data dan prediksi konsumsi energi 20 gedung pintar (komersial, industri, edukasi, residensial), menggunakan model **Random Forest Regressor** (R² ≈ 0.77, MAPE ≈ 8.97%).

**Fitur dashboard:**
- 🏠 **Ringkasan** — KPI total konsumsi, tren harian, distribusi tipe gedung, peringkat gedung
- 📊 **Eksplorasi Data** — filter interaktif (gedung, tipe, tanggal), pola waktu, hubungan cuaca vs energi, korelasi fitur
- 🔮 **Prediksi Energi** — form simulasi untuk memprediksi konsumsi kWh & estimasi biaya, lengkap dengan skenario cepat
- 🧠 **Performa Model** — metrik evaluasi, feature importance, spesifikasi model

---

## 📁 Struktur File

Pastikan struktur folder berikut **sebelum** push ke GitHub:

```
energyiq_dashboard/
├── app.py                  ← aplikasi utama Streamlit
├── requirements.txt        ← daftar dependensi Python
├── processed_data.csv      ← dataset
├── rf_model.pkl            ← model Random Forest terlatih
├── le_building.pkl         ← label encoder Building_ID
├── le_occupancy.pkl        ← label encoder Occupancy_Level
├── le_type.pkl             ← label encoder Building_Type
├── model_metadata.json     ← metadata model (metrik, daftar fitur, kelas)
├── .streamlit/
│   └── config.toml         ← tema dashboard
└── .gitignore
```

> ⚠️ **Penting:** kelima file artefak (`processed_data.csv`, `rf_model.pkl`, `le_building.pkl`, `le_occupancy.pkl`, `le_type.pkl`, `model_metadata.json`) **harus** berada di folder root yang sama dengan `app.py` — jangan dimasukkan ke subfolder, karena `app.py` membacanya dengan path relatif.

File `rf_model.pkl` berukuran ±50 MB. GitHub mengizinkan file hingga 100 MB tanpa Git LFS, jadi aman untuk di-push langsung.

---

## 🚀 Cara Deploy (GitHub → Streamlit Cloud)

### 1. Siapkan folder project
Kumpulkan semua file di atas ke satu folder lokal di komputer Anda, misalnya `energyiq_dashboard/`.

### 2. Push ke GitHub
Buka terminal di folder tersebut, lalu jalankan:

```bash
git init
git add .
git commit -m "Initial commit: EnergyIQ dashboard"
git branch -M main
git remote add origin https://github.com/<username-anda>/<nama-repo>.git
git push -u origin main
```

Ganti `<username-anda>` dan `<nama-repo>` sesuai akun & nama repository GitHub Anda. Jika repository belum dibuat, buat dulu lewat [github.com/new](https://github.com/new) (pilih **Public**, tanpa README/gitignore otomatis agar tidak konflik).

### 3. Deploy ke Streamlit Community Cloud
1. Buka **[share.streamlit.io](https://share.streamlit.io)** dan login dengan akun GitHub.
2. Klik **"New app"**.
3. Pilih repository, branch (`main`), dan **Main file path** isi dengan `app.py`.
4. Klik **"Deploy"**. Tunggu 1–3 menit hingga proses build selesai.
5. Dashboard akan otomatis tersedia di URL seperti:
   `https://<nama-repo>-<random>.streamlit.app`

### 4. Update dashboard di kemudian hari
Setiap kali Anda push perubahan baru ke branch `main` di GitHub, Streamlit Cloud akan **otomatis redeploy** — tidak perlu langkah manual tambahan.

```bash
git add .
git commit -m "Update dashboard"
git push
```

---

## 💻 Menjalankan Secara Lokal (opsional, untuk testing sebelum deploy)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard akan terbuka otomatis di `http://localhost:8501`.

---

## 🧠 Tentang Model

| Item | Detail |
|---|---|
| Algoritma | Random Forest Regressor (300 trees, max_depth=15) |
| Fitur | 14 fitur (waktu, cuaca, hunian, tipe gedung, ID gedung) |
| Strategi split | Kronologis 80% train / 20% test (anti data-leakage) |
| RMSE | 21.62 kWh |
| MAE | 17.41 kWh |
| R² | 0.7735 |
| MAPE | 8.97% |

Detail lengkap proses training, EDA, dan evaluasi ada di notebook `EnergyIQ_SmartBuilding_EAS.ipynb`.

---

## 🛠️ Troubleshooting

- **Error "FileNotFoundError" saat deploy** → pastikan kelima file artefak ada di root repo, bukan di subfolder.
- **Build gagal karena versi library** → cek versi di `requirements.txt`, sesuaikan dengan versi `scikit-learn` yang dipakai saat training (1.6.1) agar `.pkl` kompatibel.
- **Dashboard lambat saat load pertama** → normal, karena Streamlit Cloud melakukan cold start. Setelah itu akan lebih cepat berkat `@st.cache_data` dan `@st.cache_resource`.
