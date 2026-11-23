# Segmentasi Pelanggan dengan RFM dan Fuzzy C-Means

Kode dan hasil analisis untuk laporan *Case Method* Data Mining: segmentasi pelanggan toko daring
grosir perlengkapan hadiah memakai analisis RFM (Recency, Frequency, Monetary) dan algoritma
Fuzzy C-Means (FCM) pada dataset Online Retail II (UCI).

Penulis: Dhiya Najam Fattah Afad (L0124048)

## Ringkasan hasil

- 5.852 pelanggan setelah pembersihan (776.596 baris transaksi dari 1.067.371 baris mentah).
- Jumlah klaster c = 3 dipilih (kandidat c = 2 dan c = 3 menurut peringkat rata-rata MPC, Xie-Beni, Silhouette), fuzzifier m = 2.
- Tiga segmen: Pelanggan Prioritas (31,1% pelanggan, 84,3% belanja), Pelanggan Reguler (38,9%, 10,7%), Pelanggan Tidak Aktif (30,0%, 5,0%).
- 1.197 pelanggan (20,5%) berada di perbatasan antarsegmen (keanggotaan terbesar < 0,6).

## Struktur folder

| Folder / file | Isi |
|---|---|
| `kode/preprocessing_rfm.py` | Langkah 1: pembersihan data, pembentukan R, F, M, transformasi log (F, M) dan standardisasi z-score |
| `kode/langkah4_fcm.py` | Langkah 2: FCM untuk c = 2..8 (implementasi sendiri), uji sensitivitas m |
| `kode/langkah5_evaluasi.py` | Langkah 3: indeks MPC, Xie-Beni, Silhouette dan peringkat |
| `kode/langkah6_interpretasi.py` | Langkah 4: profil segmen, kontribusi belanja, pelanggan perbatasan, eta-kuadrat |
| `kode/test_preprocessing_rfm.py` | Tes otomatis untuk fungsi preprocessing |
| `run_all.py` | Menjalankan langkah 1-4 berurutan |
| `hasil/1_preprocessing/` | Log pembersihan, RFM mentah dan tertransformasi, grafik sebaran (Gambar 4.1) |
| `hasil/2_fcm/` | Pusat klaster, keanggotaan, parameter z, sebaran klaster, sensitivitas m |
| `hasil/3_evaluasi/` | Nilai dan peringkat indeks validitas, grafik evaluasi (Gambar 4.2) |
| `hasil/4_interpretasi/` | Profil segmen, pelanggan per segmen, perbatasan, faktor pembeda (Gambar 4.4) |
| `data/` | Petunjuk mengunduh dataset (file mentah tidak disertakan) |

Sebaran klaster c = 3 (Gambar 4.3) ada di `hasil/2_fcm/sebaran_c3.png`.

## Cara menjalankan

Butuh Python 3.9+.

```bash
pip install -r requirements.txt
# unduh dataset sesuai data/README.md ke data/online_retail_II.xlsx
python run_all.py
```

Waktu proses sekitar 1-2 menit. Hasil ditulis ulang ke `hasil/`. Setiap skrip juga bisa dijalankan
sendiri (`python kode/<skrip>.py --help`). Tes: `pytest kode/`.

## Parameter

Tanggal acuan 10 Desember 2011; log natural pada F dan M (R tidak ditransformasi); z-score dengan
simpangan baku populasi; c = 2..8; m = 2 (uji 1,5 dan 2,5); ambang konvergensi 1e-5; maksimum 300
iterasi; 20 inisialisasi acak (seed 42); ambang pelanggan perbatasan 0,6.

## Sumber data

Chen, D. (2019). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/502/online+retail+ii
