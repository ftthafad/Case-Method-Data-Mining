"""Menjalankan seluruh alur analisis berurutan (langkah 1-4).

Pemakaian (dari folder repo ini):
    python run_all.py                                   # memakai data/online_retail_II.xlsx
    python run_all.py --input path/ke/online_retail_II.xlsx

Keluaran ada di folder hasil/. Parameter utama (sama dengan laporan):
referensi 10 Desember 2011, log pada F dan M, c = 2..8, m = 2, eps = 1e-5,
maks 300 iterasi, 20 inisialisasi, seed 42, pilihan akhir c = 3, ambang perbatasan 0,6.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KODE = ROOT / "kode"
HASIL = ROOT / "hasil"
NAMA = "Pelanggan Prioritas;Pelanggan Reguler;Pelanggan Tidak Aktif"


def jalan(*args):
    cmd = [sys.executable, *map(str, args)]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(ROOT / "data" / "online_retail_II.xlsx"))
    p.add_argument("--c", type=int, default=3, help="jumlah klaster terpilih (default 3)")
    a = p.parse_args()
    if not Path(a.input).exists():
        sys.exit(f"Dataset tidak ditemukan: {a.input}\nLihat data/README.md untuk cara mengunduhnya.")

    pre, fcm = HASIL / "1_preprocessing", HASIL / "2_fcm"
    jalan(KODE / "preprocessing_rfm.py", "--input", a.input, "--outdir", pre,
          "--ref-date", "2011-12-10")
    jalan(KODE / "langkah4_fcm.py", "--input", pre / "rfm_transformasi.csv",
          "--outdir", fcm, "--sens-c", a.c)
    # langkah 5 dan 6 membaca rfm_transformasi.csv dari folder yang sama dengan keluaran FCM
    shutil.copy(pre / "rfm_transformasi.csv", fcm / "rfm_transformasi.csv")
    jalan(KODE / "langkah5_evaluasi.py", "--data", fcm, "--outdir", HASIL / "3_evaluasi")
    jalan(KODE / "langkah6_interpretasi.py", "--data", fcm, "--outdir",
          HASIL / "4_interpretasi", "--c", a.c, "--names", NAMA, "--ambang", 0.6)
    print("\nSelesai. Hasil ada di:", HASIL)


if __name__ == "__main__":
    main()
