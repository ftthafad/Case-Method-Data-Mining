#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing RFM untuk dataset Online Retail II (UCI) - tugas Case Method
Data Mining, algoritma Fuzzy C-Means.

Alur (setiap tahap dicatat jumlah barisnya di log_pembersihan.csv):

  1. Baca semua sheet / file CSV  ->  gabungkan
  2. Buang duplikat identik (termasuk 22.523 baris yang tercatat dobel
     karena sheet 2009-2010 dan 2010-2011 tumpang tindih pada 1-9 Des 2010)
  3. Buang baris tanpa Customer ID
  4. Buang pembatalan (Invoice berawalan "C")
  5. Buang Quantity <= 0
  6. Buang Price <= 0
  7. Buang kode non-produk (ongkir, biaya admin, penyesuaian, tes, dll.)
  8. Agregasi per pelanggan -> R, F, M
  9. Transformasi logaritma pada F dan M (R tidak, lihat --log-vars) +
     standarisasi z-score

Cara pakai:
    python preprocessing_rfm.py --input online_retail_II.xlsx --outdir hasil

Keluaran di folder --outdir:
    transaksi_bersih.csv.gz    transaksi valid (level baris item)
    rfm_mentah.csv             R, F, M per pelanggan (satuan asli)
    rfm_transformasi.csv       R, F, M + log + z-score (masukan FCM)
    log_pembersihan.csv        jumlah baris sisa tiap tahap
    ringkasan.json             parameter, skewness sebelum/sesudah, statistik
    grafik_sebaran_rfm.png     histogram sebelum vs sesudah transformasi
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Konfigurasi
# --------------------------------------------------------------------------
CANONICAL_COLUMNS = [
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "Price", "Customer ID", "Country",
]

# Nama kolom alternatif di berbagai sumber unduhan -> nama baku
COLUMN_ALIASES = {
    "InvoiceNo": "Invoice",
    "UnitPrice": "Price",
    "CustomerID": "Customer ID",
}

# Kode yang BUKAN pembelian barang. Daftar ini disusun dari pemeriksaan data
# (lihat README). Voucher hadiah (gift_0001_*) dan produk berawalan DCGS
# adalah penjualan nyata, jadi TIDAK dibuang.
NON_PRODUCT_CODES = {
    "POST",          # POSTAGE (ongkos kirim)
    "DOT",           # DOTCOM POSTAGE
    "C2",            # CARRIAGE (ongkos angkut)
    "C3",            # kode penyesuaian
    "M", "m",        # Manual (entri manual)
    "D",             # Discount
    "S",             # SAMPLES
    "BANK CHARGES",  # biaya bank
    "ADJUST", "ADJUST2",   # penyesuaian oleh staf
    "AMAZONFEE",     # biaya marketplace
    "B",             # Adjust bad debt
    "CRUK",          # komisi CRUK
    "TEST001", "TEST002",  # produk uji coba
}

RFM_COLUMNS = ["R", "F", "M"]


# --------------------------------------------------------------------------
# 1. Membaca data
# --------------------------------------------------------------------------
def load_raw(path: str | Path) -> pd.DataFrame:
    """Baca file .xlsx (semua sheet) atau .csv, seragamkan nama kolom."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        df = pd.concat(sheets.values(), ignore_index=True)
    elif path.suffix.lower() in {".csv", ".gz"}:
        df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    else:
        raise ValueError(f"Format file tidak didukung: {path.suffix}")

    df = df.rename(columns=COLUMN_ALIASES)
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")
    df = df[CANONICAL_COLUMNS].copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


# --------------------------------------------------------------------------
# 2. Pembersihan transaksi
# --------------------------------------------------------------------------
def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Terapkan filter berurutan. Kembalikan (transaksi_valid, log_tahap)."""
    log = []

    def record(name: str, frame: pd.DataFrame, previous: int | None) -> None:
        removed = 0 if previous is None else previous - len(frame)
        log.append({"tahap": name, "baris_dibuang": removed,
                    "baris_tersisa": len(frame)})

    cur = df
    record("0. Data mentah (semua sheet digabung)", cur, None)

    n = len(cur)
    cur = cur.drop_duplicates(subset=CANONICAL_COLUMNS)
    record("1. Buang duplikat identik", cur, n)

    n = len(cur)
    cur = cur[cur["Customer ID"].notna()]
    record("2. Buang baris tanpa Customer ID", cur, n)

    n = len(cur)
    cur = cur[~cur["Invoice"].str.startswith("C")]
    record("3. Buang pembatalan (Invoice berawalan C)", cur, n)

    n = len(cur)
    cur = cur[cur["Quantity"] > 0]
    record("4. Buang Quantity <= 0", cur, n)

    n = len(cur)
    cur = cur[cur["Price"] > 0]
    record("5. Buang Price <= 0", cur, n)

    n = len(cur)
    cur = cur[~cur["StockCode"].isin(NON_PRODUCT_CODES)]
    record("6. Buang kode non-produk", cur, n)

    cur = cur.copy()
    cur["Customer ID"] = cur["Customer ID"].astype(int)
    return cur.reset_index(drop=True), pd.DataFrame(log)


# --------------------------------------------------------------------------
# 3. Menghitung RFM
# --------------------------------------------------------------------------
def default_reference_date(raw: pd.DataFrame) -> pd.Timestamp:
    """Sehari setelah tanggal transaksi terakhir di seluruh dataset."""
    return raw["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)


def compute_rfm(clean: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """
    Agregasi per pelanggan.
      R = hari dari tanggal belanja terakhir sampai ref_date
      F = jumlah Invoice unik
      M = jumlah (Quantity x Price)
    """
    ref_date = pd.Timestamp(ref_date).normalize()
    work = clean.assign(
        LineValue=clean["Quantity"] * clean["Price"],
        DateOnly=clean["InvoiceDate"].dt.normalize(),
    )
    grp = work.groupby("Customer ID")
    rfm = pd.DataFrame({
        "R": (ref_date - grp["DateOnly"].max()).dt.days,
        "F": grp["Invoice"].nunique(),
        "M": grp["LineValue"].sum().round(2),
    })
    rfm.index.name = "Customer ID"
    return rfm.reset_index()


# --------------------------------------------------------------------------
# 4. Transformasi & standarisasi
# --------------------------------------------------------------------------
def transform_rfm(rfm: pd.DataFrame, log_vars: list[str]) -> pd.DataFrame:
    """
    Tambahkan kolom ln_<v> (untuk v di log_vars) dan z_<v> (z-score dari
    nilai yang dipakai: ln jika ditransformasi, nilai asli jika tidak).
    Syarat ln: nilai > 0 (R >= 1, F >= 1, M > 0 dijamin oleh pembersihan).
    """
    out = rfm.copy()
    for v in RFM_COLUMNS:
        if v in log_vars:
            if (out[v] <= 0).any():
                raise ValueError(f"{v} memuat nilai <= 0, ln tidak valid.")
            out[f"ln_{v}"] = np.log(out[v])
            base = out[f"ln_{v}"]
        else:
            base = out[v]
        std = base.std(ddof=0)
        out[f"z_{v}"] = 0.0 if std == 0 else (base - base.mean()) / std
    return out


def skewness_table(rfm: pd.DataFrame, transformed: pd.DataFrame,
                   log_vars: list[str]) -> dict:
    res = {}
    for v in RFM_COLUMNS:
        used = transformed[f"ln_{v}"] if v in log_vars else transformed[v]
        res[v] = {
            "skewness_sebelum": float(rfm[v].skew()),
            "skewness_sesudah": float(used.skew()),
            "ditransformasi_ln": v in log_vars,
        }
    return res


# --------------------------------------------------------------------------
# 5. Grafik
# --------------------------------------------------------------------------
def fmt_id(x, _=None):
    """Format angka gaya Indonesia untuk sumbu grafik: 1.234,5 (titik ribuan, koma desimal)."""
    s = f"{x:,.6g}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def plot_distributions(rfm: pd.DataFrame, transformed: pd.DataFrame,
                       log_vars: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    fig, axes = plt.subplots(2, 3, figsize=(9, 6.4))
    labels = {"R": "Recency (hari)", "F": "Frequency (faktur)",
              "M": "Monetary (£)"}
    for j, v in enumerate(RFM_COLUMNS):
        ax = axes[0, j]
        ax.hist(rfm[v], bins=50, color="#4C78A8")
        ax.set_title(f"{labels[v]} - sebelum  (skew {rfm[v].skew():.2f})".replace(".", ","),
                     fontsize=9)
        ax.set_ylabel("Jumlah pelanggan")
        if v == "M":   # sumbu-x Monetary: beberapa tanda saja, pemisah ribuan
            ax.xaxis.set_major_locator(MaxNLocator(4))
            pass
        used = transformed[f"ln_{v}"] if v in log_vars else transformed[v]
        ax = axes[1, j]
        ax.hist(used, bins=50, color="#59A14F")
        name = f"ln({v})" if v in log_vars else v
        ax.set_title(f"{name} - sesudah  (skew {used.skew():.2f})".replace(".", ","),
                     fontsize=9)
        ax.set_ylabel("Jumlah pelanggan")
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_id))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_id))
    fig.suptitle("Sebaran R, F, M sebelum dan sesudah transformasi",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Program utama
# --------------------------------------------------------------------------
def run(input_path: str, outdir: str, ref_date: str | None,
        log_vars: list[str]) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Membaca {input_path} ...")
    raw = load_raw(input_path)
    ref = pd.Timestamp(ref_date) if ref_date else default_reference_date(raw)
    print(f"      {len(raw):,} baris | periode "
          f"{raw['InvoiceDate'].min():%d %b %Y} - "
          f"{raw['InvoiceDate'].max():%d %b %Y} | tanggal acuan {ref:%d %b %Y}")

    print("[2/6] Membersihkan transaksi ...")
    clean, log = clean_transactions(raw)
    print(log.to_string(index=False))

    print("[3/6] Menghitung RFM per pelanggan ...")
    rfm = compute_rfm(clean, ref)
    print(f"      {len(rfm):,} pelanggan")

    print(f"[4/6] Transformasi ln untuk {log_vars} + standarisasi ...")
    transformed = transform_rfm(rfm, log_vars)
    skew = skewness_table(rfm, transformed, log_vars)
    for v, s in skew.items():
        print(f"      {v}: skewness {s['skewness_sebelum']:.2f} -> "
              f"{s['skewness_sesudah']:.2f}")

    print("[5/6] Menyimpan hasil ...")
    clean.to_csv(out / "transaksi_bersih.csv.gz", index=False)
    rfm.to_csv(out / "rfm_mentah.csv", index=False)
    transformed.to_csv(out / "rfm_transformasi.csv", index=False,
                       float_format="%.6f")
    log.to_csv(out / "log_pembersihan.csv", index=False)

    summary = {
        "input": str(input_path),
        "tanggal_acuan": f"{ref:%Y-%m-%d}",
        "periode_data": [f"{raw['InvoiceDate'].min():%Y-%m-%d}",
                         f"{raw['InvoiceDate'].max():%Y-%m-%d}"],
        "variabel_ln": log_vars,
        "baris_mentah": int(len(raw)),
        "baris_transaksi_valid": int(len(clean)),
        "jumlah_pelanggan": int(len(rfm)),
        "statistik_rfm_mentah": rfm[RFM_COLUMNS].describe(
            percentiles=[.25, .5, .75, .95, .99]).round(2).to_dict(),
        "skewness": skew,
        "kode_non_produk_dibuang": sorted(NON_PRODUCT_CODES),
    }
    with open(out / "ringkasan.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("[6/6] Membuat grafik ...")
    plot_distributions(rfm, transformed, log_vars,
                       out / "grafik_sebaran_rfm.png")
    print(f"Selesai. Hasil ada di: {out.resolve()}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", required=True,
                   help="Path ke online_retail_II.xlsx (atau CSV)")
    p.add_argument("--outdir", default="hasil", help="Folder keluaran")
    p.add_argument("--ref-date", default=None,
                   help="Tanggal acuan R (YYYY-MM-DD). Default: sehari "
                        "setelah transaksi terakhir.")
    p.add_argument("--log-vars", default="F,M",
                   help="Variabel yang di-ln, dipisah koma (default F,M). "
                        "R tidak di-ln karena skewness-nya moderat (0,89) dan "
                        "ln(R) justru berbalik miring ke kiri (-0,55). "
                        "Kosongkan ('') untuk tanpa transformasi.")
    args = p.parse_args()
    log_vars = [v.strip().upper() for v in args.log_vars.split(",") if v.strip()]
    bad = [v for v in log_vars if v not in RFM_COLUMNS]
    if bad:
        p.error(f"--log-vars hanya boleh berisi R, F, M (salah: {bad})")
    run(args.input, args.outdir, args.ref_date, log_vars)


if __name__ == "__main__":
    main()
