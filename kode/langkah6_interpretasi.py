#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LANGKAH 6 - Interpretasi klaster: mengubah "klaster 1, 2, 3" menjadi segmen pelanggan.

Input (folder data/, hasil langkah 3 dan 4):
  rfm_transformasi.csv   -> R, F, M dalam satuan asli + z_R, z_F, z_M (skala yang dipakai FCM)
  anggota_c3.csv         -> keanggotaan tiap pelanggan + klaster dominan (c = 3)
Output (folder hasil_langkah6/):
  profil_segmen.csv            jumlah, median & rata-rata R/F/M, % kontribusi belanja, % perbatasan
  pelanggan_segmen.csv         tiap pelanggan: segmen, keanggotaan, status perbatasan
  perbatasan_per_pasangan.csv  jumlah pelanggan perbatasan per pasangan segmen
  profil_segmen.png            batang median R, F, M per segmen
  faktor_pembeda.csv           eta-kuadrat tiap variabel (seberapa kuat memisahkan segmen)

Definisi:
  Nilai tipikal segmen   = median (data RFM miring ke kanan, rata-rata tertarik nilai ekstrem)
  % kontribusi belanja   = jumlah M anggota segmen / jumlah M seluruh pelanggan x 100%
  Pelanggan perbatasan   = keanggotaan maksimum < ambang (default 0,6);
                           pasangannya = dua segmen dengan keanggotaan tertinggi
  Eta-kuadrat            = variasi antar segmen / variasi total, pada z_R, z_F, z_M (0 sampai 1);
                           makin tinggi, makin kuat variabel itu memisahkan segmen

Pemakaian:
    python langkah6_interpretasi.py
    python langkah6_interpretasi.py --names "Prioritas;Reguler;Tidak Aktif" --ambang 0.6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_NAMES = ["Pelanggan Prioritas", "Pelanggan Reguler", "Pelanggan Tidak Aktif"]
Z = ["z_R", "z_F", "z_M"]
COLORS = ["#4C78A8", "#F58518", "#9D9D9D", "#54A24B", "#B279A2", "#E45756", "#9D755D", "#17BECF"]


# ----------------------------------------------------------------------------
# Pembacaan dan penggabungan
# ----------------------------------------------------------------------------
def load_data(data_dir, c):
    rfm = pd.read_csv(Path(data_dir) / "rfm_transformasi.csv")
    mem = pd.read_csv(Path(data_dir) / f"anggota_c{c}.csv")
    ucols = [f"u_klaster{k + 1}" for k in range(c)]
    miss = [k for k in ["Customer ID", "R", "F", "M", *Z] if k not in rfm.columns] + \
           [k for k in ["Customer ID", "klaster_dominan", *ucols] if k not in mem.columns]
    if miss:
        sys.exit(f"Kolom tidak ditemukan: {miss}")
    df = rfm[["Customer ID", "R", "F", "M", *Z]].merge(mem[["Customer ID", "klaster_dominan", *ucols]],
                                                   on="Customer ID", how="inner", validate="1:1")
    if len(df) != len(rfm) or len(df) != len(mem):
        sys.exit("Customer ID di kedua file tidak sama persis.")
    return df, ucols


# ----------------------------------------------------------------------------
# Analisis
# ----------------------------------------------------------------------------
def annotate(df, ucols, names, ambang):
    """Tambah nama segmen, keanggotaan maksimum, status perbatasan, dan segmen terdekat kedua."""
    U = df[ucols].to_numpy()
    order = np.argsort(-U, axis=1)
    out = df.copy()
    out["segmen"] = out["klaster_dominan"]
    out["nama_segmen"] = out["segmen"].map(lambda k: names[k - 1])
    out["keanggotaan_maks"] = U.max(axis=1)
    out["perbatasan"] = np.where(out["keanggotaan_maks"] < ambang, "Ya", "Tidak")
    second = order[:, 1] + 1
    out["segmen_kedua"] = np.where(out["perbatasan"] == "Ya", second, 0)
    return out


def segment_profile(df, names):
    total_m = df["M"].sum()
    rows = []
    for k, name in enumerate(names, start=1):
        g = df[df["segmen"] == k]
        border = (g["perbatasan"] == "Ya").sum()
        rows.append({
            "segmen": k, "nama_segmen": name, "jumlah_pelanggan": len(g),
            "persen_pelanggan": 100 * len(g) / len(df),
            "median_R_hari": g["R"].median(), "median_F_faktur": g["F"].median(),
            "median_M_pound": g["M"].median(),
            "rata2_R_hari": g["R"].mean(), "rata2_F_faktur": g["F"].mean(),
            "rata2_M_pound": g["M"].mean(),
            "total_belanja_pound": g["M"].sum(),
            "persen_kontribusi_belanja": 100 * g["M"].sum() / total_m,
            "jumlah_perbatasan": int(border),
            "persen_perbatasan_dalam_segmen": 100 * border / len(g),
        })
    return pd.DataFrame(rows)


def border_pairs(df, names):
    b = df[df["perbatasan"] == "Ya"]
    if b.empty:
        return pd.DataFrame(columns=["pasangan_segmen", "jumlah", "persen_dari_perbatasan",
                                     "persen_dari_seluruh_pelanggan"])
    lo = np.minimum(b["segmen"], b["segmen_kedua"])
    hi = np.maximum(b["segmen"], b["segmen_kedua"])
    cnt = pd.Series(list(zip(lo, hi))).value_counts()
    rows = [{"pasangan_segmen": f"{names[a - 1]} <-> {names[z - 1]}", "jumlah": int(n),
             "persen_dari_perbatasan": 100 * n / len(b),
             "persen_dari_seluruh_pelanggan": 100 * n / len(df)}
            for (a, z), n in cnt.items()]
    return pd.DataFrame(rows)


def eta_squared(values, labels):
    """Porsi variasi `values` yang dijelaskan oleh pembagian segmen (0..1)."""
    values = np.asarray(values, float)
    grand = values.mean()
    ss_total = ((values - grand) ** 2).sum()
    ss_between = sum(len(values[labels == k]) * (values[labels == k].mean() - grand) ** 2
                     for k in np.unique(labels))
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def discriminating_factors(df, names):
    """Eta-kuadrat R, F, M (skala z) dan rata-rata z tiap segmen, diurutkan dari yang terkuat."""
    labels = df["segmen"].to_numpy()
    rows = []
    for var, col in zip(["R", "F", "M"], Z):
        row = {"variabel": var, "eta_kuadrat": eta_squared(df[col], labels)}
        for k, name in enumerate(names, start=1):
            row[f"rata2_z_{name.replace(' ', '_')}"] = df.loc[df["segmen"] == k, col].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("eta_kuadrat", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Gambar
# ----------------------------------------------------------------------------
def fmt_id(x, _=None):
    """Format angka gaya Indonesia untuk sumbu grafik: 1.234,5 (titik ribuan, koma desimal)."""
    s = f"{x:,.6g}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def plot_profile(prof, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    spec = [("median_R_hari", "Recency: hari sejak belanja terakhir\n(makin kecil = makin baru)", "{:.0f}"),
            ("median_F_faktur", "Frequency: jumlah faktur\n(makin besar = makin sering)", "{:.0f}"),
            ("median_M_pound", "Monetary: total belanja (£)\n(makin besar = makin bernilai)", "{:,.0f}")]
    c = len(prof)
    labels = [f"{r.nama_segmen.replace(' ', chr(10), 1)}\n(n={r.jumlah_pelanggan:,})".replace(",", ".")
              for r in prof.itertuples()]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 4.4))
    for ax, (col, title, fmt) in zip(axes, spec):
        y = prof[col].to_numpy()
        ax.bar(np.arange(c), y, color=COLORS[:c], edgecolor="black", width=0.6)
        for i, v in enumerate(y):
            ax.annotate(fmt.format(v).replace(",", "."), (i, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontweight="bold")
        ax.set_xticks(np.arange(c))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=8.5)
        ax.margins(y=0.15)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_id))
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    fig.suptitle("Profil segmen: nilai median R, F, M (satuan asli)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Program utama
# ----------------------------------------------------------------------------
def run(data_dir, outdir, c=3, names=None, ambang=0.6):
    names = names or (DEFAULT_NAMES if c == 3 else [f"Segmen {k}" for k in range(1, c + 1)])
    if len(names) != c:
        sys.exit(f"Jumlah nama ({len(names)}) harus sama dengan c ({c}).")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    df, ucols = load_data(data_dir, c)
    df = annotate(df, ucols, names, ambang)
    prof = segment_profile(df, names)
    pairs = border_pairs(df, names)
    factors = discriminating_factors(df, names)

    cust = df.rename(columns={u: f"u_{names[i].replace(' ', '_')}" for i, u in enumerate(ucols)})
    cust = cust.drop(columns=["klaster_dominan", *Z])
    cust.to_csv(out / "pelanggan_segmen.csv", index=False, float_format="%.6f")
    prof.to_csv(out / "profil_segmen.csv", index=False, float_format="%.4f")
    pairs.to_csv(out / "perbatasan_per_pasangan.csv", index=False, float_format="%.4f")
    factors.to_csv(out / "faktor_pembeda.csv", index=False, float_format="%.4f")
    plot_profile(prof, out / "profil_segmen.png")

    show = prof[["nama_segmen", "jumlah_pelanggan", "persen_pelanggan", "median_R_hari",
                 "median_F_faktur", "median_M_pound", "persen_kontribusi_belanja"]]
    print(f"{len(df):,} pelanggan, c = {c}, ambang perbatasan = {ambang}\n")
    print(show.round(2).to_string(index=False))
    nb = int((df["perbatasan"] == "Ya").sum())
    print(f"\nPelanggan perbatasan: {nb:,} ({100 * nb / len(df):.2f}%)")
    if len(pairs):
        print(pairs.round(2).to_string(index=False))
    print("\nFaktor pembeda (eta-kuadrat, makin tinggi makin kuat memisahkan segmen):")
    print(factors[["variabel", "eta_kuadrat"]].round(4).to_string(index=False))
    print(f"\nHasil di: {out.resolve()}")


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default=str(here / "data"))
    p.add_argument("--outdir", default=str(here / "hasil_langkah6"))
    p.add_argument("--c", type=int, default=3, help="jumlah klaster (nama file anggota_c{c}.csv)")
    p.add_argument("--names", default=None, help='nama segmen dipisah titik koma: "a;b;c"')
    p.add_argument("--ambang", type=float, default=0.6, help="ambang keanggotaan maks untuk perbatasan")
    a = p.parse_args()
    names = [s.strip() for s in a.names.split(";")] if a.names else None
    run(a.data, a.outdir, a.c, names, a.ambang)


if __name__ == "__main__":
    main()
