#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LANGKAH 5 - Evaluasi hasil FCM untuk memilih jumlah klaster c.

Tiga indeks yang dihitung untuk tiap c:
  MPC (tinggi = baik)         MPC = (c*PC - 1)/(c - 1),   PC = (1/n) sum_i sum_k u_ik^2
  Xie-Beni (rendah = baik)    XB  = sum_i sum_k u_ik^m ||x_i - v_k||^2 / (n * min_{k!=j} ||v_k - v_j||^2)
  Silhouette (tinggi = baik)  s_i = (b_i - a_i)/max(a_i, b_i), dirata-ratakan; label = klaster dominan
Peringkat tiap indeks (1 = terbaik) dirata-ratakan; dua c dengan rata-rata terbaik
ditandai sebagai KANDIDAT. Keputusan akhir di antara kandidat (stabilitas, ukuran klaster,
makna bisnis) dilakukan di laporan, bukan oleh program.

Input (folder data/, hasil langkah 4):
  rfm_transformasi.csv, anggota_c{c}.csv, centroid_c{c}.csv
Output (folder hasil_langkah5/):
  nilai_indeks_per_c.csv, peringkat_per_c.csv, evaluasi_indeks.png

Pemakaian:
    python langkah5_evaluasi.py                # m = 2 (harus sama dengan m di langkah 4)
    python langkah5_evaluasi.py --m 2
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["z_R", "z_F", "z_M"]


# ----------------------------------------------------------------------------
# Tiga indeks
# ----------------------------------------------------------------------------
def mpc(U):
    """Modified Partition Coefficient. 0 = paling kabur, 1 = paling tegas."""
    n, c = U.shape
    pc = (U ** 2).sum() / n
    return float((c * pc - 1) / (c - 1))


def xie_beni(X, U, V, m):
    """Xie-Beni: kerapatan dibagi keterpisahan antar centroid terdekat. Makin kecil makin baik."""
    D2 = ((X[:, None, :] - V[None, :, :]) ** 2).sum(axis=2)          # (n, c)
    num = ((U ** m) * D2).sum()
    c = len(V)
    sep = min(((V[a] - V[b]) ** 2).sum() for a in range(c) for b in range(a + 1, c))
    return float(num / (len(X) * sep))


def silhouette(X, labels, chunk=500):
    """Rata-rata silhouette (jarak Euclidean). Klaster beranggota 1 diberi s = 0.
    Dihitung bertahap per blok baris agar hemat memori."""
    n = len(X)
    ks = np.unique(labels)
    onehot = (labels[:, None] == ks[None, :]).astype(float)           # (n, c)
    size = onehot.sum(axis=0)
    own = np.searchsorted(ks, labels)
    sq = (X ** 2).sum(axis=1)
    s = np.zeros(n)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        d2 = sq[lo:hi, None] + sq[None, :] - 2 * X[lo:hi] @ X.T
        D = np.sqrt(np.maximum(d2, 0.0))
        sums = D @ onehot                                              # jumlah jarak ke tiap klaster
        idx = np.arange(hi - lo)
        o = own[lo:hi]
        n_own = size[o]
        a = np.where(n_own > 1, sums[idx, o] / np.maximum(n_own - 1, 1), 0.0)
        mean_other = sums / size[None, :]
        mean_other[idx, o] = np.inf
        b = mean_other.min(axis=1)
        s[lo:hi] = np.where(n_own > 1, (b - a) / np.maximum(a, b), 0.0)
    return float(s.mean())


# ----------------------------------------------------------------------------
# Pembacaan, peringkat, gambar
# ----------------------------------------------------------------------------
def available_c(data_dir):
    cs = sorted(int(re.search(r"anggota_c(\d+)\.csv", p.name).group(1))
                for p in data_dir.glob("anggota_c*.csv"))
    if len(cs) < 2:
        sys.exit(f"Butuh minimal dua file anggota_c*.csv di {data_dir}")
    return cs


def load_all(data_dir, cs):
    df = pd.read_csv(data_dir / "rfm_transformasi.csv")
    miss = [k for k in ["Customer ID", *FEATURES] if k not in df.columns]
    if miss:
        sys.exit(f"Kolom tidak ditemukan di rfm_transformasi.csv: {miss}")
    X = df[FEATURES].to_numpy(float)
    out = {}
    for c in cs:
        mem = pd.read_csv(data_dir / f"anggota_c{c}.csv")
        cen = pd.read_csv(data_dir / f"centroid_c{c}.csv")
        if not (mem["Customer ID"].values == df["Customer ID"].values).all():
            sys.exit(f"Urutan Customer ID di anggota_c{c}.csv berbeda dari rfm_transformasi.csv")
        U = mem[[f"u_klaster{k + 1}" for k in range(c)]].to_numpy(float)
        V = cen[FEATURES].to_numpy(float)
        if V.shape[0] != c:
            sys.exit(f"centroid_c{c}.csv harus berisi {c} baris")
        out[c] = (U, V, mem["klaster_dominan"].to_numpy() - 1)
    return X, out


def make_rank(values):
    r = pd.DataFrame({"c": values["c"]})
    r["peringkat_MPC"] = values["MPC"].rank(ascending=False, method="min").astype(int)
    r["peringkat_XB"] = values["Xie_Beni"].rank(ascending=True, method="min").astype(int)
    r["peringkat_Silhouette"] = values["Silhouette"].rank(ascending=False, method="min").astype(int)
    r["peringkat_rata2"] = r[["peringkat_MPC", "peringkat_XB", "peringkat_Silhouette"]].mean(axis=1)
    top = r["peringkat_rata2"].nsmallest(2).index
    r["kandidat"] = np.where(r.index.isin(top), "Ya", "")
    return r.sort_values("c").reset_index(drop=True)


def fmt_id(x, _=None):
    """Format angka gaya Indonesia untuk sumbu grafik: 1.234,5 (titik ribuan, koma desimal)."""
    s = f"{x:,.6g}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def plot_indices(values, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    spec = [("MPC", "MPC (↑ makin baik)", True, "#4C78A8"),
            ("Xie_Beni", "Xie-Beni (↓ makin baik)", False, "#E45756"),
            ("Silhouette", "Silhouette (↑ makin baik)", True, "#54A24B")]
    cs = values["c"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    for ax, (col, title, higher, color) in zip(axes, spec):
        y = values[col].to_numpy()
        j = int(np.argmax(y) if higher else np.argmin(y))
        ax.plot(cs, y, "o-", color=color, lw=2)
        ax.scatter([cs[j]], [y[j]], s=260, marker="*", color="gold", edgecolors="black",
                   zorder=5, label=f"terbaik: c = {cs[j]} ({y[j]:.3f})".replace(".", ","))
        for xv, yv in zip(cs, y):
            ax.annotate(f"{yv:.3f}".replace(".", ","), (xv, yv), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7)
        ax.margins(y=0.16)
        ax.set_xticks(cs)
        ax.set_xlabel("Jumlah klaster c")
        ax.set_title(title, fontsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_id))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Program utama
# ----------------------------------------------------------------------------
def run(data_dir, outdir, m=2.0):
    data_dir, out = Path(data_dir), Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    cs = available_c(data_dir)
    X, parts = load_all(data_dir, cs)
    print(f"Data: {len(X):,} pelanggan; c = {cs[0]}..{cs[-1]}; m = {m:g}\n")

    rows = []
    for c in cs:
        U, V, labels = parts[c]
        rows.append({"c": c, "MPC": mpc(U), "Xie_Beni": xie_beni(X, U, V, m),
                     "Silhouette": silhouette(X, labels)})
        print(f"c={c}: MPC={rows[-1]['MPC']:.4f} | XB={rows[-1]['Xie_Beni']:.4f} | "
              f"Silhouette={rows[-1]['Silhouette']:.4f}")
    values = pd.DataFrame(rows)
    ranks = make_rank(values)
    values.to_csv(out / "nilai_indeks_per_c.csv", index=False, float_format="%.6f")
    ranks.to_csv(out / "peringkat_per_c.csv", index=False, float_format="%.4f")
    plot_indices(values, out / "evaluasi_indeks.png")
    print(f"\nHasil di: {out.resolve()}")
    return values, ranks


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default=str(here / "data"))
    p.add_argument("--outdir", default=str(here / "hasil_langkah5"))
    p.add_argument("--m", type=float, default=2.0,
                   help="fuzzifier; harus sama dengan m yang dipakai di langkah 4")
    a = p.parse_args()
    if a.m <= 1:
        sys.exit("m harus > 1")
    run(a.data, a.outdir, a.m)


if __name__ == "__main__":
    main()
