#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LANGKAH 4 - Menerapkan Fuzzy C-Means (FCM) pada data RFM yang sudah dibersihkan.

Input  : data/rfm_transformasi.csv  (keluaran langkah 3; kolom z_R, z_F, z_M)
Output : hasil_langkah4/
    ringkasan_per_c.csv        J, iterasi, ukuran klaster untuk tiap c
    centroid_c{c}.csv          centroid tiap klaster (skala z + satuan asli) + jumlah anggota
    anggota_c{c}.csv           tiap pelanggan: keanggotaan ke semua klaster + klaster dominan
    sebaran_c{c}.png           gambar sebaran data, centroid, dan garis ke centroid
    parameter_z.json           mean & std yang dipakai z-score (untuk memproses data baru)
    sensitivitas_m_c{c}.csv    (opsional, --sens-c) hasil untuk beberapa nilai m

Rumus (m = fuzzifier, c = jumlah klaster):
    Pusat klaster  v_k  = sum_i (u_ik^m * x_i) / sum_i (u_ik^m)
    Keanggotaan    u_ik = 1 / sum_j ( d_ik / d_ij )^(2/(m-1))
    Fungsi tujuan  J    = sum_i sum_k (u_ik^m) * d_ik^2
    Berhenti jika  max |u_ik(baru) - u_ik(lama)| < eps

Pemakaian:
    python langkah4_fcm.py                       # c = 2..8, m = 2
    python langkah4_fcm.py --c-min 2 --c-max 6 --m 2
    python langkah4_fcm.py --sens-c 3            # tambah uji sensitivitas m untuk c = 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["z_R", "z_F", "z_M"]
# kolom dasar tiap fitur z dan apakah dasarnya di-ln (untuk mengembalikan centroid ke satuan asli)
BASE = {"z_R": ("R", False, "R_hari"), "z_F": ("ln_F", True, "F_faktur"),
        "z_M": ("ln_M", True, "M_pound")}
PAIRS = [("z_R", "z_F"), ("z_R", "z_M"), ("z_F", "z_M")]
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#9D755D",
          "#17BECF", "#7F7F7F"]
ZERO = 1e-12


# ----------------------------------------------------------------------------
# 1. Inti algoritma FCM
# ----------------------------------------------------------------------------
def distances(X, V):
    """Jarak Euclidean tiap data ke tiap centroid -> matriks (N, c)."""
    return np.sqrt(((X[:, None, :] - V[None, :, :]) ** 2).sum(axis=2))


def centers(X, U, m):
    """v_k = rata-rata data berbobot u^m."""
    W = U ** m
    return (W.T @ X) / W.sum(axis=0)[:, None]


def memberships(D, m):
    """u_ik = 1 / sum_j (d_ik/d_ij)^(2/(m-1)), dihitung stabil secara numerik."""
    p = 2.0 / (m - 1.0)
    zero = D <= ZERO
    has0 = zero.any(axis=1)
    U = np.empty_like(D)
    ok = ~has0
    if ok.any():
        s = -p * np.log(D[ok])
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        U[ok] = e / e.sum(axis=1, keepdims=True)
    if has0.any():                       # data tepat di atas centroid
        z = zero[has0].astype(float)
        U[has0] = z / z.sum(axis=1, keepdims=True)
    return U


def objective(U, D, m):
    """J = sum sum u^m d^2."""
    return float(((U ** m) * D ** 2).sum())


def fcm(X, c, m=2.0, eps=1e-5, max_iter=300, rng=None):
    """Satu kali jalan FCM dari satu inisialisasi acak. -> (V, U, J, n_iter, konvergen)."""
    rng = rng or np.random.default_rng()
    U = rng.random((len(X), c))
    U /= U.sum(axis=1, keepdims=True)    # tiap baris berjumlah 1
    conv = False
    for it in range(1, max_iter + 1):
        V = centers(X, U, m)
        U_new = memberships(distances(X, V), m)
        delta = np.abs(U_new - U).max()
        U = U_new
        if delta < eps:
            conv = True
            break
    V = centers(X, U, m)
    return V, U, objective(U, distances(X, V), m), it, conv


def fit_best(X, c, m=2.0, eps=1e-5, max_iter=300, n_init=20, seed=42):
    """FCM diulang n_init kali dengan inisialisasi berbeda; ambil J terkecil."""
    runs = [fcm(X, c, m, eps, max_iter, np.random.default_rng(s))
            for s in np.random.SeedSequence(seed).spawn(n_init)]
    best = min(runs, key=lambda r: r[2])
    pct = 100 * np.mean([r[2] <= best[2] * (1 + 1e-6) for r in runs])
    V, U, J, it, conv = best
    order = np.argsort(-V[:, -1], kind="stable")   # klaster 1 = z_M (belanja) tertinggi
    return dict(V=V[order], U=U[:, order], J=J, iterasi=it, konvergen=conv,
                pct_J_terbaik=pct)


def adjusted_rand_index(a, b):
    """Kemiripan dua pembagian klaster (1 = identik)."""
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    t = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(t, (a, b), 1)
    c2 = lambda x: x * (x - 1) / 2
    sij, sa, sb = c2(t).sum(), c2(t.sum(1)).sum(), c2(t.sum(0)).sum()
    exp, mx = sa * sb / c2(len(a)), (sa + sb) / 2
    return 1.0 if mx == exp else float((sij - exp) / (mx - exp))


# ----------------------------------------------------------------------------
# 2. Centroid & keanggotaan -> tabel
# ----------------------------------------------------------------------------
def z_parameters(df):
    """Mean & std (populasi) yang dipakai membentuk kolom z_ dari data langkah 3."""
    return {f: {"kolom_dasar": BASE[f][0], "di_ln": BASE[f][1],
                "mean": float(df[BASE[f][0]].mean()),
                "std": float(df[BASE[f][0]].std(ddof=0))} for f in FEATURES}


def centroid_table(V, sizes, zp):
    """Centroid skala z + satuan asli (balik z-score, lalu balik ln -> exp)."""
    t = pd.DataFrame(V, columns=FEATURES)
    for j, f in enumerate(FEATURES):
        val = V[:, j] * zp[f]["std"] + zp[f]["mean"]
        t[BASE[f][2]] = np.exp(val) if zp[f]["di_ln"] else val
    t.insert(0, "klaster", np.arange(1, len(V) + 1))
    t["jumlah_anggota"] = sizes
    t["persen_anggota"] = 100 * sizes / sizes.sum()
    return t


def member_table(ids, U):
    t = pd.DataFrame(U, columns=[f"u_klaster{k + 1}" for k in range(U.shape[1])])
    t.insert(0, "Customer ID", ids)
    t["klaster_dominan"] = U.argmax(axis=1) + 1
    t["keanggotaan_maks"] = U.max(axis=1)
    return t


# ----------------------------------------------------------------------------
# 3. Gambar sebaran
# ----------------------------------------------------------------------------
def fmt_id(x, _=None):
    """Format angka gaya Indonesia untuk sumbu grafik: 1.234,5 (titik ribuan, koma desimal)."""
    s = f"{x:,.6g}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def plot_scatter(df, V, labels, c, m, path, n_lines=150, seed=0):
    """3 panel (R-F, R-M, F-M) pada kisi 2x2 (panel keempat = keterangan). Titik = pelanggan (warna = klaster dominan), bintang = centroid,
    garis tipis = sampel acak n_lines pelanggan ke centroid klaster dominannya."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    X = df[FEATURES].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_lines, len(X)), replace=False)
    nice = {"z_R": "z_R  (Recency, makin kanan = makin lama tidak belanja)",
            "z_F": "z_F  (ln Frequency, terstandardisasi)",
            "z_M": "z_M  (ln Monetary, terstandardisasi)"}
    fig, grid = plt.subplots(2, 2, figsize=(8, 7.4))
    axes = list(grid.flat)
    for ax, (fx, fy) in zip(axes[:3], PAIRS):
        jx, jy = FEATURES.index(fx), FEATURES.index(fy)
        for k in range(c):
            sel = labels == k
            ax.scatter(X[sel, jx], X[sel, jy], s=5, alpha=0.35,
                       color=COLORS[k], linewidths=0,
                       label=f"Klaster {k + 1} (n={sel.sum():,})".replace(",", "."))
        for i in idx:
            k = labels[i]
            ax.plot([X[i, jx], V[k, jx]], [X[i, jy], V[k, jy]],
                    color=COLORS[k], alpha=0.35, lw=0.6)
        ax.scatter(V[:, jx], V[:, jy], marker="*", s=300, c=COLORS[:c],
                   edgecolors="black", linewidths=1.5, zorder=5)
        for k in range(c):
            ax.annotate(str(k + 1), (V[k, jx], V[k, jy]), ha="center", va="center",
                        fontsize=8, fontweight="bold", color="black", zorder=6)
        ax.set_xlabel(nice[fx], fontsize=8)
        ax.set_ylabel(nice[fy], fontsize=8)
        ax.set_title(f"{fx[2]} vs {fy[2]}", fontsize=10)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_id))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_id))
        ax.grid(alpha=0.25)
    handles, names = axes[0].get_legend_handles_labels()
    axes[3].axis("off")
    axes[3].legend(handles, names, markerscale=3, fontsize=9, loc="upper left",
                   title=f"Klaster dominan (c = {c}, m = {m:g})".replace(".", ","), title_fontsize=9)
    axes[3].text(0.0, 0.42,
                 "Keterangan:\n- titik = pelanggan (warna = klaster dominan)\n"
                 "- bintang = pusat klaster\n"
                 f"- garis tipis = {len(idx)} pelanggan sampel\n  ke pusat klaster dominannya",
                 fontsize=9, va="top", transform=axes[3].transAxes)
    fig.suptitle(f"Hasil FCM: c = {c}, m = {m:g}".replace(".", ","), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. Program utama
# ----------------------------------------------------------------------------
def run(input_path, outdir, c_min=2, c_max=8, m=2.0, eps=1e-5, max_iter=300,
        n_init=20, seed=42, sens_c=None, sens_m=(1.5, 2.0, 2.5)):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    need = ["Customer ID", "R", "ln_F", "ln_M", *FEATURES]
    miss = [k for k in need if k not in df.columns]
    if miss:
        sys.exit(f"Kolom tidak ditemukan di {input_path}: {miss}")
    if df[FEATURES].isna().any().any():
        sys.exit("Ada nilai kosong pada z_R/z_F/z_M.")
    X = df[FEATURES].to_numpy(float)
    zp = z_parameters(df)
    (out / "parameter_z.json").write_text(json.dumps(zp, indent=2), encoding="utf-8")
    print(f"Data: {len(df):,} pelanggan, fitur {FEATURES}; m={m}, eps={eps}, n_init={n_init}\n")

    rows = []
    for c in range(c_min, c_max + 1):
        r = fit_best(X, c, m, eps, max_iter, n_init, seed)
        labels = r["U"].argmax(axis=1)
        sizes = np.bincount(labels, minlength=c)
        centroid_table(r["V"], sizes, zp).to_csv(out / f"centroid_c{c}.csv", index=False,
                                                 float_format="%.6f")
        member_table(df["Customer ID"].values, r["U"]).to_csv(
            out / f"anggota_c{c}.csv", index=False, float_format="%.6f")
        plot_scatter(df, r["V"], labels, c, m, out / f"sebaran_c{c}.png")
        rows.append({"c": c, "m": m, "J": r["J"], "iterasi": r["iterasi"],
                     "konvergen": r["konvergen"],
                     "pct_inisialisasi_capai_J_terbaik": r["pct_J_terbaik"],
                     "ukuran_klaster": "|".join(map(str, sizes))})
        print(f"c={c}: J={r['J']:.2f} | iterasi={r['iterasi']} | "
              f"capai J terbaik {r['pct_J_terbaik']:.0f}% dari {n_init} inisialisasi | "
              f"anggota {rows[-1]['ukuran_klaster']}")
    pd.DataFrame(rows).to_csv(out / "ringkasan_per_c.csv", index=False, float_format="%.6f")

    if sens_c:                                   # uji sensitivitas terhadap m
        base = fit_best(X, sens_c, 2.0, eps, max_iter, n_init, seed)
        base_lab = base["U"].argmax(axis=1)
        srows = []
        for mm in sens_m:
            r = base if mm == 2.0 else fit_best(X, sens_c, mm, eps, max_iter, n_init, seed)
            lab = r["U"].argmax(axis=1)
            srows.append({"c": sens_c, "m": mm, "J": r["J"],
                          "ukuran_klaster": "|".join(map(str, np.bincount(lab, minlength=sens_c))),
                          "ARI_vs_m2": adjusted_rand_index(lab, base_lab),
                          "persen_keanggotaan_maks_<0.6":
                              100 * float((r["U"].max(axis=1) < 0.6).mean())})
        st = pd.DataFrame(srows)
        st.to_csv(out / f"sensitivitas_m_c{sens_c}.csv", index=False, float_format="%.6f")
        print(f"\nSensitivitas m untuk c={sens_c}:\n{st.to_string(index=False)}")
    print(f"\nSelesai. Hasil di: {out.resolve()}")


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", default=str(here / "data" / "rfm_transformasi.csv"))
    p.add_argument("--outdir", default=str(here / "hasil_langkah4"))
    p.add_argument("--c-min", type=int, default=2)
    p.add_argument("--c-max", type=int, default=8)
    p.add_argument("--m", type=float, default=2.0, help="fuzzifier (> 1), default 2")
    p.add_argument("--eps", type=float, default=1e-5, help="ambang konvergensi")
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--n-init", type=int, default=20, help="jumlah inisialisasi acak per c")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sens-c", type=int, default=None,
                   help="jika diisi, jalankan uji sensitivitas m (1.5, 2, 2.5) untuk c ini")
    a = p.parse_args()
    if a.m <= 1:
        sys.exit("m harus > 1")
    if a.c_min < 2 or a.c_max < a.c_min:
        sys.exit("c-min harus >= 2 dan c-max >= c-min")
    run(a.input, a.outdir, a.c_min, a.c_max, a.m, a.eps, a.max_iter, a.n_init, a.seed,
        a.sens_c)


if __name__ == "__main__":
    main()
