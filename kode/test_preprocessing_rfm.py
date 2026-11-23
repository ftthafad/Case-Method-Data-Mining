# -*- coding: utf-8 -*-
"""
Tes untuk preprocessing_rfm.py. Jalankan:   python -m pytest -v

Kasus acuan memakai 10 baris ASLI milik pelanggan 12403 dari dataset
(dihitung manual: R = 50 hari, F = 2 invoice, M = 947,36).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocessing_rfm as pp  # noqa: E402

REF = pd.Timestamp("2011-12-10")


def make_frame(rows):
    cols = ["Invoice", "StockCode", "Description", "Quantity",
            "InvoiceDate", "Price", "Customer ID", "Country"]
    df = pd.DataFrame(rows, columns=cols)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


# 10 baris asli pelanggan 12403 (Denmark)
CUSTOMER_12403 = [
    ("514808", "84997D", "PINK 3 PIECE MINI DOTS CUTLERY SET", 72, "2010-07-06 11:39", 3.39, 12403.0, "Denmark"),
    ("514808", "84997C", "BLUE 3 PIECE MINI DOTS CUTLERY SET", 72, "2010-07-06 11:39", 3.39, 12403.0, "Denmark"),
    ("514808", "84997A", "GREEN 3 PIECE MINI DOTS CUTLERY SET", 18, "2010-07-06 11:39", 3.75, 12403.0, "Denmark"),
    ("514808", "POST", "POSTAGE", 2, "2010-07-06 11:39", 18.00, 12403.0, "Denmark"),
    ("572184", "84997D", "CHILDRENS CUTLERY POLKADOT PINK", 40, "2011-10-21 10:51", 4.15, 12403.0, "Denmark"),
    ("572184", "84997C", "CHILDRENS CUTLERY POLKADOT BLUE", 20, "2011-10-21 10:51", 4.15, 12403.0, "Denmark"),
    ("572184", "84997A", "CHILDRENS CUTLERY POLKADOT GREEN", 32, "2011-10-21 10:51", 4.15, 12403.0, "Denmark"),
    ("572184", "21770", "OPEN CLOSED METAL SIGN", 2, "2011-10-21 10:51", 4.95, 12403.0, "Denmark"),
    ("572184", "POST", "POSTAGE", 2, "2011-10-21 10:51", 18.00, 12403.0, "Denmark"),
    ("C572187", "POST", "POSTAGE", -1, "2011-10-21 11:00", 18.00, 12403.0, "Denmark"),
]


# ---------------------------------------------------------------- kasus acuan
def test_pelanggan_12403_cocok_dengan_hitungan_manual():
    raw = make_frame(CUSTOMER_12403)
    clean, _ = pp.clean_transactions(raw)
    rfm = pp.compute_rfm(clean, REF).set_index("Customer ID")
    row = rfm.loc[12403]
    assert row["R"] == 50
    assert row["F"] == 2
    assert row["M"] == pytest.approx(947.36, abs=1e-9)


def test_baris_valid_pelanggan_12403_ada_tujuh():
    clean, _ = pp.clean_transactions(make_frame(CUSTOMER_12403))
    assert len(clean) == 7
    assert set(clean["Invoice"]) == {"514808", "572184"}


# ------------------------------------------------------------- tiap filter
def base_row(**kw):
    row = dict(Invoice="500001", StockCode="85123A", Description="X",
               Quantity=1, InvoiceDate="2011-01-01 10:00", Price=2.0,
               **{"Customer ID": 1.0}, Country="United Kingdom")
    row.update(kw)
    return row


def frame_from(dicts):
    return make_frame([tuple(d.values()) for d in dicts])


def count_after(dicts):
    clean, _ = pp.clean_transactions(frame_from(dicts))
    return len(clean)


def test_duplikat_identik_dibuang_satu():
    r = base_row()
    assert count_after([r, dict(r)]) == 1


def test_baris_beda_sedikit_bukan_duplikat():
    assert count_after([base_row(), base_row(Quantity=2)]) == 2


def test_tanpa_customer_id_dibuang():
    assert count_after([base_row(**{"Customer ID": np.nan})]) == 0


def test_pembatalan_dibuang():
    assert count_after([base_row(Invoice="C500002", Quantity=-1)]) == 0


def test_quantity_nol_atau_negatif_dibuang():
    assert count_after([base_row(Quantity=-3), base_row(Invoice="500003", Quantity=0)]) == 0


def test_harga_nol_atau_negatif_dibuang():
    assert count_after([base_row(Price=0.0), base_row(Invoice="500003", Price=-5.0)]) == 0


@pytest.mark.parametrize("code", sorted(pp.NON_PRODUCT_CODES))
def test_kode_non_produk_dibuang(code):
    assert count_after([base_row(StockCode=code)]) == 0


def test_voucher_hadiah_dan_produk_dcgs_dipertahankan():
    rows = [base_row(StockCode="gift_0001_20", Invoice="500004"),
            base_row(StockCode="DCGS0058", Invoice="500005")]
    assert count_after(rows) == 2


def test_log_pembersihan_konsisten():
    raw = make_frame(CUSTOMER_12403)
    clean, log = pp.clean_transactions(raw)
    assert log.iloc[0]["baris_tersisa"] == len(raw)
    assert log.iloc[-1]["baris_tersisa"] == len(clean)
    # jumlah dibuang + sisa akhir = jumlah awal
    assert log["baris_dibuang"].sum() + len(clean) == len(raw)


# --------------------------------------------------------------------- RFM
def test_frequency_menghitung_invoice_unik_bukan_baris():
    rows = [base_row(StockCode=f"1000{i}", Invoice="500010") for i in range(5)]
    clean, _ = pp.clean_transactions(frame_from(rows))
    rfm = pp.compute_rfm(clean, REF)
    assert rfm.loc[0, "F"] == 1
    assert len(clean) == 5


def test_recency_memakai_tanggal_terakhir_dan_selisih_hari():
    rows = [base_row(Invoice="500020", InvoiceDate="2011-12-09 12:50"),
            base_row(Invoice="500021", InvoiceDate="2011-06-01 08:00")]
    clean, _ = pp.clean_transactions(frame_from(rows))
    assert pp.compute_rfm(clean, REF).loc[0, "R"] == 1


def test_monetary_adalah_jumlah_quantity_kali_price():
    rows = [base_row(Quantity=3, Price=2.5, StockCode="10001"),
            base_row(Quantity=4, Price=1.0, StockCode="10002")]
    clean, _ = pp.clean_transactions(frame_from(rows))
    assert pp.compute_rfm(clean, REF).loc[0, "M"] == pytest.approx(11.5)


def test_satu_baris_per_pelanggan():
    rows = [base_row(**{"Customer ID": 1.0}), base_row(Invoice="500030", **{"Customer ID": 2.0}),
            base_row(Invoice="500031", **{"Customer ID": 2.0})]
    clean, _ = pp.clean_transactions(frame_from(rows))
    rfm = pp.compute_rfm(clean, REF)
    assert rfm["Customer ID"].is_unique and len(rfm) == 2


def test_tanggal_acuan_default_sehari_setelah_transaksi_terakhir():
    raw = make_frame([("1", "10001", "x", 1, "2011-12-09 12:50", 1.0, 1.0, "UK")])
    assert pp.default_reference_date(raw) == pd.Timestamp("2011-12-10")


# --------------------------------------------------- transformasi & z-score
def sample_rfm():
    return pd.DataFrame({"Customer ID": range(1, 9),
                         "R": [1, 5, 20, 60, 120, 300, 500, 700],
                         "F": [1, 1, 2, 3, 5, 8, 20, 300],
                         "M": [10, 50, 120, 400, 900, 2500, 9000, 500000.0]})


def test_zscore_rata_nol_dan_simpangan_satu():
    t = pp.transform_rfm(sample_rfm(), ["F", "M"])
    for v in pp.RFM_COLUMNS:
        assert t[f"z_{v}"].mean() == pytest.approx(0, abs=1e-9)
        assert t[f"z_{v}"].std(ddof=0) == pytest.approx(1, abs=1e-9)


def test_ln_hanya_untuk_variabel_terpilih():
    t = pp.transform_rfm(sample_rfm(), ["F", "M"])
    assert "ln_R" not in t.columns
    assert t.loc[0, "ln_M"] == pytest.approx(np.log(10))
    assert t.loc[7, "ln_F"] == pytest.approx(np.log(300))


def test_ln_mengurangi_kemiringan():
    rfm = sample_rfm()
    t = pp.transform_rfm(rfm, ["F", "M"])
    assert abs(t["ln_M"].skew()) < abs(rfm["M"].skew())
    assert abs(t["ln_F"].skew()) < abs(rfm["F"].skew())


def test_ln_menolak_nilai_nol_atau_negatif():
    bad = sample_rfm()
    bad.loc[0, "M"] = 0
    with pytest.raises(ValueError):
        pp.transform_rfm(bad, ["M"])


def test_urutan_pelanggan_tidak_berubah_setelah_ln():
    rfm = sample_rfm()
    t = pp.transform_rfm(rfm, ["F", "M"])
    assert list(t["ln_M"].rank()) == list(rfm["M"].rank())


# ----------------------------------------------------- membaca file & e2e
def test_alias_nama_kolom_diseragamkan(tmp_path):
    df = make_frame(CUSTOMER_12403).rename(columns={
        "Invoice": "InvoiceNo", "Price": "UnitPrice", "Customer ID": "CustomerID"})
    f = tmp_path / "data.csv"
    df.to_csv(f, index=False)
    loaded = pp.load_raw(f)
    assert list(loaded.columns) == pp.CANONICAL_COLUMNS
    assert len(loaded) == 10


def test_kolom_wajib_hilang_memicu_error(tmp_path):
    f = tmp_path / "rusak.csv"
    pd.DataFrame({"a": [1]}).to_csv(f, index=False)
    with pytest.raises(ValueError):
        pp.load_raw(f)


def test_end_to_end_menghasilkan_semua_berkas(tmp_path):
    rows = list(CUSTOMER_12403) + [
        ("520001", "10001", "x", 5, "2011-12-09 09:00", 4.0, 9999.0, "UK"),
        ("520002", "10002", "y", 1, "2011-03-01 09:00", 50.0, 8888.0, "UK"),
        ("520003", "10003", "z", 2, "2011-05-01 09:00", 7.0, 8888.0, "UK"),
        ("520004", "10004", "w", 9, "2011-07-01 09:00", 3.0, 7777.0, "UK"),
    ]
    f = tmp_path / "mini.csv"
    make_frame(rows).to_csv(f, index=False)
    out = tmp_path / "hasil"
    summary = pp.run(str(f), str(out), None, ["F", "M"])
    for name in ["transaksi_bersih.csv.gz", "rfm_mentah.csv", "rfm_transformasi.csv",
                 "log_pembersihan.csv", "ringkasan.json", "grafik_sebaran_rfm.png"]:
        assert (out / name).exists(), name
    mentah = pd.read_csv(out / "rfm_mentah.csv").set_index("Customer ID")
    assert summary["jumlah_pelanggan"] == 4
    assert mentah.loc[12403, ["R", "F"]].tolist() == [50, 2]
    assert mentah.loc[12403, "M"] == pytest.approx(947.36)
    assert mentah.loc[9999, "R"] == 1   # transaksi terakhir 9 Des 2011, acuan 10 Des
