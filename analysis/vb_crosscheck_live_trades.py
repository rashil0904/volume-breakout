# -*- coding: utf-8 -*-
"""vb_crosscheck_live_trades.py — cross-check the Aug-18+ baseline backtest (new daily mcap source) against the
LIVE trading records in data/trades/: trade_list_*.csv (live signal list: symbol, shares, ref_price),
dhan_entries_*.csv and mtf_entries_*.csv (actual/dry-run broker orders). Read-only; new file.
"""
import sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import vb_baseline_newmcap_aug18 as V

TR = rb.BASE / "data" / "trades" / "trades"
OUT = V.OUT
START = V.START
IST = "Asia/Kolkata"


def load_live():
    L = pd.concat([pd.read_csv(f).assign(date=pd.Timestamp(os.path.basename(f)[11:21]).date())
                   for f in sorted(glob.glob(str(TR / "trade_list_*.csv")))], ignore_index=True)
    D = pd.concat([pd.read_csv(f).assign(date=pd.Timestamp(os.path.basename(f)[13:23]).date(), src="dhan")
                   for f in sorted(glob.glob(str(TR / "dhan_entries_*.csv")))], ignore_index=True)
    M = pd.concat([pd.read_csv(f).assign(date=pd.Timestamp(os.path.basename(f)[12:22]).date(), src="mtf")
                   for f in sorted(glob.glob(str(TR / "mtf_entries_*.csv")))], ignore_index=True)
    B = pd.concat([D, M], ignore_index=True)
    return L, B


def day_bars(sym, d, folder_map):
    folder = folder_map.get(sym)
    if folder is None:
        return None
    raw = pd.read_parquet(V.DIRS[folder] / f"{sym}.parquet", columns=["timestamp", "open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
    raw = raw.assign(date=ts.dt.date, hm=ts.dt.hour * 60 + ts.dt.minute)
    x = raw[raw["date"] == d]
    return x.set_index("hm") if len(x) else None


def main():
    L, B = load_live()
    S = pd.read_parquet(V.SCAN_FN); S["date"] = pd.to_datetime(S["date"]).dt.date
    new_mcap = V.load_new_mcap()
    old_lbl, old_elig = V.load_old_snapshot()
    folder_map = S.drop_duplicates("symbol").set_index("symbol")["folder"].to_dict()

    S["vr"] = S["passes_vol"] & S["passes_ret"]
    sig = S[S["vr"] & (S["date"] >= pd.Timestamp("2026-07-16").date())].copy()
    sig["in_new"] = [(s in set(new_mcap[d]["symbol"])) if d in new_mcap else np.nan for s, d in zip(sig["symbol"], sig["date"])]
    sig["in_old"] = sig["symbol"].isin(old_elig.keys())

    # ── 1. signal-set comparison, per day ────────────────────────────────────────────────────────
    rows = []
    for d, g in L.groupby("date"):
        live = set(g["symbol"]); sd = sig[sig["date"] == d]
        row = {"date": d, "n_live": len(live)}
        for tag, col in [("new", "in_new"), ("old", "in_old")]:
            if tag == "new" and d not in new_mcap:
                row.update({f"n_{tag}": np.nan, f"both_{tag}": np.nan, f"only_live_{tag}": np.nan, f"only_mine_{tag}": np.nan}); continue
            mine = set(sd[sd[col] == True]["symbol"])
            row.update({f"n_{tag}": len(mine), f"both_{tag}": len(live & mine),
                        f"only_live_{tag}": len(live - mine), f"only_mine_{tag}": len(mine - live)})
        rows.append(row)
    DAY = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)

    def period_sum(df, lo, hi, tag):
        p = df[(df["date"] >= lo) & (df["date"] <= hi)]
        p = p[p[f"n_{tag}"].notna()]
        return {"period": f"{lo}..{hi}", "source_hypothesis": tag, "days": len(p), "live_names": int(p["n_live"].sum()),
                "my_names": int(p[f"n_{tag}"].sum()), "both": int(p[f"both_{tag}"].sum()),
                "only_live": int(p[f"only_live_{tag}"].sum()), "only_mine": int(p[f"only_mine_{tag}"].sum()),
                "days_exact_match": int(((p[f"only_live_{tag}"] == 0) & (p[f"only_mine_{tag}"] == 0)).sum())}
    PS = pd.DataFrame([period_sum(DAY, START, pd.Timestamp("2026-09-18").date(), "new"),
                       period_sum(DAY, START, pd.Timestamp("2026-09-18").date(), "old"),
                       period_sum(DAY, pd.Timestamp("2026-07-16").date(), pd.Timestamp("2026-08-17").date(), "new"),
                       period_sum(DAY, pd.Timestamp("2026-07-16").date(), pd.Timestamp("2026-08-17").date(), "old")])
    print("=== 1. LIVE trade_list vs MY signals (vol+ret+mcap) ===")
    print(PS.to_string(index=False), flush=True)

    # ── diagnose mismatches (new-source hypothesis, Aug18+) ──────────────────────────────────────
    diag = []
    for r in DAY[(DAY["date"] >= START)].itertuples():
        d = r.date; live = set(L[L["date"] == d]["symbol"]); sd = sig[sig["date"] == d]
        mine = set(sd[sd["in_new"] == True]["symbol"])
        for s in sorted(live - mine):
            sc = S[(S["symbol"] == s) & (S["date"] == d)]
            if sc.empty:
                diag.append({"date": d, "symbol": s, "kind": "ONLY_LIVE", "why": "no scan row (insufficient history / no data)"}); continue
            x = sc.iloc[0]
            in_file = s in set(new_mcap[d]["symbol"]) if d in new_mcap else None
            why = ("vol fails" if not x.passes_vol else "") + (" ret fails" if not x.passes_ret else "") + ("" if in_file else " not in mcap file")
            diag.append({"date": d, "symbol": s, "kind": "ONLY_LIVE", "vol_ratio": round(x.vol_ratio, 2), "ret_pct": round(x.ret_pct, 2),
                         "in_new_file": in_file, "in_old_band": s in old_elig, "why": why.strip() or "??"})
        for s in sorted(mine - live):
            x = sd[sd["symbol"] == s].iloc[0]
            diag.append({"date": d, "symbol": s, "kind": "ONLY_MINE", "vol_ratio": round(x.vol_ratio, 2), "ret_pct": round(x.ret_pct, 2),
                         "in_new_file": True, "in_old_band": s in old_elig, "why": "on my signal set, absent from live list"})
    DG = pd.DataFrame(diag)
    print(f"\nmismatches Aug18+ (new-source hypothesis): only-live={int((DG.kind=='ONLY_LIVE').sum()) if len(DG) else 0} only-mine={int((DG.kind=='ONLY_MINE').sum()) if len(DG) else 0}")
    if len(DG):
        print(DG.to_string(index=False), flush=True)

    # ── 2. ref_price identity + list sizing rule ─────────────────────────────────────────────────
    cands = {"c1510": (910, "close"), "o1511": (911, "open"), "c1509": (909, "close"), "o1510": (910, "open"),
             "o1500": (900, "open"), "o1515": (915, "open"), "o1521": (921, "open")}
    errs = {k: [] for k in cands}; ex = []
    Lm = L[(L["date"] >= START) & L["ref_price"].notna()]
    for r in Lm.itertuples():
        bars = day_bars(r.symbol, r.date, folder_map)
        if bars is None:
            continue
        for k, (hm, col) in cands.items():
            if hm in bars.index:
                errs[k].append(abs(bars.loc[hm, col] / r.ref_price - 1) * 100)
    RP = pd.DataFrame([{"candidate": k, "n": len(v), "median_abs_err_pct": round(np.median(v), 4), "pct_within_0.05": round(np.mean(np.array(v) <= 0.05) * 100, 1)}
                       for k, v in errs.items() if v]).sort_values("median_abs_err_pct")
    print("\n=== 2a. which price is the list's ref_price? (Aug18+ list rows) ===")
    print(RP.to_string(index=False), flush=True)

    sz = []
    n_nan = int(L["ref_price"].isna().sum())
    for d, g in L.groupby("date"):
        n = len(g)
        per = min(100_000, 500_000 / n)
        for r in g[g["ref_price"].notna()].itertuples():
            sz.append({"date": d, "symbol": r.symbol, "list_shares": r.shares, "expected": int(np.floor(per / r.ref_price)), "n_list": n})
    SZ = pd.DataFrame(sz)
    print(f"\n=== 2b. list sizing rule: shares == floor(min(1L, 5L/n_list)/ref_price)? exact match {int((SZ.list_shares==SZ.expected).sum())} of {len(SZ)} "
          f"| within +/-1 share {int(((SZ.list_shares-SZ.expected).abs()<=1).sum())}", flush=True)

    # ── 2c. the 07-16/07-17 files carry live-computed diagnostics -> compare with my scan ─────────
    ex_rows = []
    for f in sorted(glob.glob(str(TR / "trade_list_2026-07-1[67].csv"))):
        d0 = pd.Timestamp(os.path.basename(f)[11:21]).date(); x = pd.read_csv(f)
        for r in x.itertuples():
            sc = S[(S["symbol"] == r.symbol) & (S["date"] == d0)]
            if sc.empty:
                continue
            q = sc.iloc[0]
            ex_rows.append({"date": d0, "symbol": r.symbol, "live_volume_ratio": r.volume_ratio, "my_volume_ratio": round(q.vol_ratio, 4),
                            "live_ret_pct": r.return_pct_vs_prev_close, "my_ret_pct": round(q.ret_pct, 4),
                            "live_prev_vwap": r.prev_day_vwap_close, "my_prev_vwap": round(q.pc, 4),
                            "live_avg36": r.avg_36day_volume, "my_avg36": round(q.avg_vol, 2)})
    EX = pd.DataFrame(ex_rows)
    print("\n=== 2c. live-computed diagnostics (07-16/07-17 files) vs my scan ===")
    print(EX.to_string(index=False), flush=True)

    # ── 3. broker orders vs list vs backtest ─────────────────────────────────────────────────────
    Tn = pd.read_csv(V.OUT / "trades_NEW_daily_mcap.csv"); Tn["entry_date"] = pd.to_datetime(Tn["entry_date"]).dt.date
    Bm = B.copy()
    Bm["key"] = list(zip(Bm["date"], Bm["symbol"]))
    Lm2 = L[L["date"] >= pd.Timestamp("2026-08-03").date()].copy()
    Lm2["key"] = list(zip(Lm2["date"], Lm2["symbol"]))
    jl = Lm2.merge(Bm[["date", "symbol", "status", "src", "fill_price", "ref_price", "quantity", "capital_base", "product", "timestamp"]],
                   on=["date", "symbol"], how="left", suffixes=("_list", "_broker"))
    jl["status"] = jl["status"].fillna("NO_BROKER_ROW")
    print("\n=== 3a. every list name since 08-03 -> broker outcome ===")
    print(jl.groupby(["src", "status"], dropna=False).size().to_string(), flush=True)
    order_days = sorted(set(B["date"])); list_days = sorted(set(L["date"]))
    print("list days with NO broker file (>= 08-03):", [str(d) for d in list_days if d >= pd.Timestamp('2026-08-03').date() and d not in set(B['date'])], flush=True)
    orph = Bm[~Bm["key"].isin(set(Lm2["key"]))]
    print(f"broker rows NOT on that day's list: {len(orph)}", flush=True)
    if len(orph):
        print(orph[["date", "symbol", "status", "src", "quantity", "capital_base"]].to_string(index=False), flush=True)

    # not_filled/rejected vs my no-trade signals
    keys_T = set(zip(Tn["symbol"], Tn["entry_date"]))
    mine_all = sig[(sig["date"] >= START) & (sig["in_new"] == True)]
    no_trade = mine_all[[(s, d) not in keys_T and d <= pd.Timestamp("2026-09-17").date() for s, d in zip(mine_all["symbol"], mine_all["date"])]]
    nf = jl[jl["status"].isin(["not_filled", "rejected"]) & (jl["date"] >= START)]
    print(f"\n=== 3b. my signals with NO backtest trade (Aug18-Sep17): {len(no_trade)} | live not_filled/rejected (Aug18+): {len(nf)} ===")
    nt = set(zip(no_trade["symbol"], no_trade["date"])); nfk = set(zip(nf["symbol"], nf["date"]))
    print(f"overlap: {len(nt & nfk)} | my-no-trade but live filled/other: {len(nt - nfk)} | live-not-filled but I traded: {len(nfk - nt)}")
    print("my no-trade signals:", sorted((str(d), s) for s, d in nt), flush=True)
    print("live not_filled/rejected:", sorted((str(d), s) for s, d in nfk), flush=True)
    print("live-not-filled but MY backtest traded:", sorted((str(d), s) for s, d in (nfk - nt)), flush=True)

    # ── 4. fills vs model ─────────────────────────────────────────────────────────────────────────
    fl = Bm[(Bm["status"] == "filled") & (Bm["date"] >= START)].merge(
        Tn[["symbol", "entry_date", "category", "shares", "avg_entry", "legs_filled", "leg_prices", "gross_pnl"]],
        left_on=["symbol", "date"], right_on=["symbol", "entry_date"], how="left")
    fl["matched_model"] = fl["avg_entry"].notna()
    print(f"\n=== 4. filled broker orders Aug18+: {len(fl)} | matched to a backtest trade: {int(fl.matched_model.sum())} ===")
    m = fl[fl["matched_model"]].copy()
    m["slip_vs_model_bps"] = (m["fill_price"] / m["avg_entry"] - 1) * 1e4
    m["slip_vs_orderref_bps"] = (m["fill_price"] / m["ref_price"] - 1) * 1e4
    m["orderref_vs_model_bps"] = (m["ref_price"] / m["avg_entry"] - 1) * 1e4
    for cat, g in [("ALL", m)] + [(c, m[m["category"] == c]) for c in ["C", "B", "A"]]:
        if len(g):
            print(f"  cat {cat}: n={len(g)} | fill vs model 15:21 entry: mean {g.slip_vs_model_bps.mean():+.1f} bps, median {g.slip_vs_model_bps.median():+.1f}, "
                  f"|abs| mean {g.slip_vs_model_bps.abs().mean():.1f} | fill vs order ref: mean {g.slip_vs_orderref_bps.mean():+.1f} bps", flush=True)
    m["ts"] = pd.to_datetime(m["timestamp"])
    print(f"  order timestamps (IST) range: {m.ts.dt.strftime('%H:%M:%S').min()} .. {m.ts.dt.strftime('%H:%M:%S').max()}", flush=True)
    # qty scaling vs backtest shares
    m["qty_over_shares"] = m["quantity"] / m["shares"]
    m["scale_expected"] = m["capital_base"] / 500_000
    print(f"  quantity / backtest shares vs capital_base/5L: median ratio {m.qty_over_shares.median():.3f} vs {m.scale_expected.median():.3f}", flush=True)

    fl_un = fl[~fl["matched_model"]]
    if len(fl_un):
        print("\n  filled live orders with NO matching backtest trade:")
        print(fl_un[["date", "symbol", "quantity", "ref_price", "fill_price", "status"]].to_string(index=False), flush=True)

    with pd.ExcelWriter(OUT / "vb_crosscheck_live_trades.xlsx", engine="openpyxl") as w:
        PS.to_excel(w, sheet_name="signal_set_summary", index=False)
        DAY.to_excel(w, sheet_name="per_day_signal_match", index=False)
        DG.to_excel(w, sheet_name="mismatch_diagnosis", index=False)
        RP.to_excel(w, sheet_name="ref_price_identity", index=False)
        SZ.to_excel(w, sheet_name="list_sizing_rule", index=False)
        EX.to_excel(w, sheet_name="live_diag_vs_scan_jul16_17", index=False)
        jl.drop(columns=["key"]).to_excel(w, sheet_name="list_to_broker", index=False)
        m.drop(columns=["ts"]).to_excel(w, sheet_name="fills_vs_model", index=False)
        no_trade.to_excel(w, sheet_name="my_signals_no_trade", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 30)
    print(f"\nSaved -> {OUT}/vb_crosscheck_live_trades.xlsx")


if __name__ == "__main__":
    main()
