# -*- coding: utf-8 -*-
"""newly_listed_first_results_entry_before_result.py — variant of newly_listed_first_results_backtest.py where the
128 trades that were ENTERED AFTER THE RESULT WAS PUBLIC (during-market filing on a normal trading day D -> entry at
D's close) are moved to be entered BEFORE the result: entry = close of the trading day before D (D-1).

Reaction/exit day is UNCHANGED (user's rule: during_market -> NEXT trading day D+1), so these trades now hold over
D (result day) and exit at D+1 open / close. All other trades are identical to the original run.
Reads results/newly_listed_first_results/first_results_raw.csv (no BSE re-fetch). New folder; original untouched.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import newly_listed_first_results_backtest as N

OUT = rb.RESULTS / "newly_listed_first_results_entry_before_result"; OUT.mkdir(parents=True, exist_ok=True)
SRC = rb.RESULTS / "newly_listed_first_results"


def build(FR, mode):
    """mode: 'orig' (entry D close, after result) | 'prev_close' (D-1 close) | 'open_d' (open of filing day D)."""
    rows1, rows2, issues = [], [], []
    cache = {}
    for _, r in FR.iterrows():
        sym, source, dt = r["symbol"], r["source"], r["announcement_datetime"]
        if sym not in cache:
            cache[sym] = N.load_stock_days(sym, source)
        sd = cache[sym]
        if sd is None or sd.empty:
            issues.append({"symbol": sym, "reason": "no_price_data"}); continue
        tdays = sorted(sd["date"].unique()); tset = set(tdays)
        d0 = dt.date(); sess = N.classify_session(dt); is_td = d0 in tset
        if sess == "pre_market" and is_td:
            reaction = d0
        else:
            later = [d for d in tdays if d > d0]; reaction = later[0] if later else None
        if reaction is None:
            issues.append({"symbol": sym, "reason": "no_trading_day_after_announcement"}); continue
        earlier = [d for d in tdays if d < reaction]
        if not earlier:
            issues.append({"symbol": sym, "reason": "no_trading_day_before_reaction_day"}); continue
        entry = earlier[-1]
        after_result = bool(entry == d0 and sess == "during_market" and is_td)   # entered post-filing (the 128)
        moved = False
        if mode == "prev_close" and after_result:
            prior = [d for d in tdays if d < d0]
            if not prior:
                issues.append({"symbol": sym, "reason": "no_trading_day_before_announcement_day"}); continue
            entry = prior[-1]; moved = True
        elif mode == "open_d" and after_result:
            moved = True                                  # entry stays on D but at the OPEN
        e = sd[sd["date"] == entry]; x = sd[sd["date"] == reaction]
        if e.empty or x.empty:
            issues.append({"symbol": sym, "reason": "missing_entry_or_exit_day_candles"}); continue
        ep = float(e.sort_values("ts")["open"].iloc[0]) if (mode == "open_d" and moved) else float(e.sort_values("ts")["close"].iloc[-1])
        xo = float(x.sort_values("ts")["open"].iloc[0]); xc = float(x.sort_values("ts")["close"].iloc[-1])
        if ep <= 0:
            issues.append({"symbol": sym, "reason": "invalid_entry_price"}); continue
        base = {"symbol": sym, "listing_date": r["listing_date"], "first_result_date": d0,
                "announce_time": dt.strftime("%H:%M"), "session": sess, "reaction_day": reaction, "entry_date": entry,
                "entry_price": ep, "was_after_result_in_original": after_result, "moved_to_before": moved,
                "headline": r["headline"]}
        rows1.append({**base, "exit_price": xo, "return_pct": round((xo - ep) / ep * 100, 3), "holding_days": (reaction - entry).days})
        rows2.append({**base, "exit_price": xc, "return_pct": round((xc - ep) / ep * 100, 3), "holding_days": (reaction - entry).days})
    return pd.DataFrame(rows1), pd.DataFrame(rows2), pd.DataFrame(issues)


def stats(T, label):
    if T.empty:
        return {"group": label, "n": 0}
    x = T["return_pct"]
    return {"group": label, "n": len(T), "win_pct": round((x > 0).mean() * 100, 1), "avg_pct": round(x.mean(), 3),
            "median_pct": round(x.median(), 3), "sum_pct": round(x.sum(), 1),
            "avg_win_pct": round(x[x > 0].mean(), 3) if (x > 0).any() else np.nan,
            "avg_loss_pct": round(x[x <= 0].mean(), 3) if (x <= 0).any() else np.nan,
            "worst_pct": round(x.min(), 2), "best_pct": round(x.max(), 2)}


def main():
    FR = pd.read_csv(SRC / "first_results_raw.csv", parse_dates=["announcement_datetime"])
    print(f"first results: {len(FR)}", flush=True)
    M = {m: build(FR, m) for m in ("orig", "prev_close", "open_d")}
    o1, o2, _ = M["orig"]; n1, n2, DI = M["open_d"]
    print(f"ORIGINAL trades: {len(o1)} | after-result: {int(o1['was_after_result_in_original'].sum())} | "
          f"NEW(open_d) trades: {len(n1)} | moved: {int(n1['moved_to_before'].sum())} | issues: {len(DI)}", flush=True)

    R = []
    for k, nm in ((0, "MORNING_OPEN_EXIT"), (1, "CLOSING_EXIT")):
        for mode, lab in (("orig", "ORIGINAL (entry filing-day close, after result)"), ("prev_close", "D-1 close (prev version)"),
                          ("open_d", "NEW: filing-day OPEN")):
            T = M[mode][k]
            R += [stats(T, f"{nm} | {lab} | ALL"), stats(T[T["was_after_result_in_original"]], f"{nm} | {lab} | the 128")]
    S = pd.DataFrame(R)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S.to_string(index=False), flush=True)

    def bucket(t):
        h, m = map(int, t.split(":")); v = h * 60 + m
        return "09:15-11:00" if v < 660 else "11:00-13:00" if v < 780 else "13:00-14:30" if v < 870 else "14:30-15:30"
    B = []
    for nm, n in (("OPEN", n1), ("CLOSE", n2)):
        m = n[n["moved_to_before"]].copy(); m["bucket"] = m["announce_time"].map(bucket)
        for b, g in m.groupby("bucket"):
            B.append({"exit": nm, **stats(g, b)})
    Bk = pd.DataFrame(B); print("\n128 moved trades by announcement time:\n" + Bk.to_string(index=False), flush=True)

    with pd.ExcelWriter(OUT / "newly_listed_first_results_entry_at_filing_day_open.xlsx", engine="openpyxl") as w:
        pd.DataFrame([{"note": "The 128 during-market filings on trading days (originally entered at the filing-day close, AFTER the result was public) are re-entered at the OPEN of the filing day itself (before the result, which came out during market hours). Reaction/exit day unchanged (next trading day after filing). All other trades identical to original."}]).to_excel(w, sheet_name="READ_ME_FIRST", index=False)
        S.to_excel(w, sheet_name="Summary", index=False)
        Bk.to_excel(w, sheet_name="Moved128_by_time", index=False)
        n1.to_excel(w, sheet_name="Trades_MorningOpen", index=False)
        n2.to_excel(w, sheet_name="Trades_Closing", index=False)
        if len(DI):
            DI.to_excel(w, sheet_name="Data_Issues", index=False)
        for sh in w.sheets.values():
            for c in sh.columns:
                width = max((len(str(x.value)) for x in c if x.value is not None), default=10)
                sh.column_dimensions[c[0].column_letter].width = min(width + 2, 34)
    print(f"\nSaved -> {OUT}/newly_listed_first_results_entry_at_filing_day_open.xlsx", flush=True)


if __name__ == "__main__":
    main()
