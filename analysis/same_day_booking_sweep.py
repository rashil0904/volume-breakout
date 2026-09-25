# -*- coding: utf-8 -*-
"""same_day_booking_sweep.py — ADDITIVE post-3:21pm same-day profit-booking overlay on the BASELINE
volume-breakout strategy (baseline_and_cross_final.py imported, NOT modified). Overlay: if a carried
position's entry-day HIGH from 3:21pm->close touches avg_entry*(1+X%), book that leg same-day at that limit
level; else carry overnight unchanged. SHORT + costs untouched (short still next-day 9:25/11:59, cover
14:39/5%, per user). Cat A UC-locked measured from filled avg (per user). Sweep X=1..10 + baseline.
Reports PER CATEGORY A/B/C separately + combined + IS(2022-24)/OOS(2025+). Return = pnl/pool*100 (fixed base).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_backtest as rb
import baseline_and_cross_final as B

OUTDIR = rb.RESULTS / "same_day_booking"; OUTDIR.mkdir(parents=True, exist_ok=True)
XS = [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; POOL = B.BASE_POOL


def main():
    import pickle
    pk = OUTDIR / "_cache.pkl"
    if pk.exists():
        cache = pickle.load(open(pk, "rb")); print(f"loaded cache ({len(cache):,} stock-days)")
    else:
        cache = B.build_cache(); pickle.dump(cache, open(pk, "wb"))
    T, _diag = B.run_config("baseline", cache)                    # run_config returns (DataFrame, diag_days)
    eg_lookup = {(c["symbol"], c["entry_date"]): c["eg"] for c in cache}

    def post321(sym, ed):
        eg = eg_lookup.get((sym, ed))
        if eg is None: return np.nan
        m = eg["hm"] >= B.HM_1521
        return float(eg["hi"][m].max()) if m.any() else np.nan
    T["post321_high"] = [post321(s, d) for s, d in zip(T["symbol"], T["entry_date"])]
    T["yr"] = pd.to_datetime(T["entry_date"]).dt.year
    catA_locked = int(((T.category == "A") & (T.cat_a_full == False)).sum())

    def pnl_at(X):
        if X is None:
            booked = pd.Series(False, index=T.index); lpnl = T["long_pnl"].values
        else:
            level = T["avg_entry"].values * (1 + X / 100.0); booked = T["post321_high"].values >= level
            lpnl = np.where(booked, T["shares"].values * T["avg_entry"].values * X / 100.0, T["long_pnl"].values)
        comb = lpnl + T["short_pnl"].values
        return pd.DataFrame({"cat": T["category"].values, "yr": T["yr"].values, "booked": np.asarray(booked, bool),
                             "gross": comb, "netA": comb - T["long_cost_023"].values - T["short_cost"].values,
                             "netB": comb - T["long_cost_038"].values - T["short_cost"].values})

    def agg(D):
        g = D["gross"].values
        return {"trades": len(D), "booked_sameday": int(D["booked"].sum()), "carried_overnight": int((~D["booked"]).sum()),
                "win_%": round((g > 0).mean() * 100, 1) if len(D) else 0,
                "gross_ret_%": round(g.sum() / POOL * 100, 2), "netA_ret_%": round(D["netA"].sum() / POOL * 100, 2), "netB_ret_%": round(D["netB"].sum() / POOL * 100, 2),
                "avg_ret_%/trade": round((g / (T['capital_deployed'].mean())).mean() * 100, 3) if len(D) else 0,
                "avg_gross_booked": round(D["gross"][D["booked"]].mean(), 0) if D["booked"].any() else np.nan,
                "avg_gross_carried": round(D["gross"][~D["booked"]].mean(), 0) if (~D["booked"]).any() else np.nan}

    per = {X: pnl_at(X) for X in XS}
    xl = lambda X: "baseline" if X is None else f"{X}%"

    def cat_table(cat, yrmask=None):
        rows = []
        for X in XS:
            D = per[X]; sub = D[D["cat"] == cat] if cat != "ALL" else D
            if yrmask is not None: sub = sub[yrmask(sub["yr"])]
            rows.append({"booking_X": xl(X), **agg(sub)})
        return pd.DataFrame(rows)

    tables = {c: cat_table(c) for c in ["A", "B", "C", "ALL"]}

    def best_region(df):
        v = df["gross_ret_%"].values; nb = np.array([np.mean(v[max(0, i - 1):i + 2]) for i in range(len(v))])
        bi = int(np.argmax(nb)); reg = [df.booking_X.iloc[j] for j in range(max(0, bi - 1), min(len(v), bi + 2))]
        return df.booking_X.iloc[bi], reg, df.booking_X.iloc[int(np.argmax(v))]

    with pd.ExcelWriter(OUTDIR / "same_day_booking_sweep.xlsx", engine="openpyxl") as w:
        info = [
            {"metric": "Overlay", "value": "ADDITIVE post-3:21pm same-day booking on BASELINE (baseline_and_cross_final untouched)"},
            {"metric": "Rule", "value": "if entry-day 3:21->close HIGH >= avg_entry*(1+X%) -> book that day at avg*(1+X%) (limit); else carry overnight"},
            {"metric": "Short & costs", "value": "UNCHANGED (short next-day 9:25/11:59 by pos/neg, cover 14:39/5%; gross/netA/netB same)"},
            {"metric": "Cat A UC-locked", "value": f"MEASURED FROM FILLED AVG (per user); {catA_locked} Cat-A positions were UC-locked/partial (cat_a_full=False) -> included on partial-fill base"},
            {"metric": "Return unit", "value": "fixed-base % = leg pnl / pool(5L) * 100 (strategy convention); netA=0.23+0.10, netB=0.38+0.10"},
            {"metric": "Baseline (no booking) gross return %", "value": tables['ALL'].iloc[0]['gross_ret_%']},
        ]
        for c in ["A", "B", "C"]:
            bc, reg, bs = best_region(tables[c])
            info.append({"metric": f"[Cat {c}] baseline / best-region", "value": f"baseline {tables[c].iloc[0]['gross_ret_%']}% | best region {reg} (center {bc}) | best single {bs}"})
        pd.DataFrame(info).to_excel(w, sheet_name="Summary", index=False)
        for c in ["A", "B", "C"]:
            tables[c].to_excel(w, sheet_name=f"Category_{c}", index=False)
            IS = cat_table(c, lambda y: y <= 2024); OOS = cat_table(c, lambda y: y >= 2025)
            m = IS[["booking_X", "gross_ret_%"]].merge(OOS[["booking_X", "gross_ret_%"]], on="booking_X", suffixes=("_IS", "_OOS"))
            m.to_excel(w, sheet_name=f"Cat{c}_IS_OOS", index=False)
        # combined side-by-side
        comb = pd.DataFrame({"booking_X": [xl(X) for X in XS]})
        for c in ["A", "B", "C", "ALL"]:
            comb[f"{c}_gross_ret%"] = tables[c]["gross_ret_%"].values
            comb[f"{c}_netA_ret%"] = tables[c]["netA_ret_%"].values
        comb.to_excel(w, sheet_name="Combined_AllCats", index=False)

    pd.set_option("display.width", 240)
    print("=" * 96 + "\nSAME-DAY BOOKING OVERLAY SWEEP — per category (additive; baseline untouched)\n" + "=" * 96)
    print(f"trades {len(T)} | Cat A/B/C {int((T.category=='A').sum())}/{int((T.category=='B').sum())}/{int((T.category=='C').sum())} | Cat-A UC-locked(partial) {catA_locked}")
    for c in ["A", "B", "C"]:
        print(f"\n--- Category {c} ---"); print(tables[c][["booking_X", "trades", "booked_sameday", "carried_overnight", "win_%", "gross_ret_%", "netA_ret_%", "avg_gross_booked", "avg_gross_carried"]].to_string(index=False))
        bc, reg, bs = best_region(tables[c]); print(f"  best region {reg} (center {bc}) | best single {bs}")
    print("\n--- COMBINED (gross_ret% by category) ---")
    cc = pd.DataFrame({"X": [xl(X) for X in XS]}); [cc.__setitem__(c, tables[c]["gross_ret_%"].values) for c in ["A", "B", "C", "ALL"]]; print(cc.to_string(index=False))
    print(f"\nSaved -> {OUTDIR}")


if __name__ == "__main__":
    main()
