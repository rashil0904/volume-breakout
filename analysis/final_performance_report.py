# -*- coding: utf-8 -*-
"""
final_performance_report.py
===========================
Comprehensive GROSS + NET performance report for conditional_split_best_t2
(t1=09:45, t2=12:00) + 14% profit target. One workbook, 6 sheets. Every % return
is paired with ₹ pnl; every metric has a gross and a net (0.23% expense) version.

Fixed-base tables use the existing per-trade sizing (1_Standard, ₹1L-capped).
Quarterly/Yearly also carry a COMPOUNDED track: per_trade_alloc starts ₹100,000,
pool ₹500,000; both scale by the same factor every period — the prior period's ENDING
value carries forward (scale UP on a gain, DOWN on a loss; loss baked in). No reset to
base, no flat carry. Yearly factor = avg of prior year's two GROSS half-year fixed-base
returns.
"""

import sys, bisect
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb
import exit_time_sweep as ets
import profit_target_sweep as pts

IST = ets.IST
OUTDIR = rb.RESULTS / "final_report"
XLSX = OUTDIR / "final_performance_report.xlsx"
TARGET = 14.0
EXPENSE = 0.0023          # 0.23% of capital deployed (net@0.23%)
EXPENSE_038 = 0.0038      # 0.38% of capital deployed (net@0.38%)
BASE_POOL = 500_000
BASE_ALLOC = 100_000
RF_ANNUAL = 0.075
CIRCUIT_THRESHOLD = 19.95   # exclude trades with return_at_entry_pct >= this (near 20% circuit)


def hm_str(hm):
    return f"{int(hm)//60:02d}:{int(hm)%60:02d}" if not (isinstance(hm, float) and np.isnan(hm)) else ""


def fetch_ohlc_and_exitday(base):
    n = len(base)
    opens = np.full((n, len(pts.CANDLE_HMS)), np.nan)
    highs = np.full((n, len(pts.CANDLE_HMS)), np.nan)
    exit_days = [None] * n
    for sym, grp in base.groupby("symbol"):
        pqf = rb.MASTER_DIR / f"{sym}.parquet"
        if not pqf.exists():
            continue
        raw = pd.read_parquet(pqf)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(IST)
        raw["date"] = raw["timestamp"].dt.date
        raw["hm"] = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
        ds = sorted(raw["date"].unique())
        sub = raw[raw["hm"].isin(pts.CANDLE_HMS)]
        po = sub.pivot_table(index="date", columns="hm", values="open", aggfunc="last").reindex(columns=pts.CANDLE_HMS)
        ph = sub.pivot_table(index="date", columns="hm", values="high", aggfunc="last").reindex(columns=pts.CANDLE_HMS)
        for pos_idx, ed in zip(grp.index, grp["date"]):
            j = bisect.bisect_right(ds, ed.date())
            if j >= len(ds):
                continue
            nd = ds[j]
            exit_days[pos_idx] = nd
            if nd in po.index:
                opens[pos_idx, :] = po.loc[nd].values
                highs[pos_idx, :] = ph.loc[nd].values
    return opens, highs, exit_days


def build_trades():
    base = ets.load_base_positions()
    entry = base["entry"].values.astype(float)
    cap_f = base["cap"].values.astype(float)
    sh_f  = base["shares"].values.astype(float)
    edate = base["date"].values
    print("Fetching next-day OHLC + exit days …")
    opens, highs, exit_days = fetch_ohlc_and_exitday(base)
    pct_high = (highs - entry[:, None]) / entry[:, None] * 100
    t1, t2 = pts.HM_0945, pts.HM_1200
    ot1, ot2 = opens[:, pts.HCOL[t1]], opens[:, pts.HCOL[t2]]
    ret_t1 = (ot1 - entry) / entry * 100
    pre_hms = [hm for hm in pts.CANDLE_HMS if hm < t1]
    bet_hms = [hm for hm in pts.CANDLE_HMS if t1 < hm < t2]

    N = len(base)
    et_a = np.empty(N, dtype=object); hm_a = np.full(N, np.nan); px_a = np.full(N, np.nan)
    valid = np.zeros(N, dtype=bool)
    for i in range(N):
        e = entry[i]; ph = pct_high[i]; et = hm = px = None
        for h in pre_hms:
            v = ph[pts.HCOL[h]]
            if not np.isnan(v) and v >= TARGET:
                et, hm, px = "early_target_pre_t1", h, e * (1 + TARGET / 100); break
        if et is None:
            if not np.isnan(ot1[i]) and ret_t1[i] > 0:
                et, hm, px = "positive_at_t1", t1, ot1[i]
            else:
                for h in bet_hms:
                    v = ph[pts.HCOL[h]]
                    if not np.isnan(v) and v >= TARGET:
                        et, hm, px = "early_target_between_t1_t2", h, e * (1 + TARGET / 100); break
                if et is None and not np.isnan(ot2[i]):
                    et, hm, px = "exit_at_t2_no_target", t2, ot2[i]
        if et is not None:
            et_a[i], hm_a[i], px_a[i], valid[i] = et, hm, px, True

    gret = (px_a - entry) / entry * 100
    vi = np.where(valid)[0]
    ed = pd.to_datetime(edate)
    T = pd.DataFrame({
        "symbol": base["symbol"].values[vi],
        "entry_date": [pd.Timestamp(edate[i]).date() for i in vi],
        "entry_price": entry[vi],
        "shares": sh_f[vi].astype(int),
        "capital_deployed": cap_f[vi],
        "exit_date": [exit_days[i] for i in vi],
        "exit_price": px_a[vi],
        "exit_type": et_a[vi],
        "exit_time": [hm_str(hm_a[i]) for i in vi],
        "gross_ret": gret[vi],
    })
    T["gross_pnl"] = T["capital_deployed"] * T["gross_ret"] / 100
    T["expense"]   = T["capital_deployed"] * EXPENSE
    T["net_pnl"]   = T["gross_pnl"] - T["expense"]
    T["net_ret"]   = T["gross_ret"] - EXPENSE * 100
    T["expense_038"] = T["capital_deployed"] * EXPENSE_038
    T["net_pnl_038"] = T["gross_pnl"] - T["expense_038"]
    T["net_ret_038"] = T["gross_ret"] - EXPENSE_038 * 100
    ts = pd.to_datetime(T["entry_date"])
    T["year"] = ts.dt.year
    T["month"] = ts.dt.strftime("%Y-%m")
    T["quarter"] = ts.dt.year.astype(str) + "Q" + ts.dt.quarter.astype(str)
    T["half_year"] = ts.dt.year.astype(str) + "H" + np.where(ts.dt.month <= 6, "1", "2")
    return T.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


# ── period metric helpers (fixed-base sizing) ────────────────────────────────
def pmetrics(df):
    n = len(df); gp = df["gross_pnl"].sum(); npl = df["net_pnl"].sum(); np38 = df["net_pnl_038"].sum()
    gw = df[df["gross_pnl"] > 0]; gl = df[df["gross_pnl"] <= 0]
    nw = df[df["net_pnl"] > 0];   nl = df[df["net_pnl"] <= 0]
    nw38 = df[df["net_pnl_038"] > 0]
    return {
        "n_trades": n,
        "gross_total_pnl_inr": round(gp, 0),
        "gross_total_return_fixedbase_pct": round(gp / BASE_POOL * 100, 4),
        "net_total_pnl_inr": round(npl, 0),
        "net_total_return_fixedbase_pct": round(npl / BASE_POOL * 100, 4),
        "net038_total_pnl_inr": round(np38, 0),
        "net038_total_return_fixedbase_pct": round(np38 / BASE_POOL * 100, 4),
        "gross_win_rate_pct": round(len(gw) / n * 100, 2) if n else 0,
        "net_win_rate_pct": round(len(nw) / n * 100, 2) if n else 0,
        "net038_win_rate_pct": round(len(nw38) / n * 100, 2) if n else 0,
        "gross_avg_return_per_trade_pct": round(df["gross_ret"].mean(), 4) if n else 0,
        "net_avg_return_per_trade_pct": round(df["net_ret"].mean(), 4) if n else 0,
        "net038_avg_return_per_trade_pct": round(df["net_ret_038"].mean(), 4) if n else 0,
        "gross_median_return_per_trade_pct": round(df["gross_ret"].median(), 4) if n else 0,
        "net_median_return_per_trade_pct": round(df["net_ret"].median(), 4) if n else 0,
        "net038_median_return_per_trade_pct": round(df["net_ret_038"].median(), 4) if n else 0,
    }


def scaled(df, alloc):
    """Returns (gross_pnl_sum, net@0.23%_sum, net@0.38%_sum) at a scaled per-trade alloc."""
    sh = np.floor(alloc / df["entry_price"].values)
    caps = sh * df["entry_price"].values
    g = sh * (df["exit_price"].values - df["entry_price"].values)
    return float(g.sum()), float((g - caps * EXPENSE).sum()), float((g - caps * EXPENSE_038).sum())


DAILY_COLS = ["date", "n_trades",
              "gross_total_pnl_inr", "gross_total_return_fixedbase_pct",
              "net_total_pnl_inr", "net_total_return_fixedbase_pct",
              "net038_total_pnl_inr", "net038_total_return_fixedbase_pct",
              "gross_win_rate_pct", "net_win_rate_pct", "net038_win_rate_pct",
              "gross_avg_return_per_trade_pct", "net_avg_return_per_trade_pct", "net038_avg_return_per_trade_pct",
              "gross_median_return_per_trade_pct", "net_median_return_per_trade_pct", "net038_median_return_per_trade_pct",
              "gross_is_winning_day", "net_is_winning_day", "net038_is_winning_day"]


def ex_circuit_period_sheets(T):
    """Daily/monthly/quarterly/yearly period tables for the trade set with near-circuit
    trades (return_at_entry_pct >= CIRCUIT_THRESHOLD) removed. Same format + CURRENT
    full-compounding rule as the all-trades period sheets. Returns {sheet_name: df}."""
    diag = pd.read_csv(rb.RESULTS / "diagnostic_table.csv",
                       usecols=["symbol", "date", "return_pct_vs_prev_close"], parse_dates=["date"])
    diag["date"] = diag["date"].dt.date
    rmap = diag.set_index(["symbol", "date"])["return_pct_vs_prev_close"].to_dict()
    ra = np.array([rmap.get((s, d), np.nan) for s, d in zip(T["symbol"], T["entry_date"])])
    n_excl = int((ra >= CIRCUIT_THRESHOLD).sum())
    Tx = T[~(ra >= CIRCUIT_THRESHOLD)].copy()
    print(f"  ex-circuit period sheets: excluded {n_excl} circuit trades, {len(Tx):,} surviving")

    daily = pd.DataFrame([
        {"date": k, **pmetrics(g),
         "gross_is_winning_day": bool(g["gross_pnl"].sum() > 0),
         "net_is_winning_day": bool(g["net_pnl"].sum() > 0),
         "net038_is_winning_day": bool(g["net_pnl_038"].sum() > 0)}
        for k, g in Tx.groupby("exit_date")]).sort_values("date")[DAILY_COLS]
    monthly = pd.DataFrame([{"month": k, **pmetrics(g)}
                            for k, g in Tx.groupby("month")]).sort_values("month")

    q_rows = []; pool, alloc = BASE_POOL, BASE_ALLOC
    for q, g in sorted(Tx.groupby("quarter"), key=lambda x: x[0]):
        cg, cn, cn38 = scaled(g, alloc)
        q_rows.append({"quarter": q, **pmetrics(g),
                       "quarterly_allocation_used": round(alloc, 0),
                       "compounded_pool_value": round(pool, 0),
                       "gross_compounded_return_pct": round(cg / pool * 100, 4),
                       "net_compounded_return_pct": round(cn / pool * 100, 4),
                       "net038_compounded_return_pct": round(cn38 / pool * 100, 4),
                       "gross_compounded_pnl_inr": round(cg, 0),
                       "net_compounded_pnl_inr": round(cn, 0),
                       "net038_compounded_pnl_inr": round(cn38, 0)})
        f = 1 + cg / pool; pool *= f; alloc *= f          # full compounding (up or down)
    quarterly = pd.DataFrame(q_rows)

    half_gross = {k: pmetrics(g)["gross_total_return_fixedbase_pct"] for k, g in Tx.groupby("half_year")}
    years = sorted(int(y) for y in Tx["year"].unique())
    y_rows = []; pool, alloc = BASE_POOL, BASE_ALLOC
    for idx, y in enumerate(years):
        if idx > 0:
            py = years[idx - 1]
            hs = [h for h in (half_gross.get(f"{py}H1"), half_gross.get(f"{py}H2")) if h is not None]
            avg_half = float(np.mean(hs)) if hs else 0.0
            f = 1 + avg_half / 100; pool *= f; alloc *= f  # full compounding (up or down)
        g = Tx[Tx["year"] == y]
        cg, cn, cn38 = scaled(g, alloc)
        y_rows.append({"year": y, **pmetrics(g),
                       "yearly_allocation_used": round(alloc, 0),
                       "compounded_pool_value": round(pool, 0),
                       "gross_compounded_return_pct": round(cg / pool * 100, 4),
                       "net_compounded_return_pct": round(cn / pool * 100, 4),
                       "net038_compounded_return_pct": round(cn38 / pool * 100, 4),
                       "gross_compounded_pnl_inr": round(cg, 0),
                       "net_compounded_pnl_inr": round(cn, 0),
                       "net038_compounded_pnl_inr": round(cn38, 0)})
    yearly = pd.DataFrame(y_rows)

    return {"daily_performance_ex_circuit": daily,
            "monthly_performance_ex_circuit": monthly,
            "quarterly_performance_ex_circuit": quarterly,
            "yearly_performance_ex_circuit": yearly}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = build_trades()
    print(f"  trades: {len(T):,}")

    # ── period tables (fixed-base) ──
    monthly = pd.DataFrame([{"month": k, **pmetrics(g)} for k, g in T.groupby("month")]).sort_values("month")
    halfy   = pd.DataFrame([{"half_year": k, **pmetrics(g)} for k, g in T.groupby("half_year")]).sort_values("half_year")
    half_gross = dict(zip(halfy["half_year"], halfy["gross_total_return_fixedbase_pct"]))

    # ── quarterly (fixed-base + compounded, pool driven by gross compounded) ──
    q_rows = []; pool, alloc = BASE_POOL, BASE_ALLOC
    for q, g in sorted(T.groupby("quarter"), key=lambda x: x[0]):
        base_m = pmetrics(g)
        cg, cn, cn38 = scaled(g, alloc)
        q_rows.append({"quarter": q, **base_m,
                       "quarterly_allocation_used": round(alloc, 0),
                       "compounded_pool_value": round(pool, 0),
                       "gross_compounded_return_pct": round(cg / pool * 100, 4),
                       "net_compounded_return_pct": round(cn / pool * 100, 4),
                       "net038_compounded_return_pct": round(cn38 / pool * 100, 4),
                       "gross_compounded_pnl_inr": round(cg, 0),
                       "net_compounded_pnl_inr": round(cn, 0),
                       "net038_compounded_pnl_inr": round(cn38, 0)})
        cr = cg / pool * 100
        # FULL COMPOUNDING — pool & alloc carry this quarter's ENDING value into the next:
        # scale up on a gain, down on a loss (loss baked in). No reset, no flat carry.
        f = 1 + cr / 100; pool *= f; alloc *= f
    quarterly = pd.DataFrame(q_rows)

    # ── yearly (fixed-base + compounded, pool driven by prior year's gross halves) ──
    years = sorted(T["year"].unique())
    y_rows = []; pool, alloc = BASE_POOL, BASE_ALLOC
    for idx, y in enumerate(years):
        if idx > 0:
            py = years[idx - 1]
            hs = [half_gross.get(f"{py}H1"), half_gross.get(f"{py}H2")]
            hs = [h for h in hs if h is not None]
            avg_half = float(np.mean(hs)) if hs else 0.0
            # FULL COMPOUNDING — scale by prior year's avg half-return every year: up when
            # positive, down when negative (loss baked in). No reset, no flat carry.
            f = 1 + avg_half / 100; pool *= f; alloc *= f
        g = T[T["year"] == y]
        base_m = pmetrics(g)
        cg, cn, cn38 = scaled(g, alloc)
        y_rows.append({"year": int(y), **base_m,
                       "yearly_allocation_used": round(alloc, 0),
                       "compounded_pool_value": round(pool, 0),
                       "gross_compounded_return_pct": round(cg / pool * 100, 4),
                       "net_compounded_return_pct": round(cn / pool * 100, 4),
                       "net038_compounded_return_pct": round(cn38 / pool * 100, 4),
                       "gross_compounded_pnl_inr": round(cg, 0),
                       "net_compounded_pnl_inr": round(cn, 0),
                       "net038_compounded_pnl_inr": round(cn38, 0)})
    yearly = pd.DataFrame(y_rows)

    # ── daily series (for Sharpe + winning/losing days) ──
    daily = T.groupby("entry_date").agg(g=("gross_pnl", "sum"), n=("net_pnl", "sum"),
                                        n38=("net_pnl_038", "sum"),
                                        c=("capital_deployed", "sum")).reset_index()
    daily["g_ret"] = daily["g"] / daily["c"] * 100
    daily["n_ret"] = daily["n"] / daily["c"] * 100
    daily["n38_ret"] = daily["n38"] / daily["c"] * 100
    rf_daily = RF_ANNUAL / 252 * 100          # in percent, to match %-daily returns
    def sharpe(series):
        return round((series.mean() - rf_daily) / series.std(ddof=1) * np.sqrt(252), 4)

    # ── Daily win/loss streaks (daily P&L by EXIT day) ──
    de = (T.groupby("exit_date").agg(g=("gross_pnl", "sum"), n=("net_pnl", "sum"),
                                     n38=("net_pnl_038", "sum"))
          .reset_index().sort_values("exit_date"))
    ex_dates = de["exit_date"].tolist()

    def streaks(pnls):
        bw_len = bl_len = 0
        bw = bl = (None, None, 0.0)
        ct = None; cl = 0; cs = ce = None; cp = 0.0
        for d, p in zip(ex_dates, pnls):
            typ = "win" if p > 0 else "los"
            if typ == ct:
                cl += 1; cp += p; ce = d
            else:
                ct, cl, cp, cs, ce = typ, 1, float(p), d, d
            if ct == "win" and cl > bw_len:
                bw_len, bw = cl, (cs, ce, cp)
            if ct == "los" and cl > bl_len:
                bl_len, bl = cl, (cs, ce, cp)
        return bw_len, bw, bl_len, bl

    g_wl, g_w, g_ll, g_l = streaks(de["g"].values)
    n_wl, n_w, n_ll, n_l = streaks(de["n"].values)
    m_wl, m_w, m_ll, m_l = streaks(de["n38"].values)      # net@0.38% streaks
    def per(t):
        return f"{t[0]} -> {t[1]}" if t[0] is not None else "—"

    # ── daily_performance (per exit day) — sheet + source for day-extremes ──
    dcols = DAILY_COLS
    daily_perf = pd.DataFrame([
        {"date": k, **pmetrics(g),
         "gross_is_winning_day": bool(g["gross_pnl"].sum() > 0),
         "net_is_winning_day": bool(g["net_pnl"].sum() > 0),
         "net038_is_winning_day": bool(g["net_pnl_038"].sum() > 0)}
        for k, g in T.groupby("exit_date")
    ]).sort_values("date")[dcols]

    # ── summary (gross | net@0.23% | net@0.38% — three parallel series) ──
    n = len(T)
    gw, gl = T[T["gross_pnl"] > 0], T[T["gross_pnl"] <= 0]
    nw, nl = T[T["net_pnl"] > 0], T[T["net_pnl"] <= 0]
    mw, ml = T[T["net_pnl_038"] > 0], T[T["net_pnl_038"] <= 0]
    def S(metric, gross, net, net038):
        return {"metric": metric, "gross": gross, "net@0.23%": net, "net@0.38%": net038}
    summary_rows = [
        S("total_return_fixedbase_pct", round(T["gross_pnl"].sum()/BASE_POOL*100, 4), round(T["net_pnl"].sum()/BASE_POOL*100, 4), round(T["net_pnl_038"].sum()/BASE_POOL*100, 4)),
        S("total_pnl_inr", round(T["gross_pnl"].sum(), 0), round(T["net_pnl"].sum(), 0), round(T["net_pnl_038"].sum(), 0)),
        S("win_rate_pct", round(len(gw)/n*100, 2), round(len(nw)/n*100, 2), round(len(mw)/n*100, 2)),
        S("n_winning_trades", len(gw), len(nw), len(mw)),
        S("n_losing_trades", len(gl), len(nl), len(ml)),
        S("n_winning_days", int((daily["g"] > 0).sum()), int((daily["n"] > 0).sum()), int((daily["n38"] > 0).sum())),
        S("n_losing_days", int((daily["g"] <= 0).sum()), int((daily["n"] <= 0).sum()), int((daily["n38"] <= 0).sum())),
        S("avg_return_per_trade_pct", round(T["gross_ret"].mean(), 4), round(T["net_ret"].mean(), 4), round(T["net_ret_038"].mean(), 4)),
        S("avg_pnl_per_trade_inr", round(T["gross_pnl"].mean(), 2), round(T["net_pnl"].mean(), 2), round(T["net_pnl_038"].mean(), 2)),
        S("avg_return_winning_trades_pct", round(gw["gross_ret"].mean(), 4), round(nw["net_ret"].mean(), 4), round(mw["net_ret_038"].mean(), 4)),
        S("avg_return_losing_trades_pct", round(gl["gross_ret"].mean(), 4), round(nl["net_ret"].mean(), 4), round(ml["net_ret_038"].mean(), 4)),
        S("median_return_per_trade_pct", round(T["gross_ret"].median(), 4), round(T["net_ret"].median(), 4), round(T["net_ret_038"].median(), 4)),
        S("median_return_winning_trades_pct", round(gw["gross_ret"].median(), 4), round(nw["net_ret"].median(), 4), round(mw["net_ret_038"].median(), 4)),
        S("median_return_losing_trades_pct", round(gl["gross_ret"].median(), 4), round(nl["net_ret"].median(), 4), round(ml["net_ret_038"].median(), 4)),
        S("avg_return_per_month_pct", round(monthly["gross_total_return_fixedbase_pct"].mean(), 4), round(monthly["net_total_return_fixedbase_pct"].mean(), 4), round(monthly["net038_total_return_fixedbase_pct"].mean(), 4)),
        S("avg_return_per_quarter_pct", round(quarterly["gross_total_return_fixedbase_pct"].mean(), 4), round(quarterly["net_total_return_fixedbase_pct"].mean(), 4), round(quarterly["net038_total_return_fixedbase_pct"].mean(), 4)),
        S("avg_return_per_half_year_pct", round(halfy["gross_total_return_fixedbase_pct"].mean(), 4), round(halfy["net_total_return_fixedbase_pct"].mean(), 4), round(halfy["net038_total_return_fixedbase_pct"].mean(), 4)),
        S("avg_return_per_year_pct", round(yearly["gross_total_return_fixedbase_pct"].mean(), 4), round(yearly["net_total_return_fixedbase_pct"].mean(), 4), round(yearly["net038_total_return_fixedbase_pct"].mean(), 4)),
        S("sharpe_ratio", sharpe(daily["g_ret"]), sharpe(daily["n_ret"]), sharpe(daily["n38_ret"])),
        S("max_consecutive_winning_days", g_wl, n_wl, m_wl),
        S("max_consecutive_winning_days_pnl_inr", round(g_w[2], 0), round(n_w[2], 0), round(m_w[2], 0)),
        S("max_consecutive_winning_days_period", per(g_w), per(n_w), per(m_w)),
        S("max_consecutive_losing_days", g_ll, n_ll, m_ll),
        S("max_consecutive_losing_days_pnl_inr", round(g_l[2], 0), round(n_l[2], 0), round(m_l[2], 0)),
        S("max_consecutive_losing_days_period", per(g_l), per(n_l), per(m_l)),
    ]

    # ── best / worst period extremes (fixed-base; gross | net@0.23% | net@0.38%) ──
    def _lbl(v):
        return str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v)

    def add_extremes(rows, df, name, label_col, profit_lbl, loss_lbl):
        gmax = df.loc[df["gross_total_pnl_inr"].idxmax()]; nmax = df.loc[df["net_total_pnl_inr"].idxmax()]
        mmax = df.loc[df["net038_total_pnl_inr"].idxmax()]
        gmin = df.loc[df["gross_total_pnl_inr"].idxmin()]; nmin = df.loc[df["net_total_pnl_inr"].idxmin()]
        mmin = df.loc[df["net038_total_pnl_inr"].idxmin()]
        rows += [
            S(f"highest_profit_{name}_pnl_inr", round(float(gmax["gross_total_pnl_inr"]), 0), round(float(nmax["net_total_pnl_inr"]), 0), round(float(mmax["net038_total_pnl_inr"]), 0)),
            S(f"highest_profit_{name}_return_pct", round(float(gmax["gross_total_return_fixedbase_pct"]), 4), round(float(nmax["net_total_return_fixedbase_pct"]), 4), round(float(mmax["net038_total_return_fixedbase_pct"]), 4)),
            S(profit_lbl, _lbl(gmax[label_col]), _lbl(nmax[label_col]), _lbl(mmax[label_col])),
            S(f"highest_loss_{name}_pnl_inr", round(float(gmin["gross_total_pnl_inr"]), 0), round(float(nmin["net_total_pnl_inr"]), 0), round(float(mmin["net038_total_pnl_inr"]), 0)),
            S(f"highest_loss_{name}_return_pct", round(float(gmin["gross_total_return_fixedbase_pct"]), 4), round(float(nmin["net_total_return_fixedbase_pct"]), 4), round(float(mmin["net038_total_return_fixedbase_pct"]), 4)),
            S(loss_lbl, _lbl(gmin[label_col]), _lbl(nmin[label_col]), _lbl(mmin[label_col])),
        ]
    add_extremes(summary_rows, daily_perf, "day",     "date",    "highest_profit_day_date",  "highest_loss_day_date")
    add_extremes(summary_rows, monthly,    "month",   "month",   "highest_profit_month",     "highest_loss_month")
    add_extremes(summary_rows, quarterly,  "quarter", "quarter", "highest_profit_quarter",   "highest_loss_quarter")
    add_extremes(summary_rows, yearly,     "year",    "year",    "highest_profit_year",      "highest_loss_year")
    summary = pd.DataFrame(summary_rows)

    # ── all_trades sheet (gross | net@0.23% | net@0.38%) ──
    all_trades = T[["entry_date", "symbol", "entry_price", "exit_date", "exit_price", "exit_type",
                    "exit_time", "shares", "capital_deployed", "gross_pnl", "gross_ret",
                    "expense", "net_pnl", "net_ret", "expense_038", "net_pnl_038", "net_ret_038",
                    "quarter", "half_year", "year"]].copy()
    all_trades.columns = ["entry_date", "symbol", "entry_price", "exit_date", "exit_price", "exit_type",
                          "exit_time", "shares", "capital_deployed", "gross_trade_pnl_inr",
                          "gross_trade_return_pct", "expense_023_inr", "net023_trade_pnl_inr",
                          "net023_trade_return_pct", "expense_038_inr", "net038_trade_pnl_inr",
                          "net038_trade_return_pct", "quarter", "half_year", "year"]

    ex_sheets = ex_circuit_period_sheets(T)

    with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        all_trades.to_excel(w, sheet_name="all_trades", index=False)
        daily_perf.to_excel(w, sheet_name="daily_performance", index=False)
        monthly.to_excel(w, sheet_name="monthly_performance", index=False)
        quarterly.to_excel(w, sheet_name="quarterly_performance", index=False)
        halfy.to_excel(w, sheet_name="half_yearly_performance", index=False)
        yearly.to_excel(w, sheet_name="yearly_performance", index=False)
        for name, df in ex_sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
        for sh in w.sheets.values():
            for col in sh.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                sh.column_dimensions[col[0].column_letter].width = min(width + 2, 30)

    # ── console recap ──
    pd.set_option("display.width", 220)
    print("\n=== SUMMARY (gross | net@0.23% | net@0.38%) ===")
    print(summary.to_string(index=False))
    tot = summary[summary["metric"] == "total_return_fixedbase_pct"].iloc[0]
    wr = summary[summary["metric"] == "win_rate_pct"].iloc[0]
    print("\n=== TOP-LINE SENSITIVITY TO EXPENSE ASSUMPTION ===")
    print(f"  {'series':<12}{'total_return_fixedbase_pct':>28}{'win_rate_pct':>16}")
    for lab, key in [("gross", "gross"), ("net@0.23%", "net@0.23%"), ("net@0.38%", "net@0.38%")]:
        print(f"  {lab:<12}{tot[key]:>28}{wr[key]:>16}")
    print(f"  expense drag 0.23% -> 0.38%: total return "
          f"{tot['net@0.23%'] - tot['net@0.38%']:.2f} pts lost")
    print("\n=== YEARLY (fixed-base + compounded, 3 series) ===")
    print(yearly[["year", "n_trades", "gross_total_return_fixedbase_pct", "net_total_return_fixedbase_pct",
                  "net038_total_return_fixedbase_pct", "gross_compounded_return_pct",
                  "net_compounded_return_pct", "net038_compounded_return_pct"]].to_string(index=False))
    print(f"\nSaved → {XLSX}")


if __name__ == "__main__":
    main()
