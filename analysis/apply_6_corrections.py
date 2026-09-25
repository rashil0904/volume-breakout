# -*- coding: utf-8 -*-
"""apply_6_corrections.py — remediate the 6 flagged results events in the stored BSE announcement dataset.
Starts from the pre-fix backup (idempotent). Two action types:
  DROP  — spurious related-party/Reg-23(9) row that shadows an already-correct results row (PEAD keeps
          earliest, so removing the wrong early row makes the correct later row the event).
  EDIT  — single-row events pointing at the wrong filing -> repoint to the verified Board-Meeting/Result
          declaration, recomputing session/reaction fields with the dataset's conventions.
Writes corrected CSV + changelog. Backup is created once and never overwritten.
"""
import sys, bisect
from pathlib import Path
from datetime import datetime
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_backtest as rb

CSV = rb.RESULTS / "bse_results_announcements" / "bse_results_announcements.csv"
BACKUP = CSV.with_name("bse_results_announcements_prebugfix_backup.csv")
CHANGELOG = rb.RESULTS / "bse_verify" / "corrections_changelog.csv"
DAILY = rb.BASE / "data" / "daily_ohlcv_all.parquet"

# DROP: (symbol, quarter, substring in headline identifying the WRONG row to remove)
DROPS = [
    ("POLYCAB", "Q3FY23", "Related Party"),                 # keeps 19-01 Board Meeting Outcome (already correct)
    ("RVNL",    "Q1FY23", "Compliance With Regulation 23"), # keeps 10-08 Unaudited Financial Results
    ("YESBANK", "Q3FY23", "Related Party"),                 # keeps 21-01 Outcome Of The Board Meeting
    ("CONCOR",  "Q2FY23", "Related Party"),                 # keeps 10-11 (results day); refined below
]
# EDIT: (symbol, quarter) -> (new datetime, headline, category)  [row must be unique after drops]
EDITS = {
    ("CONCOR", "Q2FY23"): ("2022-11-10 18:54:41", "Board Meeting Outcome for Outcome Of Board Meeting (Q2FY23 Results)", "Board Meeting"),
    ("INDIGO", "Q4FY24"): ("2024-05-23 16:18:00", "Board Meeting Outcome for Audited Financial Results Q4 & FY Mar-31-2024", "Board Meeting"),
    ("GAIL",   "Q4FY24"): ("2024-05-16 14:36:00", "Board Meeting Outcome - Audited Financial Results (Q4FY24)", "Board Meeting"),
}


def classify(dt):
    m = dt.hour * 60 + dt.minute
    return "pre_market" if m < 555 else "during_market" if m <= 930 else "post_market"


def main():
    if not BACKUP.exists():
        print("ERROR: backup missing — refusing to run without a pre-fix backup."); return
    df = pd.read_csv(BACKUP, dtype=str).reset_index(drop=True)   # always start from clean original
    print(f"loaded backup: {len(df)} rows")

    d = pd.read_parquet(DAILY, columns=["date"])
    tdays = sorted(set(pd.to_datetime(d["date"]).dt.date)); tset = set(tdays)

    def next_td(d0):
        i = bisect.bisect_right(tdays, d0)
        return tdays[i] if i < len(tdays) else None

    log = []
    # 1) DROPS
    for sym, q, sub in DROPS:
        m = (df["symbol"] == sym) & (df["quarter"] == q) & (df["headline"].astype(str).str.contains(sub, case=False, na=False))
        if m.sum() != 1:
            print(f"  !! DROP {sym} {q} '{sub}': matched {m.sum()} — skipped"); continue
        r = df[m].iloc[0]
        log.append({"action": "DROP", "symbol": sym, "quarter": q, "old_date": r["announcement_date"],
                    "old_time": r["announcement_time"], "old_session": r["announcement_session"],
                    "old_reaction_date": r["reaction_session_date"], "old_headline": str(r["headline"])[:55],
                    "new_date": "", "new_time": "", "new_session": "", "new_reaction_date": "", "new_headline": "(row removed)"})
        df = df[~m].reset_index(drop=True)
        print(f"  dropped {sym} {q}: {r['announcement_date']} {str(r['headline'])[:45]}")

    # 2) EDITS
    for (sym, q), (newdt, headline, cat) in EDITS.items():
        m = (df["symbol"] == sym) & (df["quarter"] == q)
        if m.sum() != 1:
            print(f"  !! EDIT {sym} {q}: matched {m.sum()} — skipped"); continue
        i = df.index[m][0]; r = df.loc[i]
        old = {k: r[k] for k in ["announcement_date", "announcement_time", "announcement_session", "reaction_session_date", "headline"]}
        dt = datetime.strptime(newdt, "%Y-%m-%d %H:%M:%S"); d0 = dt.date(); sess = classify(dt); is_td = d0 in tset
        if sess == "post_market" or not is_td:
            rdate, rsess = next_td(d0), "next_trading_day"
        else:
            rdate, rsess = d0, "same_day"
        df.loc[i, "announcement_date"] = dt.strftime("%d-%m-%Y")
        df.loc[i, "announcement_time"] = dt.strftime("%H:%M:%S")
        df.loc[i, "announcement_datetime"] = dt.strftime("%d-%m-%Y %H:%M")
        df.loc[i, "announcement_session"] = sess
        df.loc[i, "is_trading_day"] = str(is_td)
        df.loc[i, "reaction_session"] = rsess
        df.loc[i, "reaction_session_date"] = rdate.strftime("%Y-%m-%d") if rdate else ""
        df.loc[i, "headline"] = headline[:80]; df.loc[i, "category"] = cat
        df.loc[i, "source"] = "BSE (verified 2026-08-15)"; df.loc[i, "as_of"] = "2026-08-15"
        log.append({"action": "EDIT", "symbol": sym, "quarter": q, "old_date": old["announcement_date"],
                    "old_time": old["announcement_time"], "old_session": old["announcement_session"],
                    "old_reaction_date": old["reaction_session_date"], "old_headline": str(old["headline"])[:55],
                    "new_date": dt.strftime("%d-%m-%Y"), "new_time": dt.strftime("%H:%M:%S"), "new_session": sess,
                    "new_reaction_date": df.loc[i, "reaction_session_date"], "new_headline": headline[:55]})
        print(f"  edited {sym} {q}: {old['announcement_date']} {old['announcement_session']} -> {dt.strftime('%d-%m-%Y')} {sess}")

    df.to_csv(CSV, index=False)
    LOG = pd.DataFrame(log); LOG.to_csv(CHANGELOG, index=False)
    pd.set_option("display.width", 260)
    print(f"\nrows: {len(df)} (was {len(pd.read_csv(BACKUP))}); dropped {sum(1 for l in log if l['action']=='DROP')}, edited {sum(1 for l in log if l['action']=='EDIT')}")
    print(f"changelog: {CHANGELOG}\ncorrected: {CSV}\n")
    print(LOG[["action", "symbol", "quarter", "old_date", "old_session", "new_date", "new_session", "new_reaction_date", "new_headline"]].to_string(index=False))


if __name__ == "__main__":
    main()
