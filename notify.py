#!/usr/bin/env python3
"""
notify.py — NSE pipeline email notification
============================================
Called by run_daily_pipeline.sh after every run (success or failure).

SETUP (one-time):
  1. Enable 2-Step Verification on your Gmail account.
  2. Go to: Google Account > Security > 2-Step Verification > App passwords
  3. Create an app password for "Mail" — you get a 16-character code.
  4. Paste the 16-char code into SENDER_APP_PASSWORD below.

Usage (called by run_daily_pipeline.sh automatically):
  python3 notify.py \
      --log          logs/pipeline_log_YYYY-MM-DD.log \
      --date         YYYY-MM-DD \
      --status       success | failed \
      --failed-step  "data_loading.py" | "prepare_data.py" | "" \
      --mcap-status  fresh | stale | failed \
      --start-ts     1752600000   (unix timestamp, for runtime calc)
"""

import argparse
import csv
import html as html_lib
import json
import re
import smtplib
import sys
import time
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SENDER_EMAIL        = "rashilshah2@gmail.com"
SENDER_APP_PASSWORD = "bmvxiiajotkpkjvv"
RECIPIENT_EMAILS    = [
    "paramshah1510@gmail.com",
    "khannakartik145@gmail.com",
    "kushalcchauhan88@gmail.com",
]

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent


def _load_mcap_status() -> dict:
    """Read market_cap_daily/mcap_status.json written by fetch_market_cap.py."""
    path = PROJECT_DIR / "market_cap_daily" / "mcap_status.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# ── Log parser ────────────────────────────────────────────────────────────────

def parse_log(log_path: Path, date_str: str) -> dict:
    info = dict(
        symbols_fetched = None,
        new_candles     = None,
        no_data_count   = None,
        n_signals       = None,
        dl_runtime_s    = None,
        pd_runtime_s    = None,
        error_lines     = [],
    )

    if not log_path.exists():
        return info

    text = log_path.read_text(errors="replace")

    m = re.search(r"Symbols fetched with data\s*:\s*([\d,]+)", text)
    if m:
        info["symbols_fetched"] = int(m.group(1).replace(",", ""))

    m = re.search(r"New candles added today\s*:\s*([\d,]+)", text)
    if m:
        info["new_candles"] = int(m.group(1).replace(",", ""))

    m = re.search(r"No data today\s*:\s*([\d,]+)", text)
    if m:
        info["no_data_count"] = int(m.group(1).replace(",", ""))

    m = re.search(r"Done in ([\d.]+)s", text)
    if m:
        info["dl_runtime_s"] = float(m.group(1))

    m = re.search(r"TODAY'S TRADE LIST \(" + re.escape(date_str) + r"\): (\d+) signal", text)
    if m:
        info["n_signals"] = int(m.group(1))
    elif re.search(r"No signals today \(" + re.escape(date_str) + r"\)", text):
        info["n_signals"] = 0

    m = re.search(r"DONE\s+([\d.]+)s", text)
    if m:
        info["pd_runtime_s"] = float(m.group(1))

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.search(r"FAILED|Traceback|Error:|Exception:", line):
            start = max(0, i - 1)
            end   = min(len(lines), i + 40)
            info["error_lines"] = lines[start:end]
            break

    return info


# ── Trade table builders ──────────────────────────────────────────────────────

def _text_trade_table(path: Path) -> str:
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        return "No trades triggered today."
    header = f"{'Symbol':<12} {'Entry ₹':>10}  {'MCap Cr':>8}  {'Vol×':>5}  {'Ret%':>6}  {'Fade%':>5}"
    sep    = "─" * len(header)
    lines  = [header, sep]
    for row in rows:
        lines.append(
            f"{row.get('symbol',''):<12}"
            f" ₹{float(row.get('entry_price_315pm', 0)):>9,.2f}"
            f"  {float(row.get('market_cap_value', 0)):>8,.0f}"
            f"  {float(row.get('volume_ratio', 0)):>5.2f}×"
            f"  +{float(row.get('return_pct_vs_prev_close', 0)):>4.2f}%"
            f"  {float(row.get('fade_at_entry_pct', 0)):>4.2f}%"
        )
    return "\n".join(lines)


def _html_trade_table(path: Path) -> str:
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        return "<p><strong>No trades triggered today.</strong></p>"

    th_style = "background:#1F497D; color:white; padding:6px 10px; text-align:right;"
    th_l     = "background:#1F497D; color:white; padding:6px 10px; text-align:left;"
    body = (
        '<table border="0" cellpadding="6" cellspacing="0" '
        'style="border-collapse:collapse; font-family:Courier New,monospace; '
        'font-size:13px; margin:8px 0;">\n'
        f'<thead><tr>'
        f'<th style="{th_l}">Symbol</th>'
        f'<th style="{th_style}">Entry ₹ (3:15pm)</th>'
        f'<th style="{th_style}">MCap Cr</th>'
        f'<th style="{th_style}">Vol Ratio</th>'
        f'<th style="{th_style}">Return %</th>'
        f'<th style="{th_style}">Fade %</th>'
        '</tr></thead>\n<tbody>\n'
    )
    for i, row in enumerate(rows):
        bg   = "#EAF2FF" if i % 2 == 0 else "#FFFFFF"
        td   = f'style="padding:5px 10px; background:{bg}; text-align:right;"'
        td_l = f'style="padding:5px 10px; background:{bg}; text-align:left;"'
        body += (
            f'<tr>'
            f'<td {td_l}><strong>{row.get("symbol","")}</strong></td>'
            f'<td {td}>₹{float(row.get("entry_price_315pm",0)):,.2f}</td>'
            f'<td {td}>{float(row.get("market_cap_value",0)):,.0f}</td>'
            f'<td {td}>{float(row.get("volume_ratio",0)):.2f}×</td>'
            f'<td {td}>+{float(row.get("return_pct_vs_prev_close",0)):.2f}%</td>'
            f'<td {td}>{float(row.get("fade_at_entry_pct",0)):.2f}%</td>'
            '</tr>\n'
        )
    body += "</tbody></table>"
    return body


def _runtime_str(start_ts, dl_s, pd_s) -> str:
    if start_ts:
        total = int(time.time() - start_ts)
    elif dl_s is not None and pd_s is not None:
        total = int(dl_s + pd_s)
    else:
        return "n/a"
    m, s = divmod(total, 60)
    return f"{m}m {s}s"


def _mcap_warning_text(mcap_status: str, mcap_st: dict) -> str:
    """Plain-text stale/failed market cap warning, or empty string."""
    if mcap_status == "stale":
        date = mcap_st.get("fallback_date", "unknown date")
        return (
            f"\n⚠ WARNING: Today's market cap data could not be fetched live.\n"
            f"  Using data from {date} instead.\n"
            f"  Trade list below may be less accurate than usual.\n"
        )
    if mcap_status == "failed":
        return (
            f"\n⚠ WARNING: Market cap fetch failed with no fallback.\n"
            f"  Today's signals used semi-annual NSE snapshot data.\n"
            f"  Treat market cap values with extra caution.\n"
        )
    return ""


def _mcap_warning_html(mcap_status: str, mcap_st: dict) -> str:
    """HTML stale/failed market cap warning banner, or empty string."""
    if mcap_status == "stale":
        date = mcap_st.get("fallback_date", "unknown date")
        return (
            '<div style="background:#FFF3CD;border:1px solid #FFCC00;padding:12px 16px;'
            'margin-bottom:16px;border-radius:4px;">'
            '<strong style="color:#856404;">⚠ Stale Market Cap Data</strong><br>'
            f'Today\'s live Screener.in export failed. Market cap values are from '
            f'<strong>{date}</strong>. '
            'Trade signals below may be less accurate than usual — verify before executing.'
            '</div>'
        )
    if mcap_status == "failed":
        return (
            '<div style="background:#FFF3CD;border:1px solid #FFCC00;padding:12px 16px;'
            'margin-bottom:16px;border-radius:4px;">'
            '<strong style="color:#856404;">⚠ Market Cap Data Unavailable</strong><br>'
            'Screener.in fetch failed with no fallback. Market cap values are from '
            'semi-annual NSE snapshots (may be months old). '
            'Verify market caps manually before executing.'
            '</div>'
        )
    return ""


# ── Email builders ────────────────────────────────────────────────────────────

def build_success_email(date_str: str, info: dict, trade_list_path: Path,
                        start_ts, mcap_status: str = "fresh") -> MIMEMultipart:
    n        = info.get("n_signals") or 0
    runtime  = _runtime_str(start_ts, info.get("dl_runtime_s"), info.get("pd_runtime_s"))
    sym      = f"{info['symbols_fetched']:,}" if info.get("symbols_fetched") else "n/a"
    cndles   = f"{info['new_candles']:,}"    if info.get("new_candles")     else "n/a"
    mcap_st  = _load_mcap_status()

    subject = f"Trading Pipeline SUCCESS — {date_str} — {n} signal{'s' if n != 1 else ''}"
    if mcap_status == "stale":
        subject += " [STALE MCAP]"

    has_trades = trade_list_path.exists() and trade_list_path.stat().st_size > 50

    warn_text = _mcap_warning_text(mcap_status, mcap_st)
    warn_html = _mcap_warning_html(mcap_status, mcap_st)

    # ── Plain text ────────────────────────────────────────────────────────────
    text = "\n".join([
        f"NSE Volume Breakout Pipeline — {date_str}",
        "=" * 52,
        "",
        f"Status            : SUCCESS",
        f"Total runtime     : {runtime}",
        f"Symbols fetched   : {sym}",
        f"New candles added : {cndles}",
        f"Signals today     : {n}",
        warn_text,
        "Trade signals:",
        _text_trade_table(trade_list_path) if has_trades else "No trades triggered today.",
        "",
        "CSV attached." if has_trades else "",
        "",
        f"Full log: logs/pipeline_log_{date_str}.log",
    ])

    # ── HTML ──────────────────────────────────────────────────────────────────
    trade_html  = _html_trade_table(trade_list_path) if has_trades else \
                  "<p><strong>No trades triggered today.</strong></p>"
    attach_note = "<p style='font-size:12px;color:#555;'>Trade list CSV attached.</p>" \
                  if has_trades else ""

    html = f"""<html><body style="font-family:Calibri,Arial,sans-serif;color:#222;max-width:720px;">
<h2 style="color:#1F497D;margin-bottom:4px;">NSE Pipeline — SUCCESS</h2>
<p style="color:#555;margin-top:0;">{date_str}</p>
{warn_html}
<table style="font-size:14px;margin-bottom:20px;border-spacing:0;">
  <tr><td style="padding:3px 20px 3px 0;color:#555;">Status</td>
      <td><strong style="color:green;">SUCCESS</strong></td></tr>
  <tr><td style="padding:3px 20px 3px 0;color:#555;">Runtime</td>
      <td>{runtime}</td></tr>
  <tr><td style="padding:3px 20px 3px 0;color:#555;">Symbols fetched</td>
      <td>{sym}</td></tr>
  <tr><td style="padding:3px 20px 3px 0;color:#555;">New candles added</td>
      <td>{cndles}</td></tr>
  <tr><td style="padding:3px 20px 3px 0;color:#555;">Signals today</td>
      <td><strong>{n}</strong></td></tr>
</table>
<h3 style="color:#2E74B5;margin-bottom:6px;">Trade Signals</h3>
{trade_html}
{attach_note}
<p style="font-size:11px;color:#aaa;margin-top:32px;border-top:1px solid #eee;padding-top:8px;">
  NSE Volume Breakout — LB=36, VM=6, Split Exit<br>
  3 conditions: Market Cap ₹1,500–5,000 Cr · Volume ≥6× avg · Return ≥5% vs prev VWAP
</p>
</body></html>"""

    outer = MIMEMultipart("mixed")
    outer["Subject"] = subject
    outer["From"]    = SENDER_EMAIL
    outer["To"]      = ", ".join(RECIPIENT_EMAILS)

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text, "plain", "utf-8"))
    alt.attach(MIMEText(html, "html",  "utf-8"))
    outer.attach(alt)

    if has_trades:
        with open(trade_list_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f"attachment; filename={trade_list_path.name}")
        outer.attach(part)

    return outer


def build_failure_email(date_str: str, failed_step: str, info: dict,
                        log_path: Path, start_ts,
                        mcap_status: str = "fresh") -> MIMEMultipart:
    runtime     = _runtime_str(start_ts, info.get("dl_runtime_s"), info.get("pd_runtime_s"))
    subject     = f"Trading Pipeline FAILED — {date_str}"
    error_block = "\n".join(info.get("error_lines", [])) or "(no error detail captured in log)"
    mcap_st     = _load_mcap_status()
    warn_text   = _mcap_warning_text(mcap_status, mcap_st)
    warn_html   = _mcap_warning_html(mcap_status, mcap_st)

    # ── Plain text ────────────────────────────────────────────────────────────
    text = "\n".join([
        f"NSE Volume Breakout Pipeline — {date_str}",
        "=" * 52,
        "",
        f"Status       : FAILED",
        f"Failed step  : {failed_step or 'unknown'}",
        f"Runtime      : {runtime}",
        warn_text,
        "Error detail:",
        "-" * 40,
        error_block,
        "-" * 40,
        "",
        "Full log attached.",
        "",
        "To retry manually:",
        f"  cd '/Users/rashilshah/Desktop/Volume '",
        f"  python3 fetch_market_cap.py && python3 data_loading.py && python3 prepare_data.py",
    ])

    # ── HTML ──────────────────────────────────────────────────────────────────
    escaped_error = html_lib.escape(error_block)
    html = f"""<html><body style="font-family:Calibri,Arial,sans-serif;color:#222;max-width:720px;">
<h2 style="color:#C00000;margin-bottom:4px;">NSE Pipeline — FAILED</h2>
<p style="color:#555;margin-top:0;">{date_str}</p>
{warn_html}
<table style="font-size:14px;margin-bottom:20px;border-spacing:0;">
  <tr><td style="padding:3px 20px 3px 0;color:#555;">Status</td>
      <td><strong style="color:red;">FAILED</strong></td></tr>
  <tr><td style="padding:3px 20px 3px 0;color:#555;">Failed step</td>
      <td><strong>{failed_step or "unknown"}</strong></td></tr>
  <tr><td style="padding:3px 20px 3px 0;color:#555;">Runtime</td>
      <td>{runtime}</td></tr>
</table>
<h3 style="color:#C00000;margin-bottom:6px;">Error Detail</h3>
<pre style="background:#FFF0F0;border:1px solid #FFAAAA;padding:14px;
            font-size:12px;font-family:Courier New,monospace;
            white-space:pre-wrap;word-break:break-all;">{escaped_error}</pre>
<p style="margin-top:16px;">Full pipeline log is attached.</p>
<p style="font-size:12px;color:#555;">
  To retry:<br>
  <code>cd '/Users/rashilshah/Desktop/Volume '</code><br>
  <code>python3 fetch_market_cap.py &amp;&amp; python3 data_loading.py &amp;&amp; python3 prepare_data.py</code>
</p>
<p style="font-size:11px;color:#aaa;margin-top:32px;border-top:1px solid #eee;padding-top:8px;">
  NSE Volume Breakout — LB=36, VM=6
</p>
</body></html>"""

    outer = MIMEMultipart("mixed")
    outer["Subject"] = subject
    outer["From"]    = SENDER_EMAIL
    outer["To"]      = ", ".join(RECIPIENT_EMAILS)

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text, "plain", "utf-8"))
    alt.attach(MIMEText(html, "html",  "utf-8"))
    outer.attach(alt)

    if log_path.exists():
        with open(log_path, "rb") as f:
            part = MIMEBase("text", "plain")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f"attachment; filename={log_path.name}")
        outer.attach(part)

    return outer


# ── SMTP sender ───────────────────────────────────────────────────────────────

def send_email(msg: MIMEMultipart) -> None:
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, msg.as_string())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Send NSE pipeline result email")
    parser.add_argument("--log",          required=True,  help="Path to day's log file")
    parser.add_argument("--date",         required=True,  help="IST date YYYY-MM-DD")
    parser.add_argument("--status",       required=True,  choices=["success", "failed"])
    parser.add_argument("--failed-step",  default="",     help="Which step failed (if any)")
    parser.add_argument("--mcap-status",  default="fresh",
                        choices=["fresh", "stale", "failed"],
                        help="Market cap data freshness from fetch_market_cap.py")
    parser.add_argument("--start-ts",     type=float, default=None,
                        help="Unix timestamp of pipeline start (for runtime calc)")
    args = parser.parse_args()

    log_path        = Path(args.log)
    trade_list_path = PROJECT_DIR / "results" / f"trade_list_{args.date}.csv"
    info            = parse_log(log_path, args.date)

    if args.status == "success":
        msg = build_success_email(args.date, info, trade_list_path,
                                  args.start_ts, args.mcap_status)
    else:
        msg = build_failure_email(args.date, args.failed_step, info,
                                  log_path, args.start_ts, args.mcap_status)

    try:
        send_email(msg)
        print(f"Email sent → {', '.join(RECIPIENT_EMAILS)}  |  {msg['Subject']}")
    except Exception as e:
        print(f"ERROR: email send failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
