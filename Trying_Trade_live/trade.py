"""
Minimal buy-now / sell-open-positions script. No strategy logic -- you pass
in the symbols and quantities to buy, it places MARKET orders, and remembers
what it bought so you can sell it later with one command.

Usage:
    python trade.py buy RELIANCE:1 TCS:1
    python trade.py sell

Requires the token file to exist (run upstox_auth.py first).
"""

import json
import os
import sys
from datetime import datetime

import requests

import config
from instruments import resolve_symbol
from upstox_auth import load_access_token


def _headers():
    return {
        "Content-Type": "application/json",
        "Accept":        "application/json",
        "Authorization": f"Bearer {load_access_token()}",
    }


def _place_order(instrument_key: str, quantity: int, transaction_type: str) -> dict:
    """
    Places a market order. Returns the parsed response dict on success.
    Raises OrderError with a human-readable message on failure.
    """
    body = {
        "quantity":           quantity,
        "product":            config.PRODUCT_TYPE,
        "validity":           "DAY",
        "price":              0,
        "tag":                "manual-3pm-script",
        "instrument_token":   instrument_key,
        "order_type":         "MARKET",
        "transaction_type":   transaction_type,
        "disclosed_quantity": 0,
        "trigger_price":      0,
        "is_amo":             False,
        "slice":              True,
    }

    mode = "SANDBOX" if config.SANDBOX else "LIVE"
    try:
        resp = requests.post(config.ORDER_URL, headers=_headers(), json=body, timeout=10)
    except requests.ConnectionError as e:
        raise OrderError(f"Connection failed to {config.ORDER_URL}: {e}")
    except requests.Timeout:
        raise OrderError(f"Request timed out — order may or may not have been placed. Check the Upstox app.")

    if not resp.ok:
        # Extract Upstox error detail if available
        try:
            err = resp.json()
            msg = err.get("message") or err.get("errors") or resp.text
        except Exception:
            msg = resp.text
        raise OrderError(
            f"[{mode}] Order rejected (HTTP {resp.status_code}): {msg}\n"
            f"  endpoint:   {config.ORDER_URL}\n"
            f"  instrument: {instrument_key}\n"
            f"  qty:        {quantity}  type: {transaction_type}"
        )

    return resp.json()


class OrderError(Exception):
    pass


def _load_positions() -> list:
    if os.path.exists(config.POSITIONS_FILE):
        with open(config.POSITIONS_FILE) as f:
            return json.load(f)
    return []


def _save_positions(positions: list):
    # Write to a temp file first, then rename — prevents corruption if we crash mid-write
    tmp = config.POSITIONS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(positions, f, indent=2)
    os.replace(tmp, config.POSITIONS_FILE)


def buy(symbol_qty_pairs):
    mode = "SANDBOX" if config.SANDBOX else "LIVE"
    print(f"=== BUY  [{mode}] ===")

    positions    = _load_positions()
    placed       = []
    failed       = []

    for symbol, qty in symbol_qty_pairs:
        try:
            instrument_key = resolve_symbol(symbol)
        except ValueError as e:
            print(f"  SKIP {symbol}: {e}")
            failed.append(symbol)
            continue

        print(f"  Placing BUY: {symbol} x{qty}  ({instrument_key})")
        try:
            result   = _place_order(instrument_key, qty, "BUY")
            order_id = result.get("data", {}).get("order_ids", [None])[0]
            print(f"    ✓  order_id: {order_id}")
            positions.append({
                "symbol":         symbol,
                "instrument_key": instrument_key,
                "quantity":       qty,
                "buy_order_id":   order_id,
                "buy_date":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sandbox":        config.SANDBOX,
                "status":         "open",
            })
            placed.append(symbol)
        except OrderError as e:
            print(f"    ✗  FAILED — {e}")
            failed.append(symbol)

    if placed:
        _save_positions(positions)
        print(f"\n  Saved {len(placed)} position(s) to {config.POSITIONS_FILE}")
    if failed:
        print(f"  {len(failed)} order(s) failed and were NOT saved: {failed}")


def sell_all_open():
    mode = "SANDBOX" if config.SANDBOX else "LIVE"
    print(f"=== SELL  [{mode}] ===")

    positions      = _load_positions()
    open_positions = [p for p in positions if p.get("status") == "open"]

    if not open_positions:
        print("  No open positions to sell.")
        return

    for pos in open_positions:
        print(f"  Placing SELL: {pos['symbol']} x{pos['quantity']}")
        try:
            result   = _place_order(pos["instrument_key"], pos["quantity"], "SELL")
            order_id = result.get("data", {}).get("order_ids", [None])[0]
            print(f"    ✓  order_id: {order_id}")
            pos["sell_order_id"] = order_id
            pos["sell_date"]     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pos["status"]        = "closed"
        except OrderError as e:
            print(f"    ✗  FAILED — {e}")
            # Leave status as "open" so it can be retried

    _save_positions(positions)
    closed = sum(1 for p in open_positions if p.get("status") == "closed")
    print(f"\n  Closed {closed} of {len(open_positions)} position(s).")


def show_positions():
    positions = _load_positions()
    if not positions:
        print("No positions recorded.")
        return
    print(f"{'Symbol':<12} {'Qty':>5}  {'Status':<8}  {'Buy date':<20}  {'Buy order'}")
    print("─" * 70)
    for p in positions:
        print(f"  {p['symbol']:<10} {p['quantity']:>5}  {p['status']:<8}  "
              f"{p.get('buy_date',''):<20}  {p.get('buy_order_id','')}")


def _parse_symbol_qty_args(args):
    pairs = []
    for arg in args:
        if ":" not in arg:
            print(f"Bad argument '{arg}' — expected SYMBOL:QTY format (e.g. RELIANCE:1)")
            sys.exit(1)
        symbol, qty = arg.split(":", 1)
        try:
            pairs.append((symbol.upper(), int(qty)))
        except ValueError:
            print(f"Bad quantity in '{arg}' — qty must be an integer")
            sys.exit(1)
    return pairs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "buy":
        if len(sys.argv) < 3:
            print("Usage: python trade.py buy SYMBOL:QTY [SYMBOL:QTY ...]")
            sys.exit(1)
        buy(_parse_symbol_qty_args(sys.argv[2:]))

    elif command == "sell":
        sell_all_open()

    elif command == "positions":
        show_positions()

    else:
        print(f"Unknown command '{command}'. Use: buy | sell | positions")
        sys.exit(1)
