#!/usr/bin/env python3
"""
zerodha_auth.py — Shared Kite Connect authentication with daily token caching
==============================================================================
Cache file: zerodha_token_cache.json (same directory as this file)
  {"date": "YYYY-MM-DD", "access_token": "..."}

First run of the day  → full login flow, saves token to cache
Subsequent runs today → loads cache, verifies with kite.profile(), skips login
Next calendar day     → cache date mismatch → fresh login
"""

import json
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from kiteconnect import KiteConnect

CACHE_PATH = Path(__file__).resolve().parent / "zerodha_token_cache.json"


def load_cached_token() -> str | None:
    """Return today's cached access_token, or None if absent/stale."""
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text())
        if data.get("date") == date.today().isoformat():
            return data.get("access_token")
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def save_token_cache(access_token: str) -> None:
    """Write today's token to the cache file."""
    CACHE_PATH.write_text(
        json.dumps({"date": date.today().isoformat(), "access_token": access_token},
                   indent=2)
    )


def authenticate(api_key: str, api_secret: str) -> KiteConnect:
    """
    Return an authenticated KiteConnect instance.

    1. If a valid cached token exists for today, verify it with kite.profile()
       and return immediately — no login URL, no redirect prompt.
    2. Otherwise, open the login URL, prompt for the redirect URL, extract
       the request_token, generate a session, cache the token, and return.
    """
    kite = KiteConnect(api_key=api_key)

    cached = load_cached_token()
    if cached:
        kite.set_access_token(cached)
        try:
            kite.profile()
            print("Reusing cached access token from earlier today - no login needed.")
            return kite
        except Exception:
            print("Cached token is invalid or expired — triggering fresh login.")

    # Full login flow
    print("\nStep 1: Open this URL in your browser and complete the Zerodha login:")
    print(f"\n  {kite.login_url()}\n")
    redirect_url = input("Step 2: Paste the full URL you were redirected to: ").strip()

    params = parse_qs(urlparse(redirect_url).query)
    if "request_token" not in params:
        raise ValueError(
            "No 'request_token' found in the URL you pasted.\n"
            "Make sure you copied the full redirect URL after login."
        )
    request_token = params["request_token"][0]

    session_data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session_data["access_token"]
    kite.set_access_token(access_token)
    save_token_cache(access_token)
    print("Login successful. Token cached for the rest of today.\n")
    return kite
