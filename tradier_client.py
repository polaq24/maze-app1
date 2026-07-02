"""Tradier Brokerage API Client — OPRA real-time options data."""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, date, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

from quant_analytics import enrich_greeks

BASE = "https://api.tradier.com/v1"

try:
    import requests
except ImportError:
    requests = None

class TradierClient:
    def __init__(self, token: str):
        self._token = token
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        self._cache = {}

    def _get(self, url: str, params: dict = None) -> dict | None:
        if requests is None:
            log.warning("requests non installato — pip install requests")
            return None
        try:
            r = requests.get(url, headers=self._headers, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"Tradier API error: {e}")
            return None

    def get_spot(self, ticker: str) -> float | None:
        d = self._get(f"{BASE}/markets/quotes", {"symbols": ticker, "greeks": "false"})
        if d is None:
            return None
        quotes = d.get("quotes")
        if not quotes:
            log.warning(f"Tradier get_spot: nessun 'quotes' in risposta: {str(d)[:200]}")
            return None
        q = quotes.get("quote")
        if not q:
            log.warning(f"Tradier get_spot: nessun 'quote' in quotes: {str(quotes)[:200]}")
            return None
        if isinstance(q, list):
            q = q[0] if q else None
            if not q:
                log.warning(f"Tradier get_spot: lista quote vuota")
                return None
        val = float(q.get("last") or q.get("bid") or q.get("ask") or 0) or None
        if val is None:
            log.warning(f"Tradier get_spot: no last/bid/ask in quote: {str(q)[:200]}")
        return val

    def get_expirations(self, ticker: str) -> list[str]:
        d = self._get(f"{BASE}/markets/options/expirations", {"symbol": ticker, "includeAllRoots": "true"})
        if not d:
            log.warning("Tradier get_expirations: risposta vuota")
            return []
        if "expirations" not in d:
            log.warning(f"Tradier get_expirations: no 'expirations' in risposta: {str(d)[:200]}")
            return []
        exps_data = d["expirations"].get("date")
        if not exps_data:
            log.warning(f"Tradier get_expirations: struttura expirations={str(d['expirations'])[:500]}")
            return []
        exps = [e["date"] if isinstance(e, dict) else e for e in exps_data]
        log.info(f"Tradier expirations for {ticker}: {exps[:5]}... ({len(exps)} total)")
        return exps

    def get_option_chain(self, ticker: str, expiration: str) -> pd.DataFrame:
        d = self._get(f"{BASE}/markets/options/chains", {
            "symbol": ticker, "expiration": expiration, "greeks": "true"
        })
        if not d:
            log.warning(f"Tradier get_option_chain({expiration}): risposta vuota")
            return pd.DataFrame()
        if "options" not in d or "option" not in d.get("options", {}):
            log.warning(f"Tradier get_option_chain({expiration}): struttura inattesa: {str(d)[:300]}")
            return pd.DataFrame()
        raw = d["options"]["option"]
        if isinstance(raw, dict):
            raw = [raw]
        rows = []
        for o in raw:
            g = o.get("greeks", {}) or {}
            iv = (float(g.get("mid_iv") or 0) or float(g.get("vol") or 0)
                  or float(g.get("ask_iv") or 0) or float(g.get("bid_iv") or 0))
            rows.append(dict(
                strike=float(o["strike"]),
                option_type="call" if o.get("option_type","").lower() == "call" else "put",
                bid=float(o.get("bid") or 0),
                ask=float(o.get("ask") or 0),
                last=float(o.get("last") or 0),
                volume=int(o.get("volume") or 0),
                open_interest=int(o.get("open_interest") or 0),
                implied_volatility=iv,
                delta=float(g.get("delta") or 0),
                gamma=float(g.get("gamma") or 0),
                theta=float(g.get("theta") or 0),
                vega=float(g.get("vega") or 0),
            ))
        return pd.DataFrame(rows)

    def load_all(self, ticker: str, max_expiries: int = 10) -> tuple:
        spot = self.get_spot(ticker)
        if not spot:
            log.warning(f"Tradier: spot N/D per {ticker}")
            return None, [], None

        exps = self.get_expirations(ticker)
        today = date.today()
        exps_sorted = sorted(exps, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d").date() - today).days))
        chains = []
        loaded = 0
        for exp in exps_sorted:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            dte = (exp_date.date() - today).days
            if dte < 0:
                continue
            try:
                df = self.get_option_chain(ticker, exp)
                if df.empty:
                    continue
                df["dte"] = dte
                chains.append(df)
                loaded += 1
                if loaded >= max_expiries:
                    break
            except Exception as e:
                log.warning(f"Tradier chain error {exp}: {e}")
                continue
            log.info(f"Tradier loaded {ticker} {exp} (DTE {dte}) — {len(df)} righe")

        chains = [enrich_greeks(c, spot) for c in chains]
        log.info(f"Tradier done: {len(chains)} expiry, {sum(len(c) for c in chains)} opzioni totali")

        return spot, chains, None
