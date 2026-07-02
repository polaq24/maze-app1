"""GEXBot API Client — real-time GEX/DEX/Walls data."""
import numpy as np
import time
import logging
from datetime import datetime

log = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    requests = None

BASE = "https://api.gex.bot/v2"

class GEXBotClient:
    """Client per GEXBot REST API (Classic tier).
    
    Dà GEX pre-calcolato, zero_gamma, spot real-time, major walls.
    Non dà option chain raw → serve ancora yfinance per Greeks/DEX/PDF.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "maze-terminal/1.0",
            "Accept": "application/json",
        }
        self._cache = {}
        self._cache_ttl = 15

    def _get(self, url: str) -> dict | None:
        if requests is None:
            log.warning("requests non installato — pip install requests")
            return None
        try:
            r = requests.get(url, headers=self._headers, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"GEXBot API error: {e}")
            return None

    def fetch_gex(self, ticker: str) -> dict | None:
        """GET /v2/{ticker}/classic/gex_full con cache 15s."""
        cached = self._cache.get(ticker)
        if cached and (time.time() - cached["ts"]) < self._cache_ttl:
            return cached["data"]
        url = f"{BASE}/{ticker}/classic/gex_full"
        data = self._get(url)
        if data:
            self._cache[ticker] = {"ts": time.time(), "data": data}
        return data

    def fetch_majors(self, ticker: str, weight="vol") -> dict | None:
        """GET /v2/{ticker}/classic/gex/majors — key GEX levels."""
        url = f"{BASE}/{ticker}/classic/gex/majors"
        return self._get(url)

    def get_spot(self, ticker: str) -> float | None:
        d = self.fetch_gex(ticker)
        return d.get("spot") if d else None

    def get_zero_gamma(self, ticker: str) -> float | None:
        d = self.fetch_gex(ticker)
        return d.get("zero_gamma") if d else None

    def gex_per_strike(self, ticker: str, weight="vol") -> dict:
        """GEX per strike (pre-computato da GEXBot).
        
        weight='vol' → GEX pesato sul volume
        weight='oi'  → GEX pesato su OI
        
        Returns dict compatibile con compute_gex().
        """
        d = self.fetch_gex(ticker)
        if not d or "strikes" not in d:
            return {"strikes": np.array([]), "gex": np.array([]),
                    "total_gex": 0, "zero_gamma": 0,
                    "max_gex_strike": 0, "min_gex_strike": 0}

        raw = d["strikes"]
        strikes = np.array([s[0] for s in raw])
        gex_idx = 1 if weight == "vol" else 2
        gex = np.array([s[gex_idx] for s in raw])

        total = d.get(f"sum_gex_{'vol' if weight == 'vol' else 'oi'}", 0)
        zg = d.get("zero_gamma", 0)

        max_i = np.argmax(gex) if len(gex) > 0 else 0
        min_i = np.argmin(gex) if len(gex) > 0 else 0

        return {
            "strikes": strikes,
            "gex": gex,
            "total_gex": total,
            "zero_gamma": zg,
            "spot": d.get("spot", 0),
            "max_gex_strike": strikes[max_i] if len(strikes) > max_i else 0,
            "min_gex_strike": strikes[min_i] if len(strikes) > min_i else 0,
            "major_pos": d.get(f"major_pos_{'vol' if weight == 'vol' else 'oi'}"),
            "major_neg": d.get(f"major_neg_{'vol' if weight == 'vol' else 'oi'}"),
        }

    def clear_cache(self):
        self._cache = {}

