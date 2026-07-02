"""Maze Capital Terminal — real-time WebSocket server."""
import asyncio, json, logging, sqlite3, re
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    _vader = None

from quant_analytics import (
    compute_gex, compute_exposure_profile, enrich_greeks, gamma_walls,
    premium_flow, compute_dex, open_interest_profile, expected_move,
    fetch_macro, gamma_bands as gb_func, gex_regime_heatmap, oi_term_heatmap,
    dom_bookmap, linreg_channels, garch_mc_paths,
    pca_decomposition, ridge_predict,
    cpcv_risk_assessment, ergodic_market_gap,
)
from macro_quant import (
    DataFetcher as MacroFetcher, RegimeDetector,
    RegimeShiftPredictor, CorrelationAnalyzer, Regime, Bias,
)
from tradier_client import TradierClient
from gexbot_client import GEXBotClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
def clean_nan(obj):
    """Sostituisce NaN/Inf con None per JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(x) for x in obj]
    elif isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    return obj

log = logging.getLogger(__name__)
DB_PATH = Path(__file__).parent / "macro_history.db"

# ── FastAPI ─────────────────────────────────────────────────────────────
app = FastAPI(title="Maze Capital Terminal")
connected_clients: set[WebSocket] = set()
_client_filter = {"dte_min": 0, "dte_max": 365, "source": "all", "timeframe": "5d"}
_filter_event = asyncio.Event()

# ── SQLite ──────────────────────────────────────────────────────────────
@contextmanager
def db_conn():
    c = sqlite3.connect(str(DB_PATH))
    try: yield c
    finally: c.close()

# ── Background data ─────────────────────────────────────────────────────
_state = {"spot": 0, "timestamp": "", "payload": None, "ready": False}
_data_ready = asyncio.Event()

async def data_loop():
    tradier = TradierClient("dJoOgKq0zvagRUm1HaNaaVNcQiGq")
    gexbot = None
    try:
        gexbot = GEXBotClient("gexbot_custom_rnOqKSpAUAYL5EP1PGlsOCo9wrGqZ9dgINXqufryjyM")
    except Exception:
        pass
    fetcher = MacroFetcher()
    regime_det = RegimeDetector()
    shift_pred = RegimeShiftPredictor()
    corr = CorrelationAnalyzer()

    while True:
        try:
            ticker = "QQQ"
            now = datetime.now()

            # ── Macro ───────────────────────────────────────────────
            raw = fetcher.fetch_all()
            macro_df = MacroFetcher.align_series(raw)
            r = regime_det.detect(macro_df)
            b = corr.directional_bias(macro_df)
            s = shift_pred.predict(macro_df)
            si = {
                "vix_level": float(macro_df.get("VIX", [0]).iloc[-1]) if "VIX" in macro_df else 0,
                "dxy_level": float(macro_df.get("DXY", [0]).iloc[-1]) if "DXY" in macro_df else 0,
                "wti_level": float(macro_df.get("WTI", [0]).iloc[-1]) if "WTI" in macro_df else 0,
                "copper_level": float(macro_df.get("COPPER", [0]).iloc[-1]) if "COPPER" in macro_df else 0,
                "skew_level": float(macro_df.get("SKEW", [0]).iloc[-1]) if "SKEW" in macro_df else 0,
                "vix_term": "B",
            }

            # ── Options ─────────────────────────────────────────────
            spot = 0.0
            combined = pd.DataFrame()
            opt = pd.DataFrame()
            gex_data = None
            gexbot_data = None

            # GEXBot
            if gexbot:
                try:
                    gexbot_data = gexbot.get_gex(ticker)
                except Exception:
                    pass

            # Tradier
            try:
                tradier_spot, tradier_chains, tradier_hist = tradier.load_all(ticker, max_expiries=10)
                if tradier_spot and tradier_chains:
                    tradier_chains = [enrich_greeks(c, tradier_spot) for c in tradier_chains if not c.empty]
                    combined = pd.concat(tradier_chains, ignore_index=True)
                    spot = tradier_spot
            except Exception as e:
                log.warning(f"Tradier fallito: {e}")

            # Fallback spot via yfinance se Tradier non ha dato spot
            if spot == 0:
                try:
                    t = yf.Ticker(ticker)
                    h = t.history(period="2d")
                    if not h.empty:
                        spot = float(h["Close"].iloc[-1])
                except Exception:
                    pass

            # yfinance per combined se Tradier non ha dato nulla
            if combined.empty and spot > 0:
                try:
                    t = yf.Ticker(ticker)
                    exps = t.options
                    if exps:
                        today = datetime.now().date()
                        exps_sorted = sorted(exps, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d").date() - today).days))
                        loaded = 0
                        for exp in exps_sorted:
                            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
                            if dte < 0: continue
                            try:
                                c = t.option_chain(exp)
                                df = pd.concat([c.calls.assign(option_type="C"), c.puts.assign(option_type="P")], ignore_index=True)
                                df["dte"] = dte
                                df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
                                df = df.dropna(subset=["strike"])
                                combined = pd.concat([combined, enrich_greeks(df, spot)], ignore_index=True)
                                loaded += 1
                                if loaded >= 35: break
                            except Exception:
                                continue
                except Exception:
                    pass

            # opt: yfinance per DTE singolo (opzionale)
            if spot > 0:
                try:
                    t = yf.Ticker(ticker)
                    exps = t.options
                    if exps:
                        today = datetime.now().date()
                        exps_sorted = sorted(exps, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d").date() - today).days))
                        for exp in exps_sorted:
                            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
                            if dte >= 15:
                                o = t.option_chain(exp)
                                calls = o.calls.copy(); calls["option_type"] = "C"
                                puts = o.puts.copy(); puts["option_type"] = "P"
                                df = pd.concat([calls, puts], ignore_index=True)
                                df["dte"] = dte
                                opt = enrich_greeks(df, spot)
                                break
                except Exception:
                    pass

            # ── Analytics ────────────────────────────────────────────
            payload = {
                "timestamp": now.isoformat(),
                "spot": round(spot, 2) if spot > 0 else 0,
                "ticker": ticker,
                "regime": str(r.get("regime", Regime.TRANSITION).value) if isinstance(r, dict) else "TRANSITION",
                "bias": str(b.value) if hasattr(b, "value") else str(b),
                "shift_prob": round(float(s.get("shift_probability", 0)), 3) if isinstance(s, dict) else 0,
            }

            if not combined.empty and spot > 0:
                # Applica filtro DTE da client
                cf = _client_filter
                combined_f = combined[combined["dte"].between(cf["dte_min"], cf["dte_max"])].copy() if "dte" in combined.columns else combined
                if combined_f.empty:
                    combined_f = combined.copy()
                # Filtra strike ±15% da spot per calcoli sensati
                near_mask = combined_f["strike"].between(spot * 0.85, spot * 1.15)
                combined_near = combined_f[near_mask].copy()

                try:
                    exp_df = compute_exposure_profile(combined_near, spot)
                    if not exp_df.empty:
                        near = exp_df["strike"].between(spot * 0.95, spot * 1.05)
                        tbl = exp_df[near].copy().sort_values("strike")
                        gk_cols = ["vanna", "charm", "speed", "vomma", "color"]
                        gp = {"strikes": tbl["strike"].tolist()}
                        for c in ["gex_call","gex_put","gex_net","dex_call","dex_put","dex_net",
                                  "oi_call","oi_put","oi_total"]:
                            gp[c] = tbl[c].tolist() if c in tbl.columns else []
                        for gk in gk_cols:
                            for s in ["call","put"]:
                                col = f"{gk}_{s}"
                                gp[col] = [round(float(v), 0) for v in tbl[col].tolist()] if col in tbl.columns else []
                            net_col = f"{gk}_net"
                            gp[net_col] = [round(float(v), 0) for v in tbl[net_col].tolist()] if net_col in tbl.columns else []
                        payload["gamma_profile"] = gp
                except Exception as exc:
                    log.warning(f"compute_exposure_profile fallita: {exc}")

                try:
                    gex = compute_gex(combined_near, spot)
                    if len(gex.get("strikes", [])) > 0:
                        payload["gex"] = {
                            "strikes": gex["strikes"].tolist(),
                            "values": gex["gex"].tolist(),
                            "zero_gamma": round(float(gex["zero_gamma"]), 2),
                            "total_gex": round(float(gex["total_gex"]), 0),
                            "max_gex_strike": round(float(gex["max_gex_strike"]), 2),
                            "min_gex_strike": round(float(gex["min_gex_strike"]), 2),
                        }
                except Exception as exc:
                    log.warning(f"compute_gex fallita: {exc}")

                try:
                    dex = compute_dex(combined_near, spot)
                    if len(dex.get("strikes", [])) > 0:
                        payload["dex"] = {
                            "strikes": dex["strikes"].tolist(),
                            "call_dex": dex["call_dex"].tolist(),
                            "put_dex": dex["put_dex"].tolist(),
                            "total_dex": round(float(dex["total_dex"]), 0),
                        }
                except Exception as exc:
                    log.warning(f"compute_dex fallita: {exc}")

                try:
                    walls = gamma_walls(combined_near, spot)
                    payload["walls"] = {
                        "call_wall": round(float(walls["call_wall"]), 2),
                        "put_wall": round(float(walls["put_wall"]), 2),
                        "call_strength": round(float(walls["call_strength"]), 0),
                        "put_strength": round(float(walls["put_strength"]), 0),
                    }
                except Exception as exc:
                    log.warning(f"gamma_walls fallita: {exc}")

                try:
                    oi = open_interest_profile(combined_near, spot)
                    if len(oi.get("strikes", [])) > 0:
                        payload["oi_profile"] = {
                            "strikes": oi["strikes"].tolist(),
                            "oi": oi["oi"].tolist(),
                            "call_oi": oi["call_oi"].tolist(),
                            "put_oi": oi["put_oi"].tolist(),
                            "max_oi_strike": round(float(oi["max_oi_strike"]), 2),
                        }
                except Exception as exc:
                    log.warning(f"open_interest_profile fallita: {exc}")

                # Expected Move
                try:
                    atm_iv = combined_near["implied_volatility"].median() if "implied_volatility" in combined_near else 0.25
                    atm_dte = int(combined_near["dte"].median()) if "dte" in combined_near else 30
                    em = expected_move(spot, atm_iv, atm_dte)
                    if em:
                        em_dollar = float(em.get("expected_move", 0))
                        em_pct = float(em.get("expected_move_pct", 0))
                        # Normal distribution PDF data
                        sigma = em_dollar
                        levels_pdf = {}
                        for s in [1, 2, 3]:
                            levels_pdf[f"+{s}sigma"] = round(spot + s * sigma, 2)
                            levels_pdf[f"-{s}sigma"] = round(spot - s * sigma, 2)
                        # Percentili da distribuzione normale (senza scipy)
                        p5 = round(float(spot - 1.645 * sigma), 2)
                        p25 = round(float(spot - 0.675 * sigma), 2)
                        p50 = round(spot, 2)
                        p75 = round(float(spot + 0.675 * sigma), 2)
                        p95 = round(float(spot + 1.645 * sigma), 2)
                        payload["expected_move"] = {
                            "up": round(em_dollar, 2),
                            "down": round(-em_dollar, 2),
                            "up_pct": round(em_pct, 2),
                            "down_pct": round(-em_pct, 2),
                            "spot": round(spot, 2),
                            "sigma": round(sigma, 2),
                            "sigma_pct": round(em_pct, 2),
                            "levels": levels_pdf,
                            "percentiles": {"p5": p5, "p25": p25, "p50": p50, "p75": p75, "p95": p95},
                        }
                except Exception as exc:
                    log.warning(f"expected_move fallita: {exc}")

                # Volatility Skew
                try:
                    vol_df = combined_near[combined_near["implied_volatility"].notna() & (combined_near["implied_volatility"] > 0.05)].copy()
                    if not vol_df.empty:
                        vol_df["otype"] = vol_df["option_type"].str.lower().str[0]
                        calls_iv = vol_df[vol_df["otype"] == "c"].groupby("strike")["implied_volatility"].mean()
                        puts_iv = vol_df[vol_df["otype"] == "p"].groupby("strike")["implied_volatility"].mean()
                        all_skew = sorted(set(calls_iv.index) | set(puts_iv.index))
                        payload["vol_skew"] = {
                            "strikes": all_skew,
                            "call_iv": [round(float(calls_iv.get(k, 0) * 100), 1) for k in all_skew],
                            "put_iv": [round(float(puts_iv.get(k, 0) * 100), 1) for k in all_skew],
                        }
                except Exception as exc:
                    log.warning(f"vol_skew fallita: {exc}")

                # Gamma Bands
                try:
                    gb = gb_func(combined_near, spot)
                    if gb is not None and not gb.empty:
                        bands_list = []
                        for _, row in gb.iterrows():
                            bands_list.append({
                                "zone": row.get("zona", ""),
                                "dir": row.get("direzione", ""),
                                "lo": round(float(row.get("livello_lo", 0)), 2),
                                "hi": round(float(row.get("livello_hi", 0)), 2),
                                "center": round(float(row.get("centro", 0)), 2),
                                "note": row.get("nota", ""),
                            })
                        payload["gamma_bands"] = bands_list
                except Exception as exc:
                    log.warning(f"gamma_bands fallita: {exc}")

                # Bookmap (solo strike vicini a spot con volume reale)
                try:
                    bookmap_df = combined_near[combined_near["strike"].between(spot * 0.94, spot * 1.06)].copy()
                    if bookmap_df.empty:
                        bookmap_df = combined_near[combined_near["strike"].between(spot * 0.92, spot * 1.08)].copy()
                    dom = dom_bookmap(bookmap_df, spot)
                    if dom and len(dom.get("levels", [])) > 0:
                        payload["bookmap"] = {
                            "levels": [round(float(k), 2) for k in dom["levels"]],
                            "liquidity": [round(float(v), 2) for v in dom["liquidity"]],
                        }
                except Exception as exc:
                    log.warning(f"dom_bookmap fallita: {exc}")

                # Premium Flow
                try:
                    pf = premium_flow(combined_near, spot)
                    if pf and pf.get("by_strike"):
                        payload["premium_flow"] = {
                            "ratio": round(float(pf.get("ratio", 1)), 2),
                            "total_call": round(float(pf.get("total_call", 0)), 0),
                            "total_put": round(float(pf.get("total_put", 0)), 0),
                            "by_strike": pf["by_strike"],
                        }
                except Exception as exc:
                    log.warning(f"premium_flow fallita: {exc}")

                # Greeks per strike (vanna, charm, speed, vomma, color)
                try:
                    gk = combined_near[combined_near["strike"].between(spot * 0.95, spot * 1.05)].copy()
                    if not gk.empty:
                        gk_sorted = gk.sort_values("strike")
                        gk_strikes = [float(s) for s in gk_sorted["strike"].unique()]
                        def _greeks_by_strike(gk_df, col):
                            return gk_df.groupby("strike")[col].sum().reindex(gk_strikes, fill_value=0).tolist()
                        calls = gk_sorted[gk_sorted["option_type"] == "c"]
                        puts = gk_sorted[gk_sorted["option_type"] == "p"]
                        payload["greeks_detail"] = {
                            "strikes": [round(s, 2) for s in gk_strikes],
                            "vanna_call": [round(float(v), 0) for v in _greeks_by_strike(calls, "vanna")],
                            "vanna_put": [round(float(v), 0) for v in _greeks_by_strike(puts, "vanna")],
                            "charm_call": [round(float(v), 0) for v in _greeks_by_strike(calls, "charm")],
                            "charm_put": [round(float(v), 0) for v in _greeks_by_strike(puts, "charm")],
                            "speed_call": [round(float(v), 0) for v in _greeks_by_strike(calls, "speed")],
                            "speed_put": [round(float(v), 0) for v in _greeks_by_strike(puts, "speed")],
                            "vomma_call": [round(float(v), 0) for v in _greeks_by_strike(calls, "vomma")],
                            "vomma_put": [round(float(v), 0) for v in _greeks_by_strike(puts, "vomma")],
                            "color_call": [round(float(v), 0) for v in _greeks_by_strike(calls, "color")],
                            "color_put": [round(float(v), 0) for v in _greeks_by_strike(puts, "color")],
                        }
                except Exception as exc:
                    log.warning(f"greeks_detail fallita: {exc}")

                # Centroid map
                try:
                    gk_all = combined_near.copy()
                    gk_all["premium_est"] = (gk_all["last_price"].fillna(0) * gk_all["open_interest"].fillna(0)).clip(0)
                    call_df = gk_all[gk_all["option_type"] == "c"]
                    put_df = gk_all[gk_all["option_type"] == "p"]
                    call_cent = np.average(call_df["strike"], weights=call_df["premium_est"].clip(0).values + 1) if len(call_df) > 0 else spot
                    put_cent = np.average(put_df["strike"], weights=put_df["premium_est"].clip(0).values + 1) if len(put_df) > 0 else spot
                    total_p = float(gk_all["premium_est"].sum())
                    call_pct = round(float(call_df["premium_est"].sum()) / total_p * 100, 1) if total_p > 0 else 50
                    payload["centroid"] = {
                        "call_centroid": round(float(call_cent), 2),
                        "put_centroid": round(float(put_cent), 2),
                        "call_premium_pct": call_pct,
                        "put_premium_pct": round(100 - call_pct, 1),
                    }
                except Exception as exc:
                    log.warning(f"centroid fallita: {exc}")

            # Regression + MC GARCH (usano dati storici prezzi)
            hist_prices = None
            try:
                if tradier_hist is not None and not tradier_hist.empty and "close" in tradier_hist.columns:
                    hist_prices = tradier_hist["close"].values
                if hist_prices is None or len(hist_prices) < 30:
                    t = yf.Ticker(ticker)
                    h = t.history(period="6mo")
                    if not h.empty:
                        hist_prices = h["Close"].values
            except Exception:
                pass

            if hist_prices is not None and len(hist_prices) > 20:
                # Regression bands (OLS + Ridge)
                try:
                    lr = linreg_channels(hist_prices, window=min(89, len(hist_prices)))
                    if lr:
                        # Ridge regression per confronto
                        ridge_y = hist_prices[-min(89, len(hist_prices)):]
                        ridge_x = np.arange(len(ridge_y)).reshape(-1, 1)
                        ridge_pred, ridge_coef, ridge_intercept = ridge_predict(ridge_x, ridge_y, ridge_x, alpha=5.0)
                        price_subset = hist_prices[-min(89, len(hist_prices)):]
                        payload["regression"] = {
                            "slope": round(float(lr["slope"]), 4),
                            "intercept": round(float(lr["intercept"]), 2),
                            "std": round(float(lr["std"]), 2),
                            "r_squared": round(float(lr["r_squared"]), 4),
                            "x": lr["x"].tolist() if hasattr(lr["x"], "tolist") else list(lr["x"]),
                            "fitted": [round(float(v), 2) for v in lr["fitted"]],
                            "price": [round(float(v), 2) for v in price_subset],
                            "channels": {k: [round(float(vv), 2) for vv in v.tolist()] for k, v in lr["channels"].items()},
                            "ridge_fitted": [round(float(v), 2) for v in ridge_pred],
                            "ridge_coef": [round(float(v), 4) for v in ridge_coef],
                            "ridge_intercept": round(float(ridge_intercept), 2),
                        }
                except Exception as exc:
                    log.warning(f"regression fallita: {exc}")

                # MC GARCH
                try:
                    n_steps = min(30, len(hist_prices) // 5)
                    paths, finals, vol = garch_mc_paths(hist_prices, n_paths=200, n_steps=n_steps)
                    avg = np.mean(paths, axis=0)
                    pcts = np.percentile(finals, [5, 25, 50, 75, 95])
                    # Sample percorsi per non mandare 200 curve
                    sample_idx = np.linspace(0, paths.shape[0] - 1, 50, dtype=int)
                    sampled = paths[sample_idx]
                    payload["mc_garch"] = {
                        "paths": [[round(float(v), 2) for v in row] for row in sampled],
                        "average": [round(float(v), 2) for v in avg],
                        "percentiles": {
                            "p5": round(float(pcts[0]), 2),
                            "p25": round(float(pcts[1]), 2),
                            "p50": round(float(pcts[2]), 2),
                            "p75": round(float(pcts[3]), 2),
                            "p95": round(float(pcts[4]), 2),
                        },
                        "vol_initial": round(float(vol["vol_initial"]) * 100, 2),
                        "vol_mean": round(float(vol["vol_mean"]) * 100, 2),
                        "vol_terminal": round(float(vol["vol_terminal"]) * 100, 2),
                    }
                except Exception as exc:
                    log.warning(f"mc_garch fallita: {exc}")

            if gexbot_data and len(gexbot_data.get("strikes", [])) > 0:
                payload["gexbot"] = {
                    "strikes": [int(s) for s in gexbot_data["strikes"]],
                    "gex": [float(v) for v in gexbot_data["gex"]],
                    "total_gex": round(float(gexbot_data.get("total_gex", 0)), 0),
                    "zero_gamma": round(float(gexbot_data.get("zero_gamma", spot)), 2),
                }

            # Macro summary
            payload["macro"] = {
                k: round(float(v), 2) if isinstance(v, (int, float, np.floating)) else str(v)
                for k, v in si.items()
            }
            # G/I scores per radar chart
            if isinstance(r, dict) and "gip" in r:
                payload["g_score"] = round(float(r["gip"].get("G", 0)), 3)
                payload["i_score"] = round(float(r["gip"].get("I", 0)), 3)
            else:
                payload["g_score"] = 0
                payload["i_score"] = 0

            # OI term heatmap (strike × DTE)
            if not combined.empty and spot > 0:
                try:
                    oi_heat = oi_term_heatmap(combined, spot, max_dte=45, bin_width=2.5)
                    if oi_heat:
                        payload["oi_heatmap"] = clean_nan({
                            "strikes": oi_heat["strikes"].tolist(),
                            "dtes": [int(d) for d in oi_heat["dtes"]],
                            "total_oi": oi_heat["total_oi"].tolist(),
                            "call_oi": oi_heat["call_oi"].tolist(),
                            "put_oi": oi_heat["put_oi"].tolist(),
                            "call_pct": oi_heat["call_pct"].tolist(),
                        })
                except Exception:
                    pass

            # GEX regime heatmap (prezzo × tempo futuro)
            if not combined.empty and spot > 0:
                try:
                    prices_arr, days_arr, gex_mat = gex_regime_heatmap(combined, spot, r=0.05, n_prices=40, n_days=16)
                    if prices_arr is not None and days_arr is not None and gex_mat is not None:
                        payload["gex_surface"] = {
                            "prices": [round(float(p), 2) for p in prices_arr],
                            "days": [int(d) for d in days_arr],
                            "gex_matrix": [[round(float(v), 2) for v in row] for row in gex_mat],
                        }
                except Exception:
                    pass

            # CPCV risk assessment
            if not combined.empty and spot > 0:
                try:
                    payload["cpcv"] = cpcv_risk_assessment(combined, spot)
                except Exception:
                    pass

            # Ergodicity market gap
            if not combined.empty and spot > 0:
                try:
                    payload["ergodic"] = ergodic_market_gap(combined, spot)
                except Exception:
                    pass

            # Macro history (ultimi 5gg per chart time-series)
            try:
                with db_conn() as c:
                    hist_df = pd.read_sql(
                        "SELECT timestamp, vix, dxy, wti, copper, skew, gex_total, gex_zg, vvr, regime "
                        "FROM macro_history WHERE timestamp >= ? ORDER BY timestamp ASC",
                        c, params=((datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),))
                if not hist_df.empty:
                    payload["macro_history"] = {
                        "timestamps": hist_df["timestamp"].tolist(),
                        "vix": [round(float(v), 2) if v else None for v in hist_df["vix"]],
                        "dxy": [round(float(v), 2) if v else None for v in hist_df["dxy"]],
                        "wti": [round(float(v), 2) if v else None for v in hist_df["wti"]],
                        "gex_total": [round(float(v), 2) if v else None for v in hist_df["gex_total"]],
                        "gex_zg": [round(float(v), 2) if v else None for v in hist_df["gex_zg"]],
                        "regime": hist_df["regime"].tolist(),
                    }
            except Exception:
                pass

            # VVR calcolato dal regime
            try:
                vvr_map = {"EXPANSION": 3.5, "SLOWDOWN": 5.5, "REFLATION": 4.5, "STAGFLATION": 7.5, "TRANSITION": 5.0}
                regime_name = str(r.get("regime", "").value) if isinstance(r, dict) else ""
                payload["vvr"] = vvr_map.get(regime_name, 5.0)
            except Exception:
                payload["vvr"] = 5.0

            # PCA decomposition su macro data
            try:
                pca_cols = ["SPX", "VIX", "DXY", "WTI", "COPPER", "SKEW", "VVIX"]
                if isinstance(macro_df, pd.DataFrame) and all(c in macro_df.columns for c in pca_cols):
                    data_mat = macro_df[pca_cols].dropna().values
                    pca_result = pca_decomposition(data_mat, n_components=3)
                    if pca_result:
                        ev = pca_result["explained_variance"]
                        payload["pca"] = {
                            "pc1_var": round(float(ev[0]) * 100, 1),
                            "pc2_var": round(float(ev[1]) * 100, 1),
                            "pc3_var": round(float(ev[2]) * 100, 1),
                            "cumulative": round(float(pca_result["cumulative_var"][-1]) * 100, 1),
                            "explained": [round(float(v) * 100, 1) for v in ev],
                        }
            except Exception:
                pass

            # Efficient Frontier (SPY + QQQ + risk-free)
            try:
                t = yf.Ticker(ticker)
                qqq_hist = t.history(period="6mo")["Close"]
                spy = yf.Ticker("SPY")
                spy_hist = spy.history(period="6mo")["Close"]
                if len(qqq_hist) > 50 and len(spy_hist) > 50:
                    qqq_ret = qqq_hist.pct_change().dropna().values[-126:]
                    spy_ret = spy_hist.pct_change().dropna().values[-126:]
                    rf = 0.05 / 252
                    n_port = 50
                    weights = np.linspace(0, 1, n_port)
                    sharpe = []
                    for w in weights:
                        p_ret = w * qqq_ret + (1 - w) * spy_ret
                        excess = np.mean(p_ret) - rf
                        std = np.std(p_ret)
                        sharpe.append(excess / std * np.sqrt(252) if std > 0 else 0)
                    best_idx = int(np.argmax(sharpe))
                    payload["efficient_frontier"] = {
                        "max_sharpe": round(float(sharpe[best_idx]), 3),
                        "qqq_weight": round(float(weights[best_idx]), 3),
                        "spy_weight": round(float(1 - weights[best_idx]), 3),
                        "sharpe_curve": [round(float(s), 3) for s in sharpe],
                    }
            except Exception:
                pass

            # News (yfinance)
            try:
                t = yf.Ticker(ticker)
                news_items = t.news[:8] if t.news else []
                news_list = []
                for item in news_items:
                    title = str(item.get("title", ""))
                    sentiment = ""
                    if _vader and title:
                        scores = _vader.polarity_scores(title)
                        if scores["compound"] >= 0.15:
                            sentiment = "POSITIVE"
                        elif scores["compound"] <= -0.15:
                            sentiment = "NEGATIVE"
                        else:
                            sentiment = "NEUTRAL"
                    news_list.append({
                        "title": title,
                        "publisher": str(item.get("publisher", "")),
                        "link": str(item.get("link", "")),
                        "time": str(item.get("providerPublishTime", "")),
                        "type": str(item.get("type", "")),
                        "sentiment": sentiment,
                    })
                payload["news"] = news_list
            except Exception:
                payload["news"] = []

            _state["spot"] = spot
            _state["timestamp"] = now.isoformat()
            _state["payload"] = payload
            _state["ready"] = True

            # Broadcast
            msg = json.dumps({"type": "update", "data": clean_nan(payload)}, default=str)
            dead = set()
            for ws in connected_clients:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.add(ws)
            connected_clients.difference_update(dead)

        except Exception as e:
            log.error(f"Data loop error: {e}", exc_info=True)
            if _state["ready"] and _state["payload"]:
                _state["payload"]["error"] = str(e)
                msg = json.dumps({"type": "update", "data": clean_nan(_state["payload"])}, default=str)
            else:
                msg = json.dumps({"type": "update", "data": {"spot": 0, "ticker": ticker, "error": str(e)}}, default=str)
            for ws in connected_clients:
                try: await ws.send_text(msg)
                except: pass

        _data_ready.set()

        # Aspetta 30s o fino a nuovo filtro
        try:
            await asyncio.wait_for(asyncio.shield(_filter_event.wait()), timeout=30)
        except asyncio.TimeoutError:
            pass
        _filter_event.clear()

# ── WebSocket ───────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    log.info(f"WS client connected (total: {len(connected_clients)})")
    if not _state["ready"]:
        try:
            await asyncio.wait_for(_data_ready.wait(), timeout=30)
        except asyncio.TimeoutError:
            pass
    if _state["ready"] and _state["payload"]:
        try:
            msg = json.dumps({"type": "update", "data": clean_nan(_state["payload"])}, default=str)
            log.info(f"Sending initial state: {len(_state['payload'])} keys, spot={_state['payload'].get('spot')}")
            await ws.send_text(msg)
        except Exception as exc:
            log.warning(f"Initial state send error: {exc}")
    else:
        log.warning(f"WS: ready={_state['ready']}, payload={'set' if _state['payload'] else 'None'}")
    try:
        while True:
            txt = await ws.receive_text()
            try:
                cmd = json.loads(txt)
                if cmd.get("type") == "filter":
                    global _client_filter
                    _client_filter = {
                        "dte_min": int(cmd.get("dte_min", 0)),
                        "dte_max": int(cmd.get("dte_max", 365)),
                        "source": str(cmd.get("source", "all")),
                        "timeframe": str(cmd.get("timeframe", "5d")),
                    }
                    _filter_event.set()
                    log.info(f"Filter: DTE {_client_filter['dte_min']}-{_client_filter['dte_max']}, src={_client_filter['source']}, tf={_client_filter['timeframe']}")
            except (json.JSONDecodeError, ValueError):
                pass
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(ws)
        log.info(f"WS client disconnected (total: {len(connected_clients)})")

# ── Static files ────────────────────────────────────────────────────────
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

# ── Startup ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    asyncio.create_task(data_loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
