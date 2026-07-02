"""
SPX Probability Explorer
========================
Visualizzazione interattiva delle probabilita implicite SPX.
Seleziona la scadenza da un menu a tendina e vedi la distribuzione
di probabilita, lo spot, l'expected value e i target.

Filosofia: automazione meccanica, semplicita, nessuna discrezionalita.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import simpson
from dataclasses import dataclass
from typing import Optional
import warnings
import logging
import subprocess, json, os, sys
from quant_analytics import compute_gex, monte_carlo_paths, monte_carlo_confidence, fetch_macro, regime_from_vix, expected_move

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    ticker: str = "^SPX"
    risk_free_rate: float = 0.05
    current_price: Optional[float] = None
    min_dte: int = 3
    max_dte: int = 120
    n_expirations: int = 10  # quante scadenze campionare
    target_dte: int = 45  # scadenza principale per la dashboard

CONFIG = Config()

# =============================================================================
# DATA
# =============================================================================

class DataFetcher:
    def __init__(self):
        self._yf = self._check_yfinance()
        self._conda_python = self._find_conda_python()

    def _check_yfinance(self):
        try:
            import yfinance as yf
            return yf
        except ImportError:
            return None

    def _find_conda_python(self):
        paths = [
            r"C:\Users\Gabriel\OpenBB\conda\envs\openbb\python.exe",
            r"C:\Users\Gabriel\OpenBB\conda\Scripts\python.exe",
        ]
        for p in paths:
            if os.path.isfile(p):
                return p
        return None

    def fetch_current_price(self):
        # Try Tastyworks first (real-time)
        try:
            from tastyworks_fetcher import fetch_spot
            spot = fetch_spot()
            if spot:
                return spot
        except Exception:
            pass
        # Try dxFeed next (real-time se credenziali valide)
        try:
            from dxfeed_connector import fetch_spot
            spot = fetch_spot()
            if spot:
                return spot
        except Exception:
            pass
        if not self._yf:
            return 4500.0
        spx = self._yf.Ticker(CONFIG.ticker)
        data = spx.history(period="5d")
        return float(data["Close"].iloc[-1]) if not data.empty else 4500.0

    def fetch_multiple_chains(self, min_dte=7, max_dte=120, n_chains=10):
        # 0) Tastyworks (real-time, OPRA incluso, deposito $0)
        chains = self._fetch_via_tasty(min_dte, max_dte, n_chains)
        if chains:
            log.info("Usato Tastyworks (real-time)")
            return chains

        # 1) dxFeed WebSocket (real-time se credenziali valide)
        chains = self._fetch_via_dxfeed(min_dte, max_dte, n_chains)
        if chains:
            log.info("Usato dxFeed (real-time)")
            return chains

        # 2) OpenBB + CBOE (delayed ~15min, include IV reale)
        chains = self._fetch_via_openbb(min_dte, max_dte, n_chains)
        if chains:
            log.info("Usato OpenBB + CBOE")
            return chains

        # 3) yfinance (delayed ~20min)
        chains = self._fetch_via_yfinance(min_dte, max_dte, n_chains)
        if chains:
            log.info("Usato yfinance")
            return chains

        # 4) Dummy sintetici (offline)
        log.warning("Nessuna fonte dati disponibile -> dummy")
        return self._dummy_chains(S0=CONFIG.current_price or 4500.0,
                                  min_dte=min_dte, max_dte=max_dte, n=n_chains)

    def _fetch_via_tasty(self, min_dte, max_dte, n_chains):
        try:
            from tastyworks_fetcher import fetch_multiple_chains, fetch_spot
            spot = fetch_spot()
            if spot:
                CONFIG.current_price = spot
            return fetch_multiple_chains(min_dte, max_dte, n_chains)
        except Exception as e:
            log.debug(f"Tastyworks non disponibile: {e}")
            return None

    def _fetch_via_dxfeed(self, min_dte, max_dte, n_chains):
        try:
            from dxfeed_connector import fetch_chains, fetch_spot
            spot = fetch_spot()
            if spot:
                CONFIG.current_price = spot
            chains = fetch_chains(min_dte, max_dte, n_chains)
            return chains
        except Exception as e:
            log.debug(f"dxFeed non disponibile: {e}")
            return None

    def _fetch_via_openbb(self, min_dte, max_dte, n_chains):
        if not self._conda_python:
            return None
        fetcher_path = os.path.join(os.path.dirname(__file__), "openbb_fetcher.py")
        if not os.path.isfile(fetcher_path):
            return None
        try:
            result = subprocess.run(
                [self._conda_python, fetcher_path,
                 "--min-dte", str(min_dte),
                 "--max-dte", str(max_dte),
                 "--max-chains", str(n_chains)],
                capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                log.warning(f"OpenBB error: {result.stderr[:200]}")
                return None
            data = json.loads(result.stdout)
            if "error" in data:
                return None
            chains = []
            S0 = data["underlying_price"]
            CONFIG.current_price = S0
            for exp in data["expirations"]:
                dte = exp["dte"]
                rows = []
                for opt in exp["options"]:
                    p = opt.get("last_price") or opt.get("bid") or opt.get("ask")
                    if p is None or p <= 0:
                        continue
                    rows.append({
                        "strike": opt["strike"],
                        "lastPrice": p,
                        "type": opt["option_type"].lower(),
                        "dte": dte,
                        "implied_volatility": opt.get("implied_volatility"),
                    })
                if rows:
                    chains.append(pd.DataFrame(rows).sort_values("strike"))
            return chains if chains else None
        except Exception as e:
            log.warning(f"OpenBB fallback: {e}")
            return None

    def _fetch_via_yfinance(self, min_dte, max_dte, n_chains):
        if not self._yf:
            return None
        spx = self._yf.Ticker(CONFIG.ticker)
        exps = spx.options
        if not exps:
            return None
        today = datetime.now().date()
        exp_dates = []
        for e in exps:
            d = datetime.strptime(e, "%Y-%m-%d").date()
            dte = (d - today).days
            if min_dte <= dte <= max_dte:
                exp_dates.append((dte, e))
        exp_dates.sort()
        if len(exp_dates) > n_chains:
            indices = np.linspace(0, len(exp_dates) - 1, n_chains).astype(int)
            exp_dates = [exp_dates[i] for i in indices]
        chains = []
        for dte, exp_str in exp_dates:
            log.info(f"yfinance: {exp_str} ({dte} DTE)...")
            try:
                chain = spx.option_chain(exp_str)
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                calls["type"] = "call"; puts["type"] = "put"
                df = pd.concat([calls, puts], ignore_index=True)
                df["dte"] = dte
                df = df[df["volume"] > 0].copy() if "volume" in df.columns else df
                if not df.empty:
                    chains.append(df)
            except Exception as e:
                log.warning(f"Errore {exp_str}: {e}")
        return chains if chains else None

    def _dummy_chains(self, S0, min_dte=7, max_dte=120, n=10):
        """Dati sintetici per test offline."""
        from scipy.stats import norm
        chains = []
        dtes = np.linspace(min_dte, max_dte, n).astype(int)
        for dte in dtes:
            T = dte / 365.0
            sigma = 0.12 + 0.06 * np.exp(-T * 2) + 0.02 * np.random.randn()
            sigma = np.clip(sigma, 0.08, 0.40)
            strikes = np.arange(S0 * 0.5, S0 * 1.5, 15)
            rows = []
            for K in strikes:
                d1 = (np.log(S0 / K) + (CONFIG.risk_free_rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                call = S0 * norm.cdf(d1) - K * np.exp(-CONFIG.risk_free_rate * T) * norm.cdf(d2)
                put = K * np.exp(-CONFIG.risk_free_rate * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
                rows.append({"strike": K, "lastPrice": max(call, 0.01), "type": "call", "dte": dte, "implied_volatility": sigma})
                rows.append({"strike": K, "lastPrice": max(put, 0.01), "type": "put", "dte": dte, "implied_volatility": sigma})
            chains.append(pd.DataFrame(rows).sort_values("strike"))
        return chains


# =============================================================================
# PDF EXTRACTOR
# =============================================================================

class PDFExtractor:
    """
    Breeden-Litzenberger: f(K) = e^{rT} * d^2C/dK^2

    Usa SOLO opzioni OTM (piu' liquide):
    - K < S0: OTM Put -> convertito a Call via put-call parity
    - K > S0: OTM Call diretto
    Boundary: natural (d2C/dK2 = 0 agli estremi, corretto per call prices)
    """

    def extract(self, df: pd.DataFrame, S0: float) -> Optional[dict]:
        T = max(float(df["dte"].iloc[0]), 0.5) / 365.0
        r = CONFIG.risk_free_rate

        lp = next((c for c in ["last_price", "lastPrice", "last"] if c in df.columns), None)
        if lp is None:
            log.error("Nessuna colonna prezzo trovata (last_price/lastPrice/last)")
            return None
        type_col = "option_type" if "option_type" in df.columns else "type"
        calls = df[(df[type_col].str.lower().str[0] == "c") & (df[lp] > 0.01)].copy()
        puts = df[(df[type_col].str.lower().str[0] == "p") & (df[lp] > 0.01)].copy()
        if calls.empty or puts.empty:
            return None

        call_by_strike = calls.groupby("strike")[lp].mean()
        put_by_strike = puts.groupby("strike")[lp].mean()

        # Unisci tutti gli strikes disponibili
        all_strikes = sorted(set(call_by_strike.index) | set(put_by_strike.index))
        all_strikes = [k for k in all_strikes if S0 * 0.7 < k < S0 * 1.3]

        synth_calls = []
        for K in all_strikes:
            if K >= S0 and K in call_by_strike.index:
                # OTM Call diretto
                synth_calls.append((K, call_by_strike[K]))
            elif K < S0 and K in put_by_strike.index:
                # OTM Put -> Call sintetica via put-call parity: C = P + S - K*e^{-rT}
                P = put_by_strike[K]
                C_synth = P + S0 - K * np.exp(-r * T)
                if C_synth > 0.01:
                    synth_calls.append((K, C_synth))

        if len(synth_calls) < 8:
            return None

        K_arr = np.array([x[0] for x in synth_calls])
        C_arr = np.array([x[1] for x in synth_calls])
        idx = np.argsort(K_arr)
        K_arr, C_arr = K_arr[idx], C_arr[idx]

        try:
            spline = CubicSpline(K_arr, C_arr, bc_type="natural")
        except ValueError:
            return None

        K_grid = np.linspace(K_arr.min(), K_arr.max(), 300)
        d2C_dK2 = spline.derivative(2)(K_grid)
        pdf = np.exp(r * T) * d2C_dK2
        pdf = np.clip(pdf, 1e-10, None)

        total = simpson(pdf, K_grid)
        if total > 0:
            pdf = pdf / total

        # Sanity check: expected value deve essere vicino al forward
        ev = simpson(K_grid * pdf, K_grid)
        forward = S0 * np.exp(r * T)
        if abs(ev - forward) > 0.15 * forward:
            log.warning(f"  DTE {df['dte'].iloc[0]:.0f}: EV={ev:.0f} vs forward={forward:.0f}, skip")
            return None

        # Estrai IV reali se presenti
        iv_dict = {}
        if "implied_volatility" in df.columns:
            iv_df = df[df["implied_volatility"].notna() & (df["implied_volatility"] > 0)]
            for _, row in iv_df.iterrows():
                k = float(row["strike"])
                iv = float(row["implied_volatility"])
                ot = str(row.get(type_col, "c")).lower()
                iv_dict[k] = {"iv": iv, "type": ot}

        return {"K": K_grid, "pdf": pdf, "S0": S0, "T": T,
                "dte": float(df["dte"].iloc[0]),
                "iv_skew": iv_dict}


# =============================================================================
# PDF COLLECTOR - Raccoglie tutte le PDF su una griglia comune
# =============================================================================

class PDFCollector:
    """Allinea tutte le PDF su una griglia comune di strike."""

    def collect(self, pdfs: list, S0: float):
        if not pdfs:
            return None, None, None

        # Griglia comune (ridotta per performance)
        all_K = np.concatenate([p["K"] for p in pdfs])
        K_common = np.linspace(all_K.min(), all_K.max(), 200)

        # DTE ordinati
        dtes = np.array([p["dte"] for p in pdfs])
        idx = np.argsort(dtes)
        dtes = dtes[idx]
        pdfs_sorted = [pdfs[i] for i in idx]

        # Interpola ogni PDF sulla griglia comune
        pdf_grid = []
        for p in pdfs_sorted:
            f = interp1d(p["K"], p["pdf"], kind="linear", bounds_error=False, fill_value=0)
            pv = np.clip(f(K_common), 0, None)
            total = simpson(pv, K_common)
            if total > 0:
                pv = pv / total
            pdf_grid.append(pv)

        return K_common, dtes, np.array(pdf_grid)


# =============================================================================
# INTERACTIVE PDF VIEWER (Plotly con dropdown)
# =============================================================================

def _newton_iv(S, K, T, price, typ):
    """Newton-Raphson IV solver. typ='call' or 'put'."""
    from scipy.stats import norm
    if price <= 0 or T <= 0:
        return 0.2
    intrinsic = max(0, S - K) if typ == "call" else max(0, K - S)
    if price <= intrinsic + 0.01:
        return 0.01
    def bs(sigma):
        d1 = (np.log(S / K) + (CONFIG.risk_free_rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if typ == "call":
            return S * norm.cdf(d1) - K * np.exp(-CONFIG.risk_free_rate * T) * norm.cdf(d2)
        return K * np.exp(-CONFIG.risk_free_rate * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    iv = 0.2
    for _ in range(30):
        p_est = bs(iv)
        diff = p_est - price
        if abs(diff) < 1e-6:
            break
        d1 = (np.log(S / K) + (CONFIG.risk_free_rate + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1) if iv > 1e-6 else 1.0
        if abs(vega) < 1e-12:
            break
        iv -= diff / vega
        iv = np.clip(iv, 0.005, 5.0)
    return float(np.clip(iv, 0.01, 2.0))


class InteractivePDFViewer:
    """Genera un HTML interattivo con menu a tendina per selezionare la scadenza.
    Mostra: curva PDF, SPOT, Expected Value, Target, aree di confidenza.
    """

    def _check_plotly(self):
        try:
            import plotly
            return True
        except ImportError:
            return False

    def render(self, K, dtes, pdf_grid, S0, pdfs_raw, filepath="spx_probability_viewer.html"):
        if not self._check_plotly():
            log.warning("plotly non installato. Nessun grafico generato.")
            return None

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        evs = []
        stds = []
        peaks_by_dte = []
        for i in range(len(dtes)):
            pv = pdf_grid[i]
            ev = simpson(K * pv, K)
            evs.append(ev)
            std = np.sqrt(simpson((K - ev)**2 * pv, K))
            stds.append(std)
            mean_p = pv.mean()
            pk = []
            for j in range(1, len(pv) - 1):
                if pv[j] > pv[j-1] and pv[j] > pv[j+1] and pv[j] > mean_p * 1.3:
                    pk.append(K[j])
            peaks_by_dte.append(pk[:3])

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.08,
            subplot_titles=("Distribuzione di Probabilita'", "Volatility Skew"),
        )

        conf_colors = {"50%": "rgba(0,255,136,0.15)",
                       "68%": "rgba(255,215,0,0.12)",
                       "90%": "rgba(255,107,53,0.10)",
                       "95%": "rgba(233,69,96,0.08)"}

        # Precalcola probabilita' su intervalli di 50 punti (bins fissi)
        cdfs = []
        prob_bars = []
        bin_centers = None
        for i in range(len(dtes)):
            pv = pdf_grid[i]
            cdf = np.cumsum(pv) * (K[1] - K[0])
            cdfs.append(cdf)
            # Crea bins da 50 punti su tutto il range K
            bin_w = 50
            bins = np.arange(K[0], K[-1] + bin_w, bin_w)
            bc = (bins[:-1] + bins[1:]) / 2
            if bin_centers is None:
                bin_centers = bc
            probs = np.array([simpson(pv[l:r], K[l:r])
                              for l, r in zip(
                np.searchsorted(K, bins[:-1]),
                np.searchsorted(K, bins[1:])
            )])
            prob_bars.append(probs)

        for i in range(len(dtes)):
            pv = pdf_grid[i]
            dte = int(dtes[i])
            ev = evs[i]
            std = stds[i]
            cdf = cdfs[i]
            pb = prob_bars[i]
            lo = max(K[0], S0 - 3.5 * std)
            hi = min(K[-1], S0 + 3.5 * std)
            mask = (K >= lo) & (K <= hi)
            Kv, Pv, Cdfv = K[mask], pv[mask], cdf[mask]
            # Filtra bins nella finestra visibile
            bm = (bin_centers >= lo) & (bin_centers <= hi)
            Bc, Pb = bin_centers[bm], pb[bm] * 100  # in %

            # Barre della probabilita' (istogramma)
            fig.add_trace(
                go.Bar(x=Bc, y=Pb, width=40,
                       marker={"color": "#00d2ff", "opacity": 0.5},
                       name=f"Prob {dte}D",
                       hovertemplate=(
                           f"DTE={dte}<br>"
                           "Strike: %{x:.0f}<br>"
                           "Probabilita': <b>%{{y:.2f}}%%</b><br>"
                           "<extra></extra>"
                       ),
                       visible=(i == 0)),
                row=1, col=1)

            # Aree di confidenza (rettangoli)
            cdf = np.cumsum(pv) * (K[1] - K[0])
            ymax_rect = Pb.max() * 1.3 if len(Pb) > 0 else 1
            for level, color_hex in [("50%", "rgba(0,255,136,0.15)"),
                                      ("68%", "rgba(255,215,0,0.12)"),
                                      ("90%", "rgba(255,107,53,0.10)"),
                                      ("95%", "rgba(233,69,96,0.08)")]:
                pct = float(level.replace("%", "")) / 100
                tail = (1 - pct) / 2
                lk = np.interp(tail, cdf, K)
                uk = np.interp(1 - tail, cdf, K)
                fig.add_trace(
                    go.Scatter(x=[lk, lk, uk, uk],
                               y=[0, ymax_rect, ymax_rect, 0],
                               fill="toself", mode="none",
                               fillcolor=color_hex,
                               showlegend=False, hoverinfo="skip",
                               visible=(i == 0)),
                    row=1, col=1)

            fig.add_trace(
                go.Scatter(x=[S0, S0], y=[0, Pb.max() * 1.2],
                           mode="lines", line={"color": "white", "width": 2.5, "dash": "solid"},
                           name=f"SPOT {S0:.0f}",
                           visible=(i == 0)),
                row=1, col=1)

            fig.add_trace(
                go.Scatter(x=[ev, ev], y=[0, Pb.max() * 1.0],
                           mode="lines", line={"color": "#ffd700", "width": 2, "dash": "dash"},
                           name=f"EV {ev:.0f}",
                           visible=(i == 0)),
                row=1, col=1)

            for rank, k_peak in enumerate(peaks_by_dte[i][:3]):
                if abs(k_peak / S0 - 1) > 0.005:
                    colors = ["#ffd700", "#ff8c00", "#ff4500"]
                    fig.add_trace(
                        go.Scatter(x=[k_peak, k_peak], y=[0, Pb.max() * 0.7 - rank * 0.05],
                                   mode="lines", line={"color": colors[rank], "width": 1.5, "dash": "dot"},
                                   name=f"T{rank+1} {k_peak:.0f}",
                                   visible=(i == 0)),
                        row=1, col=1)

            # Volatility Skew (usa IV reale se disponibile)
            iv_data = pdfs_raw[i].get("iv_skew", {}) if pdfs_raw and i < len(pdfs_raw) else {}
            if iv_data:
                # IV reale da OpenBB/CBOE o dxFeed
                calls_iv = {k: v["iv"] for k, v in iv_data.items() if v["type"] == "call"}
                puts_iv = {k: v["iv"] for k, v in iv_data.items() if v["type"] == "put"}
                atm_strike = min(iv_data.keys(), key=lambda k: abs(k - S0))
                ivs = []
                for k in sorted(set(calls_iv) | set(puts_iv)):
                    iv = calls_iv.get(k) or puts_iv.get(k)
                    if iv and 0.05 < iv < 1.0:
                        ivs.append({"strike": k, "iv": iv})
                if ivs:
                    ivdf = pd.DataFrame(ivs).sort_values("strike")
                    fig.add_trace(
                        go.Scatter(x=ivdf["strike"], y=ivdf["iv"] * 100,
                                   mode="lines+markers",
                                   line={"color": "#e94560", "width": 1.5},
                                   marker={"size": 2},
                                   name=f"IV {dte}D",
                                   hovertemplate=f"DTE={dte}<br>Strike: %{{x:.0f}}<br>IV: %{{y:.1f}}%<extra></extra>",
                                   visible=(i == 0)),
                        row=2, col=1)
            elif pdfs_raw and len(pdfs_raw) > i and "_opt_df" in pdfs_raw[i]:
                # Fallback: crude IV dalla mid price
                odf = pdfs_raw[i]["_opt_df"]
                if odf is not None and not odf.empty:
                    lp = next((c for c in ["last_price", "lastPrice", "last"] if c in odf.columns), "last_price")
                    tcol = "option_type" if "option_type" in odf.columns else "type"
                    c_df = odf[odf[tcol].str.lower().str[0] == "c"].groupby("strike")[lp].mean()
                    p_df = odf[odf[tcol].str.lower().str[0] == "p"].groupby("strike")[lp].mean()
                    strikes_iv = sorted(set(c_df.index) & set(p_df.index))
                    T_dte = dte / 365.0
                    ivs = []
                    for sk in strikes_iv:
                        mid = (c_df[sk] + p_df[sk]) / 2
                        iv = self._quick_iv(S0, sk, T_dte, mid)
                        if iv and 0.05 < iv < 1.0:
                            ivs.append({"strike": sk, "iv": iv})
                    if ivs:
                        ivdf = pd.DataFrame(ivs).sort_values("strike")
                        fig.add_trace(
                            go.Scatter(x=ivdf["strike"], y=ivdf["iv"] * 100,
                                       mode="lines+markers",
                                       line={"color": "#e94560", "width": 1.5},
                                       marker={"size": 2},
                                       name=f"IV {dte}D",
                                       hovertemplate=f"DTE={dte}<br>Strike: %{{x:.0f}}<br>IV: %{{y:.1f}}%<extra></extra>",
                                       visible=(i == 0)),
                            row=2, col=1)

        n_traces_per_dte = len(fig.data) // len(dtes)
        steps = []
        for i in range(len(dtes)):
            start = i * n_traces_per_dte
            end = (i + 1) * n_traces_per_dte
            step = {"method": "update",
                    "label": f"{int(dtes[i])} DTE",
                    "args": [{"visible": [False] * len(fig.data)}]}
            for j in range(start, min(end, len(fig.data))):
                step["args"][0]["visible"][j] = True
            steps.append(step)

        fig.update_layout(
            updatemenus=[{
                "buttons": steps,
                "direction": "down",
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.05, "xanchor": "left",
                "y": 1.15, "yanchor": "top",
                "bgcolor": "#16213e", "font": {"color": "white"}, "bordercolor": "#444",
            }],
            title={"text": f"SPX Probability Explorer - Spot {S0:.0f}",
                   "x": 0.5, "font": {"size": 18, "color": "white"}},
            annotations=[{
                "text": "<b>NOTA:</b> Le barre mostrano la PROBABILITA' EFFETTIVA (%%), non densita'. Ogni barra = probabilita' in intervallo di 50 punti. Aree colorate = range di confidenza.",
                "x": 0.5, "y": 1.03, "xref": "paper", "yref": "paper",
                "showarrow": False, "font": {"size": 10, "color": "#888"},
            }],
            paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
            font={"color": "#cccccc"},
            hovermode="x unified", margin={"l": 60, "r": 30, "t": 80, "b": 40},
            legend={"font": {"color": "#cccccc"}, "bgcolor": "rgba(0,0,0,0)"},
        )
        fig.update_xaxes(title="Strike", gridcolor="#333", color="#888", row=1, col=1)
        fig.update_yaxes(title="Probabilita' (%) su intervalli di 50 punti",
                         gridcolor="#333", color="#888", row=1, col=1)
        fig.update_yaxes(title="Vol. Impl. (%)", gridcolor="#333",
                         color="#888", row=2, col=1)
        fig.update_xaxes(title="Strike", gridcolor="#333", color="#888", row=2, col=1)
        fig.update_yaxes(title="IV (%)", gridcolor="#333", color="#888", row=2, col=1)

        fig.write_html(filepath)
        log.info(f"Viewer salvato: {filepath}")
        try:
            import os
            os.startfile(os.path.abspath(filepath))
        except Exception:
            pass
        return filepath

    def _quick_iv(self, S, K, T, price):
        if price <= 0 or T <= 0:
            return None
        return _newton_iv(S, K, T, price, "call")


# =============================================================================
# HEATMAP (Probabilita' Prezzo vs Tempo)
# =============================================================================

class _BinHelper:
    """Helper: converte PDF grid in probabilita' per bins da 50 punti."""

    @staticmethod
    def compute(K, pdf_grid, bin_w=50):
        bins = np.arange(K[0], K[-1] + bin_w, bin_w)
        bc = (bins[:-1] + bins[1:]) / 2
        prob_grid = np.zeros((pdf_grid.shape[0], len(bc)))
        for i in range(pdf_grid.shape[0]):
            for j, (l, r) in enumerate(zip(
                    np.searchsorted(K, bins[:-1]),
                    np.searchsorted(K, bins[1:]))):
                if r > l:
                    prob_grid[i, j] = simpson(pdf_grid[i][l:r], K[l:r]) * 100
        return bc, prob_grid


class HeatmapPlot:
    """Mappa di calore 2D: X=DTE, Y=Strike, colore = PROBABILITA' EFFETTIVA (%)."""

    def render(self, K, dtes, pdf_grid, S0, gex_data=None, filepath="spx_heatmap.html"):
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        bc, prob_grid = _BinHelper.compute(K, pdf_grid)
        fig = go.Figure(data=go.Heatmap(
            z=prob_grid, x=dtes, y=bc,
            colorscale="Viridis", zsmooth="best",
            hovertemplate="DTE: %{x:.0f}<br>Strike: %{y:.0f}<br>Prob: %{z:.2f}%%<extra></extra>",
            colorbar={"title": {"text": "Prob. (%)", "side": "right"}},
        ))
        fig.add_trace(go.Scatter(
            x=dtes, y=[S0] * len(dtes), mode="lines+markers",
            name=f"SPOT {S0:.0f}", line={"color": "white", "width": 3},
            marker={"size": 6, "color": "white", "symbol": "diamond"},
        ))
        evs = [simpson(K * pdf_grid[i], K) for i in range(len(dtes))]
        fig.add_trace(go.Scatter(
            x=dtes, y=evs, mode="lines+markers", name="Expected Value",
            line={"color": "cyan", "width": 2, "dash": "dash"},
            marker={"size": 4, "color": "cyan"},
        ))

        # Zero Gamma line (GEX)
        if gex_data:
            zg_vals = []
            for dte in dtes:
                nearest = min(gex_data.keys(), key=lambda k: abs(k - dte))
                zg_vals.append(gex_data[nearest]["zero_gamma"])
            fig.add_trace(go.Scatter(
                x=dtes, y=zg_vals, mode="lines+markers",
                name="Zero Gamma", line={"color": "#ff4500", "width": 2, "dash": "dot"},
                marker={"size": 5, "color": "#ff4500", "symbol": "triangle-down"},
            ))

        fig.update_layout(
            title={"text": f"SPX Probability Heatmap - Spot {S0:.0f}",
                   "x": 0.5, "font": {"size": 18, "color": "white"}},
            xaxis={"title": "DTE", "gridcolor": "#444", "color": "#ccc", "dtick": 10},
            yaxis={"title": "Strike", "gridcolor": "#444", "color": "#ccc"},
            paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
            font={"color": "#cccccc"}, hovermode="x unified",
            margin={"l": 60, "r": 40, "t": 60, "b": 60},
        )
        fig.write_html(filepath)
        log.info(f"Heatmap salvata: {filepath}")
        try:
            import os
            os.startfile(os.path.abspath(filepath))
        except Exception:
            pass
        return filepath


# =============================================================================
# 3D PROBABILITY SURFACE (Plotly interattivo)
# =============================================================================

class Surface3DPlot:
    """Superficie 3D: X=Strike, Y=DTE, Z=Probabilita' effettiva (%)."""

    def render(self, K, dtes, pdf_grid, S0, filepath="spx_3d_surface.html"):
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        bc, prob_grid = _BinHelper.compute(K, pdf_grid)
        K_m, D_m = np.meshgrid(bc, dtes)

        fig = go.Figure(data=[go.Surface(
            x=K_m, y=D_m, z=prob_grid,
            colorscale="Viridis", opacity=0.9,
            hovertemplate="Strike: %{x:.0f}<br>DTE: %{y:.0f}<br>Prob: %{z:.2f}%%<extra></extra>",
            colorbar={"title": {"text": "Prob. (%)", "side": "right"}},
        )])

        z_max = prob_grid.max() if prob_grid.size > 0 else 1

        # Linea SPOT
        fig.add_trace(go.Scatter3d(
            x=[S0] * len(dtes), y=dtes, z=[0] * len(dtes),
            mode="lines+markers",
            name=f"SPOT {S0:.0f}",
            line={"color": "red", "width": 5},
            marker={"size": 5, "color": "red"},
        ))

        # Linea Expected Value
        evs = [simpson(K * pdf_grid[i], K) for i in range(len(dtes))]
        fig.add_trace(go.Scatter3d(
            x=evs, y=dtes,
            z=[prob_grid[i].max() * 0.5 for i in range(len(dtes))],
            mode="lines+markers",
            name="Expected Value",
            line={"color": "cyan", "width": 3, "dash": "dash"},
            marker={"size": 3, "color": "cyan"},
        ))

        fig.update_layout(
            title={"text": f"SPX Probability Surface - Spot {S0:.0f}",
                   "x": 0.5, "font": {"size": 16, "color": "white"}},
            scene={
                "xaxis": {"title": "Strike", "color": "white", "gridcolor": "#444"},
                "yaxis": {"title": "DTE", "color": "white", "gridcolor": "#444"},
                "zaxis": {"title": "Probabilita'", "color": "white", "gridcolor": "#444"},
                "camera": {"eye": {"x": 1.5, "y": -1.5, "z": 1.0}},
                "bgcolor": "#1a1a2e",
            },
            paper_bgcolor="#1a1a2e",
            font={"color": "#cccccc"},
            margin={"l": 0, "r": 0, "t": 50, "b": 0},
        )

        fig.write_html(filepath)
        log.info(f"Superficie 3D salvata: {filepath}")
        try:
            import os
            os.startfile(os.path.abspath(filepath))
        except Exception:
            pass
        return filepath


# =============================================================================
# METRICS per singola scadenza (dashboard testuale)
# =============================================================================

class MetricsCalculator:

    def compute_all(self, pdf: dict, opt_df: pd.DataFrame) -> dict:
        S0 = pdf["S0"]
        K, p = pdf["K"], pdf["pdf"]
        return {
            "spot": round(S0, 2),
            "dte": int(opt_df["dte"].iloc[0]) if not opt_df.empty else 45,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "expected_ranges": self._expected_ranges(K, p, S0),
            "market_bias": self._market_bias(opt_df, S0),
            "targets": self._targets(K, p, S0),
        }

    def _expected_ranges(self, K, p, S0):
        if len(K) < 2:
            return {}
        dx = K[1] - K[0]
        cdf = np.cumsum(p) * dx
        ranges = {}
        for level, name in [(0.50, "50%"), (0.68, "68% (1sd)"),
                            (0.90, "90%"), (0.95, "95% (2sd)")]:
            tail = (1 - level) / 2
            lower = np.interp(tail, cdf, K)
            upper = np.interp(1 - tail, cdf, K)
            ranges[name] = {"lower": round(lower, 1), "upper": round(upper, 1),
                            "down%": round((lower / S0 - 1) * 100, 2),
                            "up%": round((upper / S0 - 1) * 100, 2)}
        return ranges

    def _market_bias(self, df, S0):
        if df.empty:
            return {"bias": "NEUTRAL", "skew": 0}
        dte = float(df["dte"].iloc[0])
        T = dte / 365.0
        lp = next((c for c in ["last_price", "lastPrice", "last"] if c in df.columns), "last_price")
        type_col = "option_type" if "option_type" in df.columns else "type"
        calls = df[df[type_col].str.lower().str[0] == "c"]
        puts = df[df[type_col].str.lower().str[0] == "p"]
        def get_iv(group, target, typ):
            if group.empty:
                return None
            row = group.iloc[(group["strike"] - target).abs().argsort()[:1]]
            return self._approx_iv(S0, float(row["strike"].iloc[0]),
                                   T, float(row[lp].iloc[0]), typ) if not row.empty else None
        iv_put = get_iv(puts, S0 * 0.95, "put")
        iv_call = get_iv(calls, S0 * 1.05, "call")
        if iv_put is None or iv_call is None:
            return {"bias": "NEUTRAL", "skew": 0}
        skew = iv_put - iv_call
        if skew > 0.03:
            bias = "BEARISH"
        elif skew > 0.01:
            bias = "SLIGHTLY_BEARISH"
        elif skew < -0.03:
            bias = "BULLISH"
        elif skew < -0.01:
            bias = "SLIGHTLY_BULLISH"
        else:
            bias = "NEUTRAL"
        return {"bias": bias, "skew": round(skew, 4),
                "put_iv": round(iv_put, 4), "call_iv": round(iv_call, 4)}

    def _approx_iv(self, S, K, T, price, typ):
        return _newton_iv(S, K, T, price, typ)

    def _targets(self, K, p, S0):
        if len(K) < 5:
            return []
        targets = []
        ev = simpson(K * p, K)
        targets.append({"tipo": "Valore Atteso", "livello": round(ev, 1),
                        "delta%": round((ev / S0 - 1) * 100, 2)})
        std = np.sqrt(simpson((K - ev)**2 * p, K))
        skew = simpson(((K - ev) / max(std, 1e-6))**3 * p, K)
        targets.append({"tipo": "Deviazione Std", "livello": round(std, 1),
                        "delta%": round(std / S0 * 100, 2)})
        targets.append({"tipo": "Asimmetria", "livello": round(skew, 4), "delta%": None})
        peaks = []
        mean_p = p.mean()
        for i in range(1, len(p) - 1):
            if p[i] > p[i-1] and p[i] > p[i+1] and p[i] > mean_p * 1.3:
                peaks.append((K[i], p[i]))
        peaks.sort(key=lambda x: -x[1])
        for i, (k_v, prob) in enumerate(peaks[:3]):
            if abs(k_v / S0 - 1) > 0.005:
                targets.append({"tipo": f"Target #{i+1}", "livello": round(k_v, 1),
                                "delta%": round((k_v / S0 - 1) * 100, 2),
                                "prob_rel": round(prob / p.max(), 4)})
        mask_up = K > S0 * 1.05
        mask_down = K < S0 * 0.95
        prob_up = simpson(p[mask_up], K[mask_up]) if mask_up.any() else 0
        prob_down = simpson(p[mask_down], K[mask_down]) if mask_down.any() else 0
        targets.append({"tipo": "Prob >5% UP", "livello": round(prob_up * 100, 1), "delta%": None})
        targets.append({"tipo": "Prob >5% DOWN", "livello": round(prob_down * 100, 1), "delta%": None})
        return targets


# =============================================================================
# DASHBOARD
# =============================================================================

class Dashboard:

    BAR = "=" * 74

    def render(self, metrics: dict) -> str:
        lines = [self.BAR,
                 f"  SPX PROBABILITY EXPLORER",
                 f"  {metrics.get('date', 'N/A')}",
                 self.BAR]
        lines.append(f"\n  SPOT: {metrics['spot']:<10.2f}  DTE: {metrics['dte']}")

        lines.append(f"\n  {'RANGES ATTESI':^72}")
        lines.append(f"  {'Confidenza':<18} {'Lower':>10} {'Upper':>10} {'Downside':>10} {'Upside':>10}")
        lines.append(f"  {'-'*60}")
        for name, r in metrics.get("expected_ranges", {}).items():
            lines.append(f"  {name:<18} {r['lower']:>10.1f} {r['upper']:>10.1f} "
                         f"{r['down%']:>+9.2f}% {r['up%']:>+8.2f}%")

        mb = metrics.get("market_bias", {})
        lines.append(f"\n  {'BIAS DI MERCATO':^72}")
        lines.append(f"  Direzione: {mb.get('bias', 'N/A')}")
        lines.append(f"  Skew: {mb.get('skew', 0):+.4f}  |  Put IV: {mb.get('put_iv', 0):.2%}  Call IV: {mb.get('call_iv', 0):.2%}")

        targets = metrics.get("targets", [])
        lines.append(f"\n  {'TARGET':^72}")
        for t in targets:
            if t.get("prob_rel") is not None:
                lines.append(f"  {t['tipo']:<22} {t['livello']:>10.1f}  ({t['delta%']:+7.2f}%)  [rel prob: {t['prob_rel']:.2f}]")
            elif t.get("delta%") is not None:
                lines.append(f"  {t['tipo']:<22} {t['livello']:>10.1f}  ({t['delta%']:+7.2f}%)")
            else:
                lines.append(f"  {t['tipo']:<22} {t['livello']:>10.1f}%")

        lines.append(f"\n{self.BAR}")
        return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n{'  SOVEREIGN QUANT ECOSYSTEM  ':*^74}\n")

    data = DataFetcher()
    pdf_ext = PDFExtractor()
    metrics_calc = MetricsCalculator()
    dash = Dashboard()

    S0 = data.fetch_current_price()
    CONFIG.current_price = S0
    log.info(f"SPX: {S0:.2f}")

    # MACRO
    macro = fetch_macro()
    if macro:
        v = macro.get("vix")
        d = macro.get("dxy")
        c = macro.get("copper")
        regime = regime_from_vix(v)
        log.info(f"VIX: {v} | DXY: {d} | Copper: {c} | Regime: {regime}")
        print(f"  Macro: VIX={v} DXY={d} Copper={c} | {regime}")

    # Scarica catene per multiple scadenze
    chains = data.fetch_multiple_chains(
        min_dte=CONFIG.min_dte, max_dte=CONFIG.max_dte,
        n_chains=CONFIG.n_expirations
    )
    log.info(f"Scadenze caricate: {len(chains)}")

    # Estrae PDF per ogni scadenza (salva opt_df per lo skew/GEX)
    pdfs = []
    for df in chains:
        p = pdf_ext.extract(df, S0)
        if p is not None:
            p["_opt_df"] = df
            pdfs.append(p)

    if not pdfs:
        log.error("Nessuna PDF estratta.")
        return

    log.info(f"PDF estratte: {len(pdfs)}")

    # GEX per ogni chain (se dati gamma/OI disponibili)
    gex_data = {}
    for df in chains:
        dte = float(df["dte"].iloc[0])
        g = compute_gex(df, S0)
        if g and len(g["strikes"]) > 0:
            gex_data[dte] = g
    if gex_data:
        total_gex = sum(g["total_gex"] for g in gex_data.values())
        log.info(f"GEX totale: {total_gex:+,.0f}")
        print(f"  GEX Totale: {total_gex:+,.0f}")

    # Allinea tutte le PDF su griglia comune
    collector = PDFCollector()
    K, dtes, pdf_grid = collector.collect(pdfs, S0)
    if K is None:
        log.error("Griglia non generata.")
        return

    # Expected Move per la scadenza principale
    main_pdf = min(pdfs, key=lambda p: abs(p["dte"] - CONFIG.target_dte))
    iv_avg = None
    if main_pdf.get("iv_skew"):
        ivs = [v["iv"] for v in main_pdf["iv_skew"].values() if v["iv"]]
        if ivs:
            iv_avg = np.mean(ivs)
    if iv_avg:
        em = expected_move(S0, iv_avg, main_pdf["dte"])
        log.info(f"Expected Move {main_pdf['dte']}D: {em['expected_move']:.0f} ({em['expected_move_pct']:.1f}%)")
        print(f"  Expected Move: {em['lower']:.0f} - {S0:.0f} - {em['upper']:.0f} ({em['expected_move_pct']:.1f}%)")

    # Viewer interattivo con dropdown (curva PDF 2D)
    viewer = InteractivePDFViewer()
    plot_file = viewer.render(K, dtes, pdf_grid, S0, pdfs)
    if plot_file:
        log.info(f"Viewer: {plot_file}")

    # Heatmap (Prezzo vs Tempo) con zero-gamma line
    heatmap = HeatmapPlot()
    heat_file = heatmap.render(K, dtes, pdf_grid, S0, gex_data)
    if heat_file:
        log.info(f"Heatmap: {heat_file}")

    # Superficie 3D
    surface = Surface3DPlot()
    surf_file = surface.render(K, dtes, pdf_grid, S0)
    if surf_file:
        log.info(f"3D Surface: {surf_file}")

    # Monte Carlo paths per la scadenza principale (bonus file)
    mc_file = None
    if main_pdf:
        K_mc = main_pdf["K"]
        p_mc = main_pdf["pdf"]
        paths, finals = monte_carlo_paths(K_mc, p_mc, n_paths=3000, n_steps=50)
        mc_file = _render_monte_carlo(paths, S0, main_pdf["dte"])

    # Dashboard per la scadenza principale
    main_chain = None
    for df in chains:
        if abs(float(df["dte"].iloc[0]) - main_pdf["dte"]) < 2:
            main_chain = df
            break
    if main_chain is None:
        main_chain = chains[0]

    metrics = metrics_calc.compute_all(main_pdf, main_chain)
    print(dash.render(metrics))
    print(f"\n  File generati:")
    print(f"  [1] Viewer PDF:       {plot_file or 'NON GENERATO'}")
    print(f"  [2] Heatmap:          {heat_file or 'NON GENERATO'}")
    print(f"  [3] Superficie 3D:    {surf_file or 'NON GENERATO'}")
    if mc_file:
        print(f"  [4] Monte Carlo:      {mc_file}")
    print(f"  Macro: VIX={macro.get('vix','?')} DXY={macro.get('dxy','?')} Cu={macro.get('copper','?')}")
    print(f"  Regime: {regime_from_vix(macro.get('vix'))}")
    print("\nAnalisi completata. I file HTML si sono aperti nel browser.")
    print("Usa il menu a tendina nel Viewer per selezionare la scadenza.")
    print("Ruota la Superficie 3D con il mouse per esplorare le anomalie.\n")


def _render_monte_carlo(paths, S0, dte, filepath="spx_monte_carlo.html"):
    """Genera grafico Monte Carlo con Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    conf = monte_carlo_confidence(paths)
    t = np.arange(paths.shape[1])

    fig = go.Figure()

    # Aree di confidenza
    for lower, upper, color, name in [
        (5, 95, "rgba(0,255,136,0.08)", "90%"),
        (25, 75, "rgba(0,255,136,0.15)", "50%"),
    ]:
        fig.add_trace(go.Scatter(
            x=list(t) + list(t[::-1]),
            y=list(conf[lower]) + list(conf[upper][::-1]),
            fill="toself", mode="none",
            fillcolor=color, name=name,
        ))

    # Sample paths (50 su 3000 per chiarezza)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(paths), min(50, len(paths)), replace=False)
    for i in idx:
        fig.add_trace(go.Scatter(
            x=t, y=paths[i], mode="lines",
            line={"width": 0.5, "color": "rgba(0,210,255,0.2)"},
            showlegend=False, hoverinfo="skip",
        ))

    # Mediana
    fig.add_trace(go.Scatter(
        x=t, y=conf[50], mode="lines",
        line={"color": "#00d2ff", "width": 2},
        name="Mediana",
    ))

    # Spot level
    fig.add_hline(y=S0, line={"color": "white", "width": 1.5, "dash": "dash"},
                  annotation_text=f"SPOT {S0:.0f}")

    fig.update_layout(
        title={"text": f"Monte Carlo Paths - {dte} DTE - Spot {S0:.0f}",
               "x": 0.5, "font": {"size": 16, "color": "white"}},
        xaxis={"title": "Time Step", "gridcolor": "#444", "color": "#ccc"},
        yaxis={"title": "SPX Level", "gridcolor": "#444", "color": "#ccc"},
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font={"color": "#cccccc"},
        hovermode="x unified",
        margin={"l": 60, "r": 30, "t": 60, "b": 40},
    )

    fig.write_html(filepath)
    log.info(f"Monte Carlo salvato: {filepath}")
    try:
        os.startfile(os.path.abspath(filepath))
    except Exception:
        pass
    return filepath


if __name__ == "__main__":
    main()
