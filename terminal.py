"""
MAZE CAPITAL — Terminal v5.0
=============================
GEXBot real-time + yfinance dual-source engine.
Run: streamlit run terminal.py
"""

import sys, os, time, io, sqlite3
from pathlib import Path
from datetime import datetime, timedelta, date
from contextlib import contextmanager
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.integrate import simpson
from scipy.stats import norm as scipy_norm

pio.templates["maze"] = pio.templates["plotly_dark"]
pio.templates["maze"].layout.update(uirevision="reset", autosize=True)
pio.templates.default = "maze"

sys.path.insert(0, str(Path(__file__).parent))

from quant_analytics import (
    compute_gex, monte_carlo_confidence, fetch_macro, expected_move,
    garch_mc_paths, compute_all_greeks, gamma_walls,
    premium_flow, linreg_channels, intraday_anomalies, dom_bookmap, volume_profile,
    compute_dex, open_interest_profile, gex_centroid_map, enrich_greeks, gamma_bands,
    vxn_from_chain, gex_regime_heatmap, save_eod_snapshot,
    compute_exposure_profile,
    bl_quantile_levels, knockout_barriers, iv_skew_extremes,
    etf_rebalancing_levels, compute_probabilistic_levels,
    box_plot_stats, z_score_analysis, correlation_signal, noise_filter,
    implied_risk_moments, structural_confluence, decompose_risk_premia,
    calculate_expected_shortfall, adapted_var, cornish_fisher_adjustment,
)
from macro_quant import (
    Config, DataFetcher as MacroFetcher, CorrelationAnalyzer,
    RegimeDetector, RegimeShiftPredictor, StatisticalValidator,
    Regime, Bias,
)
from gexbot_client import GEXBotClient
from tradier_client import TradierClient
from mgtgo_panel import render_mgtgo

log = logging.getLogger(__name__)

st.set_page_config(page_title="Maze Capital Terminal", page_icon="⬡",
                   layout="wide", initial_sidebar_state="expanded")

# ── COLORI ────────────────────────────────────────────────────────────
DARK="#0a0e17"; CARD="#111827"; CARD_B="#1f2937"; ACCENT="#00d2ff"
GREEN="#22c55e"; RED="#ef4444"; GOLD="#fbbf24"; YELLOW="#eab308"
BROWN="#a16207"; TEXT="#e2e8f0"; TEXT_M="#64748b"; ORANGE="#f97316"

st.markdown(f"""
<style>
    .stApp,div[data-testid="stAppViewContainer"] {{ background:{DARK}; color:{TEXT}; }}
    .block-container {{ padding:.3rem .8rem!important; max-width:100%!important; }}
    h1,h2,h3,h4,h5,h6 {{ color:{TEXT}!important; }}
    .panel {{ background:{CARD}; border:1px solid {CARD_B}; border-radius:6px; padding:6px 10px; margin:3px 0; }}
    .ptitle {{ font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:{TEXT_M}; margin-bottom:3px; display:flex; justify-content:space-between; }}
    .stPlotlyChart {{ height:auto!important; }}
    .modebar {{ opacity:.3!important; transition:opacity .2s; }}
    .modebar:hover {{ opacity:1!important; background:{CARD_B}; border-radius:4px; }}
    footer,div[data-testid="stToolbar"] {{ display:none!important; }}
    .stDataFrame,div[data-testid="dataframe-container"] {{ background:transparent!important; }}
    div[data-testid="stMetric"] {{ background:{CARD_B}; border-radius:4px; padding:2px 6px; }}
    div[data-testid="stMetric"] label {{ color:{TEXT_M}!important; font-size:9px!important; }}
    div[data-testid="stMetric"] div {{ font-size:16px!important; font-weight:700!important; }}
    .stSelectbox, .stSlider, .stCheckbox {{ font-size:11px!important; }}
</style>
""", unsafe_allow_html=True)

# ── GAMMA BANDS DA GEXBot ──────────────────────────────────────────────
def gamma_bands_from_gexbot(gex_data: dict, spot: float) -> pd.DataFrame:
    ZONES = [
        ("Zona di rottura ^", 1.50, 2.00),
        ("Resistenza primaria ^", 0.75, 1.50),
        ("Resistenza 2deg ^", 0.35, 0.75),
        ("Attrito ^", 0.15, 0.35),
        ("-- SPOT (Max Pain zone) --", -0.15, 0.15),
        ("Attrito v", -0.35, -0.15),
        ("Supporto 2deg v", -0.75, -0.35),
        ("Supporto primario v", -1.50, -0.75),
        ("Zona di rottura v", -2.00, -1.50),
    ]
    zg = gex_data.get("zero_gamma", spot)
    cw_k = gex_data.get("major_pos")
    pw_k = gex_data.get("major_neg")
    total_gex = gex_data.get("total_gex", 0)
    strikes = gex_data.get("strikes", np.array([]))
    gex_vals = gex_data.get("gex", np.array([]))
    max_oi = None
    if len(strikes) > 0 and len(gex_vals) > 0:
        max_oi = strikes[np.argmax(np.abs(gex_vals))]
    rows = []
    for name, pct_lo, pct_hi in ZONES:
        lo = spot * (1 + pct_lo / 100)
        hi = spot * (1 + pct_hi / 100)
        centro = (lo + hi) / 2
        if "rottura" in name.lower():
            direzione = "^" if "^" in name else "v"
            if total_gex < 0:
                nota = "GEX short gamma" if total_gex > -50000 else "GEX short gamma estremo"
            else:
                nota = "GEX long gamma" if total_gex < 50000 else "GEX long gamma estremo"
        elif "primaria" in name.lower():
            if "^" in name:
                direzione = "^"
                nota = "Call wall" if cw_k and abs(cw_k - centro) / centro < 0.01 else "Resistenza chiave"
            else:
                direzione = "v"
                nota = "Put wall" if pw_k and abs(pw_k - centro) / centro < 0.01 else "Supporto chiave"
        elif "2deg" in name:
            direzione = "^" if "^" in name else "v"
            nota = "Alto GEX" if max_oi and abs(max_oi - centro) / centro < 0.015 else ("Resistenza secondaria" if "^" in name else "Supporto secondario")
        elif "Attrito" in name:
            direzione = "^" if "^" in name else "v"
            nota = "Range" if abs(zg - centro) / centro < 0.02 else ("Attrito rialzista" if "^" in name else "Attrito ribassista")
        else:
            direzione = "-"
            nota = "Zona magnetica ATM"
        rows.append(dict(direzione=direzione, zona=name, pct_low=pct_lo, pct_high=pct_hi, livello_lo=round(lo, 2), livello_hi=round(hi, 2), centro=round(centro, 2), nota=nota))
    return pd.DataFrame(rows)

# ── SESSION ───────────────────────────────────────────────────────────
for k,cls in {"fetcher":MacroFetcher,"corr":CorrelationAnalyzer,
              "regime_det":RegimeDetector,"shift_pred":RegimeShiftPredictor}.items():
    if k not in st.session_state: st.session_state[k]=cls()
for k in ("last_refresh","data","spot","chains","history_db","expanded_chart"):
    if k not in st.session_state: st.session_state[k]=None if k!="last_refresh" else datetime.now()
for k,default in [("ticker_sel","SPY"),("ds_src","Hybrid"),("data_source","Hybrid")]:
    if k not in st.session_state: st.session_state[k]=default

# ── GEXBot init ────────────────────────────────────────────────────
GEXBOT_API_KEY = "gexbot_custom_rnOqKSpAUAYL5EP1PGlsOCo9wrGqZ9dgINXqufryjyM"
if "gexbot" not in st.session_state:
    try:
        st.session_state.gexbot = GEXBotClient(GEXBOT_API_KEY)
    except Exception:
        st.session_state.gexbot = None
if "data_source" not in st.session_state:
    st.session_state.data_source = "Hybrid"  # Hybrid / yfinance / GEXBot
if "gexbot_spot" not in st.session_state:
    st.session_state.gexbot_spot = None
if "tradier" not in st.session_state:
    st.session_state.tradier = TradierClient("dJoOgKq0zvagRUm1HaNaaVNcQiGq")

# ── SQLITE LOGGING ────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent/"macro_history.db"

class HistoryDB:
    def __init__(self):
        self._init_db()
    @contextmanager
    def conn(self):
        c=sqlite3.connect(str(DB_PATH)); yield c; c.close()
    def _init_db(self):
        with self.conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS macro_history(id INTEGER PRIMARY KEY AUTOINCREMENT,timestamp TEXT NOT NULL,regime TEXT,g_score REAL,i_score REAL,confidence REAL,bias TEXT,shift_prob REAL,vix REAL,wti REAL,copper REAL,dxy REAL,skew REAL,vix_term TEXT,gex_env TEXT,gex_total REAL,gex_zg REAL,vvr REAL,UNIQUE(timestamp))")
            for col,typ in [("gex_env","TEXT"),("gex_total","REAL"),("gex_zg","REAL"),("vvr","REAL")]:
                try: c.execute(f"ALTER TABLE macro_history ADD COLUMN {col} {typ}")
                except: pass
            c.execute("CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY AUTOINCREMENT,timestamp TEXT NOT NULL,type TEXT,message TEXT,acknowledged INTEGER DEFAULT 0)")
            c.commit()
    def log(self,r,b,s,si,ge=None,gt=None,gz=None,vvr=None):
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"); g=r.get("gip",{})
        with self.conn() as c:
            c.execute("INSERT OR IGNORE INTO macro_history(timestamp,regime,g_score,i_score,confidence,bias,shift_prob,vix,wti,copper,dxy,skew,vix_term,gex_env,gex_total,gex_zg,vvr) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (ts,r["regime"].value,round(g.get("G",0),4),round(g.get("I",0),4),round(r.get("confidence",0),2),b.value,round(s.get("shift_probability",0),3),round(si.get("vix_level",0),1),round(si.get("wti_level",0),1),round(si.get("copper_level",0),2),round(si.get("dxy_level",0),1),round(si.get("skew_level",0),0),si.get("vix_term",""),ge,gt,gz,vvr)); c.commit()
    def history(self,days=30):
        with self.conn() as c:
            return pd.read_sql(f"SELECT * FROM macro_history WHERE timestamp>=? ORDER BY timestamp ASC",c,params=((datetime.now()-timedelta(days=days)).strftime("%Y-%m-%d"),))

# ── ECONOMIC CALENDAR ─────────────────────────────────────────────────
class EconCal:
    MONTHS_IT={"Jan":"Gen","Feb":"Feb","Mar":"Mar","Apr":"Apr","May":"Mag","Jun":"Giu","Jul":"Lug","Aug":"Ago","Sep":"Set","Oct":"Ott","Nov":"Nov","Dec":"Dic"}
    HIGH_IMPACT=["NFP","Non Farm","Payrolls","Employment","CPI","Consumer Price","FOMC","Fed Decision","Interest Rate","GDP","Gross Domestic","PPI","Producer Price","Retail Sales","ISM","PMI","Michigan","Unemployment Rate","JOLTS","ADP"]
    def upcoming(self,hours=48):
        import calendar as cal_mod; today=datetime.now(); rows=[]
        cal=cal_mod.monthcalendar(today.year,today.month)
        tf=None
        for w in cal:
            if w[cal_mod.FRIDAY]!=0:
                if tf is None: tf=w[cal_mod.FRIDAY]
                else: tf=w[cal_mod.FRIDAY]; break
        if tf: rows.append({"datetime":today.replace(day=tf,hour=8,minute=30),"title":"Options Expiry (OpEx)","impact":"High","source":"Calendar"})
        ff=None
        for w in cal:
            if w[cal_mod.FRIDAY]!=0: ff=w[cal_mod.FRIDAY]; break
        if ff: rows.append({"datetime":today.replace(day=ff,hour=8,minute=30),"title":"NFP (Employment)","impact":"High","source":"Calendar"})
        cd=14 if today.month in (2,4,6,8,10,12) else 15
        rows.append({"datetime":today.replace(day=min(cd,cal_mod.monthrange(today.year,today.month)[1]),hour=8,minute=30),"title":"CPI","impact":"High","source":"Calendar"})
        fomc={1:28,3:18,4:29,6:17,7:29,9:16,10:28,12:9}
        if today.month in fomc:
            try:
                from zoneinfo import ZoneInfo
                et_tz=ZoneInfo("America/New_York")
                local_tz=datetime.now().astimezone().tzinfo
                fomc_et=today.replace(day=fomc[today.month],hour=14,minute=0,second=0,microsecond=0).replace(tzinfo=et_tz)
                fomc_local=fomc_et.astimezone(local_tz).replace(tzinfo=None)
                rows.append({"datetime":fomc_local,"title":"FOMC Decision","impact":"High","source":"Calendar"})
            except Exception:
                rows.append({"datetime":today.replace(day=fomc[today.month],hour=14,minute=0),"title":"FOMC Decision","impact":"High","source":"Calendar"})
        df=pd.DataFrame(rows).sort_values("datetime") if rows else pd.DataFrame()
        if df.empty: return df
        now=datetime.now(); cut=now+timedelta(hours=hours)
        return df[(df["datetime"]>=now)&(df["datetime"]<=cut)].sort_values("datetime")

# ── DATA LOADERS ──────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_macro():
    raw=st.session_state.fetcher.fetch_all()
    df=MacroFetcher.align_series(raw)
    r=st.session_state.regime_det.detect(df); b=st.session_state.corr.directional_bias(df)
    s=st.session_state.shift_pred.predict(df); cm=st.session_state.corr.latest_correlation_matrix(df)
    return df,r,b,s,cm

@st.cache_data(ttl=30)
def load_options(ticker):
    try:
        import yfinance as yf
        t=yf.Ticker(f"^{ticker}" if ticker in ("SPX",) else ticker)
        h=t.history(period="5d"); spot=float(h["Close"].iloc[-1]) if not h.empty else 5000.0
        exps=t.options; chains=[]; today=datetime.now().date()
        if exps:
            exps_sorted=sorted(exps,key=lambda e:abs((datetime.strptime(e,"%Y-%m-%d").date()-today).days))
            loaded=0
            for exp in exps_sorted:
                dte=(datetime.strptime(exp,"%Y-%m-%d").date()-today).days
                if dte<0: continue
                try:
                    c=t.option_chain(exp)
                    df=pd.concat([c.calls.assign(type="call"),c.puts.assign(type="put")],ignore_index=True)
                    if df.empty: continue
                    df["dte"]=dte
                    df["strike"]=pd.to_numeric(df["strike"],errors="coerce")
                    df=df.dropna(subset=["strike"])
                    chains.append(enrich_greeks(df,spot))
                    loaded+=1
                    if loaded>=35: break
                except Exception:
                    continue
            # sanity: se spot yfinance e' palesemente fuori, usa midpoint strikes e ricalcola greci
            if chains and spot<200:
                all_s=pd.concat([c["strike"] for c in chains])
                spot=float(all_s.median())
                chains=[enrich_greeks(c,spot) for c in chains]
        return spot,chains,h
    except Exception:
        return None,None,None

@st.cache_data(ttl=300)
def hist_prices(ticker):
    try:
        import yfinance as yf
        h=yf.Ticker(ticker).history(period="6mo")
        if h is None or h.empty:
            return None, None
        close = h["Close"].values
        bars = [{"time": t.strftime("%Y-%m-%d"), "open": float(r["Open"]), "high": float(r["High"]), "low": float(r["Low"]), "close": float(r["Close"])} for t, r in h.iterrows()]
        return close, bars
    except Exception:
        return None, None

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="color:{ACCENT};font-size:22px;font-weight:800;">⬡ MAZE CAPITAL</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{TEXT_M};font-size:10px;margin-bottom:10px;">Terminal v4.0</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:{CARD_B};border-radius:4px;padding:4px 10px;font-size:10px;color:{TEXT_M};">Fattori: SPX | VIX | DXY | COPPER | WTI | SKEW</div>', unsafe_allow_html=True)
    st.markdown(f'<hr style="border-color:{CARD_B};margin:6px 0;">', unsafe_allow_html=True)
    if st.button("⟳ Refresh Data",type="primary",width="stretch"):
        st.cache_data.clear()
        st.rerun()
    e1,e2=st.columns(2)
    with e1:
        if st.button("⬇ HTML",width="stretch"):
            st.toast("Report HTML pronto")
    with e2:
        if st.button("⬇ CSV",width="stretch"):
            st.toast("CSV storico pronto")
    st.select_slider("MGT-GO height", options=[350, 450, 600, 800], value=600, key="chart_height")
    st.caption(f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    with st.expander("⚙ Debug Data", expanded=True):
        dummy_placeholder = st.empty()

# ── LOAD ──────────────────────────────────────────────────────────────
ticker_sel=st.session_state.ticker_sel; ds_src=st.session_state.ds_src
st.session_state.data_source=ds_src
df,macro_reg,bias,shift,corr_mat=load_macro()
if ds_src == "Tradier":
    try:
        spot, chains, hist = st.session_state.tradier.load_all(ticker_sel)
        if spot is None or not chains:
            raise ValueError("Nessun dato Tradier")
    except Exception as e:
        log.warning(f"Tradier load_all exception: {e}")
        spot, chains, hist = (None, [], None)

    if spot is None or not chains:
        log.warning(f"Tradier fallback yfinance per {ticker_sel}")
        spot, chains, hist = load_options(ticker_sel)
else:
    spot, chains, hist = load_options(ticker_sel)
macro=fetch_macro()
gip=macro_reg.get("gip",{}); shift_p=shift.get("shift_probability",0)
signals=macro_reg.get("signals",{}); spot=spot or 5000.0

# ── GEXBot data ─────────────────────────────────────────────────────
gexbot_gex = None
st.session_state.gexbot_spot = None  # pulito ogni run
gexbot_ok = (st.session_state.data_source in ("Hybrid","GEXBot")
             and st.session_state.gexbot is not None)
if gexbot_ok:
    try:
        gb_spot = st.session_state.gexbot.get_spot(ticker_sel)
        if gb_spot and gb_spot > 0:
            st.session_state.gexbot_spot = gb_spot
        gexbot_gex = st.session_state.gexbot.gex_per_strike(ticker_sel, weight="vol")
        # sanity: se ZG >40% da spot, GEXBot dati errati
        if gexbot_gex and gexbot_gex.get("zero_gamma") and abs(gexbot_gex["zero_gamma"]/spot-1) > 0.4:
            gexbot_gex = None
            st.session_state.gexbot_spot = None
    except Exception:
        gexbot_gex = None
        st.session_state.gexbot_spot = None

r_val=macro_reg.get("regime","TRANSIZIONE")
if isinstance(r_val,Regime): r_str=r_val.value
elif isinstance(r_val,(int,np.integer)): r_str=["TRANSIZIONE","EXPANSION","SLOWDOWN","REFLATION","STAGFLATION"][min(int(r_val),4)]
else: r_str=str(r_val).upper()
r_str=r_str if r_str in ("EXPANSION","SLOWDOWN","REFLATION","STAGFLATION","TRANSIZIONE") else "TRANSIZIONE"

rc_map={"EXPANSION":GREEN,"SLOWDOWN":"#3b82f6","REFLATION":ORANGE,"STAGFLATION":RED,"TRANSIZIONE":TEXT_M}
rc=rc_map.get(r_str,TEXT_M)
bg_map={"EXPANSION":"#00cc66","SLOWDOWN":"#3399ff","REFLATION":"#ff8800","STAGFLATION":"#ff3333","TRANSIZIONE":"#999999"}
bg=bg_map.get(r_str,"#999999")
shift_color="#ff4444" if shift_p>.5 else "#ffaa00" if shift_p>.3 else "#888"

combined=pd.concat(chains,ignore_index=True) if chains else pd.DataFrame()
price_data, ohlc_bars = hist_prices("SPY" if ticker_sel in ("SPX","SPY") else ticker_sel)
dtes_avail=sorted(combined["dte"].unique()) if not combined.empty else [0]
for k,default in [("dte_sel",30),("sr",15),("refresh_sec",15),("auto",True),("audio_on",True)]:
    if k not in st.session_state: st.session_state[k]=default
dte_sel=st.session_state.dte_sel; sr=st.session_state.sr
refresh_sec=st.session_state.refresh_sec; auto=st.session_state.auto; audio_on=st.session_state.audio_on
dte_use=min(dtes_avail,key=lambda d:abs(d-dte_sel))
opt=combined[combined["dte"]==dte_use].copy() if not combined.empty else pd.DataFrame()

# Avviso se Tradier selezionato MA sta usando yfinance (fallback)
if ds_src == "Tradier" and (not chains or len(chains) == 0):
    st.sidebar.warning("⚠ Tradier selezionato ma dati da yfinance (fallback). Controlla token/connessione.", icon="⚠️")
elif ds_src == "Tradier":
    st.sidebar.info(f"✅ Tradier: {sum(len(c) for c in chains)} opzioni in {len(chains)} expiry", icon="🔴")

# Debug diagnostica dati (sidebar)
n_rows_raw = sum(len(c) for c in chains) if chains else 0
has_gamma_txt = "✅" if ("gamma" in opt.columns and not opt.empty and opt["gamma"].sum() > 0) else "❌"
oi_sum_opt = opt["open_interest"].fillna(0).sum() if not opt.empty and "open_interest" in opt.columns else 0
vol_sum_opt = opt["volume"].fillna(0).sum() if not opt.empty and "volume" in opt.columns else 0
oi_sum_all = combined["open_interest"].fillna(0).sum() if not combined.empty and "open_interest" in combined.columns else 0
vol_sum_all = combined["volume"].fillna(0).sum() if not combined.empty and "volume" in combined.columns else 0
wtxt = f"OI:{oi_sum_opt:.0f}" if oi_sum_opt > 0 else f"Vol:{vol_sum_opt:.0f}" if vol_sum_opt > 0 else "Peso:gamma"
iv_m = 0
if not opt.empty and "implied_volatility" in opt.columns:
    ivs = opt["implied_volatility"].dropna()
    ivs = ivs[(ivs >= 0.05) & (ivs <= 2.0)]
    if not ivs.empty: iv_m = float(ivs.median())
ds = st.session_state.get("data_source", "Hybrid")
src_tag = {"Tradier":"🔴Tradier","GEXBot":"⚡GEXBot","Hybrid":"🐢yf+⚡"}.get(ds,"🐢yf")
if gexbot_gex is not None and len(gexbot_gex.get("strikes",[]))>0: src_tag = "⚡GEXBot"
s_debug = (f"{src_tag} Spot:{spot:.0f} DTE:{dte_use} {wtxt} IV:{iv_m:.0%}"
           + (f" R:{opt['strike'].min():.0f}–{opt['strike'].max():.0f}" if not opt.empty else " ⚠ vuoto")
           + f" | OIₐ:{oi_sum_all:,.0f} Volₐ:{vol_sum_all:,.0f}")
st.sidebar.caption(s_debug)
st.caption(f"Expiry disponibili: {dtes_avail[:8]}  |  DTE in uso: {dte_use}")
if dte_use <= 3:
    st.sidebar.info("📌 DTE ≤3: dati OI possono essere concentrati lontano dall'ATM. Prova DTE 7-14 per profilo più rappresentativo.", icon="💡")

# ═══════════════════════════════════════════════════════════════════════
# REGIME BANNER (stile GIP Dashboard)
# ═══════════════════════════════════════════════════════════════════════
vx=macro.get("vix","—"); v1=macro.get("vix1d","—"); v9=macro.get("vix9d","—"); vv=macro.get("vvix","—")
vn=macro.get("vxn","—")
dx=macro.get("dxy","—"); cu=macro.get("copper","—")
vx_s=f"{vx:.1f}" if isinstance(vx,float) else "—"
v1_s=f"{v1:.1f}" if isinstance(v1,float) else "—"
v9_s=f"{v9:.1f}" if isinstance(v9,float) else "—"
vv_s=f"{vv:.0f}" if isinstance(vv,float) else "—"
vvr = vv / vx if isinstance(vv,float) and isinstance(vx,float) and vx > 0 else None
vvr_s = f"{vvr:.1f}" if vvr else "—"
vvr_c = "#22c55e" if vvr and vvr < 4.5 else "#fbbf24" if vvr and vvr <= 6.5 else "#ef4444" if vvr else TEXT_M
vn_s=f"{vn:.1f}" if isinstance(vn,float) else "—"
dx_s=f"{dx:.1f}" if isinstance(dx,float) else "—"
cu_s=f"{cu:.2f}" if isinstance(cu,float) else "—"

vxn_proxy = None
if ds_src == "Tradier" and ticker_sel == "QQQ" and not combined.empty:
    vxn_proxy = vxn_from_chain(combined)
vxn_p_s = f"{vxn_proxy:.1%}" if vxn_proxy else "—"

gex_total = None
gex_opt_total = None
gex_tmp = None
if not combined.empty and "gamma" in combined.columns:
    gex_tmp = compute_gex(combined, spot)
    gex_total = gex_tmp.get("total_gex", 0)
elif not opt.empty and "gamma" in opt.columns:
    gex_tmp = compute_gex(opt, spot)
    gex_total = gex_tmp.get("total_gex", 0)
if not opt.empty and "gamma" in opt.columns:
    gex_opt_tmp = compute_gex(opt, spot)
    gex_opt_total = gex_opt_tmp.get("total_gex", 0)
gex_env = "POS" if gex_total and gex_total > 0 else "NEG" if gex_total and gex_total < 0 else "\u2014"
gex_opt_env = "POS" if gex_opt_total and gex_opt_total > 0 else "NEG" if gex_opt_total and gex_opt_total < 0 else None
gex_env_c = "#22c55e" if gex_env == "POS" else "#ef4444" if gex_env == "NEG" else TEXT_M

# γ-ENV Flip Alert
_init = st.session_state.get("_init_done", False)
prev_env = st.session_state.get("prev_gex_env")
if _init and prev_env and gex_env in ("POS","NEG") and gex_env != prev_env:
    flip_msg = f"γ-ENV FLIP: {prev_env} → {gex_env}"
    st.toast(flip_msg, icon="⚠")
    if audio_on:
        sr=22050; t=np.linspace(0,0.15,int(sr*0.15),False)
        beep=np.sin(880*2*np.pi*t); beep[-int(sr*0.02):]=0
        import base64, io, wave
        buf=io.BytesIO(); wf=wave.open(buf,'wb'); wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes((beep*32767).astype(np.int16).tobytes()); wf.close()
        st.audio(f"data:audio/wav;base64,{base64.b64encode(buf.getvalue()).decode()}", autoplay=True)
    try:
        with sqlite3.connect(str(DB_PATH)) as c:
            c.execute("INSERT INTO alerts(timestamp,type,message) VALUES(?,?,?)",(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"gamma_env_flip",flip_msg)); c.commit()
    except: pass
st.session_state["prev_gex_env"] = gex_env if gex_env in ("POS","NEG") else None

# Log SQLite (dopo GEX computation)
db=HistoryDB()
gex_zg = gex_tmp.get("zero_gamma", spot) if gex_tmp else None
vvr_log = round(vv / vx, 2) if isinstance(vv,float) and isinstance(vx,float) and vx > 0 else None
db.log(macro_reg, bias, shift, signals, gex_env if gex_total else None, gex_total, gex_zg, vvr_log)

oi_debug = ""
if not combined.empty and "gamma" in combined.columns:
    oi_c = "open_interest" in combined.columns
    vo_c = "volume" in combined.columns
    oi_s = combined["open_interest"].fillna(0).sum() if oi_c else 0
    vo_s = combined["volume"].fillna(0).sum() if vo_c else 0
    wt = "OI" if oi_s > 0 else "Vol" if vo_s > 0 else "γ(unwt)"
    atm_rows = combined[combined["strike"].between(spot*0.98, spot*1.02)]
    gam_avg = atm_rows["gamma"].mean() if not atm_rows.empty else 0
    gam_max = atm_rows["gamma"].max() if not atm_rows.empty else 0
    gex_db = gex_total if gex_total else 0
    oi_debug = f'<span style="font-size:8px;opacity:.5;margin-left:8px;">OI:{oi_s:,.0f} Vol:{vo_s:,.0f} [{wt}] γATMμ:{gam_avg:.5f} γATMmax:{gam_max:.5f} GEXₐ:{gex_db:,.0f}</span>'

st.markdown(f"""
<div style="background:{bg};padding:6px 16px;border-radius:6px;color:white;margin-bottom:6px;">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;font-size:11px;">
        <div><span style="font-size:9px;opacity:.8;">REGIME</span><span style="font-size:16px;font-weight:700;margin-left:6px;">{r_str}</span></div>
        <div><span style="font-size:9px;opacity:.8;">{ticker_sel}</span><span style="font-size:14px;font-weight:600;margin-left:4px;">{spot:.1f}</span></div>
        <div><span style="font-size:9px;opacity:.8;">CONF</span><span style="font-size:14px;font-weight:600;margin-left:4px;">{macro_reg.get("confidence",0):.1f}</span></div>
        <div><span style="font-size:9px;opacity:.8;">BIAS</span><span style="font-size:14px;font-weight:600;margin-left:4px;">{bias.value}</span></div>
        <div><span style="font-size:9px;opacity:.8;">\u03b3-ENV</span><span style="font-size:13px;font-weight:700;color:{gex_env_c};margin-left:2px;">{gex_env}</span>{f'<span style="font-size:9px;opacity:.6;margin-left:3px;">({dte_use}d:{gex_opt_env})</span>' if gex_opt_env and gex_opt_env != gex_env else ''}</div>
        <div><span style="font-size:9px;opacity:.8;">SHIFT</span><span style="font-size:14px;font-weight:600;color:{shift_color};margin-left:4px;">{shift_p*100:.0f}%</span></div>
        <div>VXN {vn_s} VXN* {vxn_p_s} | VIX1D {v1_s} VIX {vx_s} VIX9D {v9_s} VVIX {vv_s} <span style="color:{vvr_c};">VVR {vvr_s}</span> | DXY {dx_s} | Cu {cu_s}</div>
        <div style="font-size:10px;opacity:.7;">{datetime.now():%H:%M:%S}{oi_debug}</div>
    </div>
    {"<div style='margin-top:3px;font-size:10px;font-weight:600;'>⚠ REGIME SHIFT — preparare piano di contingenza</div>" if shift_p>.5 else ""}
</div>
""", unsafe_allow_html=True)

# ── CONTROLLI (ticker, sorgente, DTE, refresh) ────────────────────
c1,c2,c3,c4,c5,c6,c7=st.columns([.6,.6,.8,.8,.6,.4,.4],gap="small")
with c1: st.selectbox("Ticker",["SPY","QQQ","SPX","IWM","GLD"],key="ticker_sel",label_visibility="collapsed")
with c2: st.selectbox("Sorgente",["Tradier","Hybrid","GEXBot","yfinance"],key="ds_src",
                       help="Tradier: OPRA live via Tradier",label_visibility="collapsed")
with c3: st.slider("Scadenza (DTE)",0,60,step=1,key="dte_sel",label_visibility="collapsed")
with c4: st.slider("Strike Range %",5,30,step=5,key="sr",label_visibility="collapsed")
with c5: st.selectbox("Refresh",[5,15,30,60,300],index=1,key="refresh_sec",label_visibility="collapsed")
with c6: st.checkbox("Auto",True,key="auto")
with c7: st.checkbox("Alert",True,key="audio_on")

# ═══════════════════════════════════════════════════════════════════════
# CALENDARIO EVENTI
# ═══════════════════════════════════════════════════════════════════════
econ=EconCal(); econ_df=econ.upcoming(hours=48)
if not econ_df.empty:
    econ_df["datetime"]=pd.to_datetime(econ_df["datetime"])
    econ_df["Orario"]=econ_df["datetime"].dt.strftime("%a %d %b %H:%M")
    econ_df["Tra"]=econ_df["datetime"].apply(lambda x:f"{int((x-datetime.now()).total_seconds()//60)}min" if x>datetime.now() else "IN CORSO")
    st.dataframe(econ_df[["Orario","title","impact","Tra","source"]].rename(columns={"Orario":"Orario","title":"Evento","impact":"Impatto","Tra":"Tra","source":"Fonte"}),
                 width="stretch",hide_index=True,height=80)
    im=datetime.now()+timedelta(minutes=30)
    imm=econ_df[(econ_df["datetime"]>datetime.now())&(econ_df["datetime"]<=im)]
    if not imm.empty:
        st.error(f"⚠️ {len(imm)} evento/i entro 30min!")
else:
    st.info("Nessun evento ad alto impatto prossimo.")

# ═══════════════════════════════════════════════════════════════════════
# GRIGLIA 3-COLONNE
# ═══════════════════════════════════════════════════════════════════════
L,C,R=st.columns([1,1.4,1],gap="small")

# ── helpers ──────────────────────────────────────────────────────────
CHART_CFG={"displayModeBar":"hover", "scrollZoom":True, "responsive":True,
           "modeBarButtonsToRemove":["sendDataToCloud","lasso2d","select2d","hoverClosestCartesian","hoverCompareCartesian"],
           "modeBarButtonsToAdd":["drawline","eraseshape"],
           "toImageButtonOptions":{"format":"png","filename":"maze_chart","scale":2}}

def _fig(**kw):
    return go.Figure(**kw).update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_M, size=9), hovermode="x unified",
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        margin=dict(l=4,r=4,t=2,b=6),
        dragmode="zoom",
        **kw.pop("layout", {}))

# ═══════════════════════════════════════════════════════════════════════
# COLONNA SX — GEX|DEX / Bookmap / OI|Prem / Greeks
# ═══════════════════════════════════════════════════════════════════════
SR_V = 0.20
sr_lo = spot * (1 - SR_V)
sr_hi = spot * (1 + SR_V)
xax_range = [sr_lo, sr_hi]

with L:
    # 1 - GEX AGGREGATO (GEXBot/Tradier) | GEX per-DTE
    gb_col, yf_col = st.columns(2, gap="small")
    with gb_col:
        gex_agg = gexbot_gex if gexbot_gex and len(gexbot_gex.get("strikes",[])) > 0 else None
        gex_agg_src = "⚡"
        if not gex_agg and not combined.empty and "gamma" in combined.columns:
            _combined_filt = combined[combined["strike"].between(spot * 0.85, spot * 1.15)] if "strike" in combined.columns else combined
            gex_agg = compute_gex(_combined_filt, spot)
            gex_agg_src = "🔴" if ds_src == "Tradier" else "🐢"
        st.markdown(f'<div class="panel"><div class="ptitle"><span>GEX Aggregato</span><span style="color:{TEXT_M};">{gex_agg_src} tutte expiry</span></div>', unsafe_allow_html=True)
        if gex_agg and len(gex_agg.get("strikes",[])) > 0:
            fig=go.Figure()
            fig.add_trace(go.Bar(x=gex_agg["strikes"],y=gex_agg["gex"],marker_color=[GREEN if v>0 else RED for v in gex_agg["gex"]],opacity=.7,showlegend=False))
            fig.add_vline(x=gex_agg["zero_gamma"],line=dict(color=GOLD,width=1.5,dash="dash"),annotation_text=f"ZG {gex_agg['zero_gamma']:.0f}",annotation_font_size=8,annotation_font_color=GOLD)
            fig.add_vline(x=spot,line=dict(color="white",width=1.5))
            fig.update_layout(height=150,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=7),hovermode="x",bargap=.05,xaxis=dict(range=xax_range,showgrid=False,visible=True,title=None,tickfont=dict(size=6),tickangle=-45,nticks=8),yaxis=dict(showgrid=False,visible=False),margin=dict(l=2,r=2,t=2,b=18))
            st.plotly_chart(fig,width="stretch",config=CHART_CFG)
            st.markdown(f'<div style="font-size:8px;color:{TEXT_M};display:flex;justify-content:space-between;"><span>ZG {gex_agg["zero_gamma"]:.0f}</span><span>{gex_agg_src} {gex_agg["total_gex"]:+,.0f}</span></div>', unsafe_allow_html=True)
        else: st.caption("GEX N/D")
        st.markdown('</div>', unsafe_allow_html=True)
    with yf_col:
        st.markdown(f'<div class="panel"><div class="ptitle"><span>GEX {dte_use}D</span><span style="color:{TEXT_M};">yfinance</span></div>', unsafe_allow_html=True)
        # Filtra strike +15% da spot per GEX 0DTE (evita ZG fantasma)
        _opt_filt = opt[opt["strike"].between(spot * 0.85, spot * 1.15)] if not opt.empty and "strike" in opt.columns else opt
        gex_dte = compute_gex(_opt_filt, spot) if not _opt_filt.empty and "gamma" in _opt_filt.columns else None
        if gex_dte and len(gex_dte["strikes"]) > 0:
            fig=go.Figure()
            fig.add_trace(go.Bar(x=gex_dte["strikes"],y=gex_dte["gex"],marker_color=[GREEN if v>0 else RED for v in gex_dte["gex"]],opacity=.7,showlegend=False))
            fig.add_vline(x=gex_dte["zero_gamma"],line=dict(color=GOLD,width=1.5,dash="dash"),annotation_text=f"ZG {gex_dte['zero_gamma']:.0f}",annotation_font_size=8,annotation_font_color=GOLD)
            fig.add_vline(x=spot,line=dict(color="white",width=1.5))
            fig.update_layout(height=150,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=7),hovermode="x",bargap=.05,xaxis=dict(range=xax_range,showgrid=False,visible=True,title=None,tickfont=dict(size=6),tickangle=-45,nticks=8),yaxis=dict(showgrid=False,visible=False),margin=dict(l=2,r=2,t=2,b=18))
            st.plotly_chart(fig,width="stretch",config=CHART_CFG)
            st.markdown(f'<div style="font-size:8px;color:{TEXT_M};display:flex;justify-content:space-between;"><span>ZG {gex_dte["zero_gamma"]:.0f}</span><span>{gex_dte["total_gex"]:+,.0f}</span></div>', unsafe_allow_html=True)
        else: st.caption("N/D")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # GEX Regime Heatmap
    with st.expander("🌡️ GEX Regime Heatmap", expanded=False):
        try:
            prices, days, gex_mat = gex_regime_heatmap(combined, spot)
            if gex_mat is not None:
                gex_norm = np.clip(gex_mat, np.percentile(gex_mat, 5), np.percentile(gex_mat, 95))
                fig = go.Figure(go.Heatmap(
                    x=[f"+{d}d" for d in days],
                    y=[f"{p:.0f}" for p in prices],
                    z=gex_norm,
                    colorscale=[
                        [0.0,  "#a855f7"],
                        [0.45, "#1a0a2e"],
                        [0.5,  "#000000"],
                        [0.55, "#0a1628"],
                        [1.0,  "#3b82f6"],
                    ],
                    zmid=0, showscale=True,
                    colorbar=dict(title=dict(text="GEX", side="right"), thickness=10, len=0.8,
                                  tickfont=dict(size=9),
                    tickvals=[float(np.percentile(gex_norm, 10)), 0.0, float(np.percentile(gex_norm, 90))],
                    ticktext=["NEG", "0", "POS"]),
                    hoverongaps=False,
                    hovertemplate="Prezzo: %{y}<br>Giorno: %{x}<br>GEX: %{z:+,.0f}<extra></extra>",
                ))
                fig.add_hline(y=float(spot), line_color="white", line_width=1, line_dash="dash",
                              annotation_text=f"Spot {spot:.0f}", annotation_font_color="white", annotation_font_size=9)
                zg_val = gex_zg if gex_zg else None
                if zg_val:
                    fig.add_hline(y=float(zg_val), line_color="#fbbf24", line_width=1, line_dash="dot",
                                  annotation_text=f"ZG {zg_val:.0f}", annotation_font_color="#fbbf24", annotation_font_size=9)
                fig.update_layout(height=320, margin=dict(l=50, r=60, t=30, b=30),
                                  paper_bgcolor="#0a0e17", plot_bgcolor="#0a0e17",
                                  font=dict(color="#64748b", size=9),
                                  title=dict(text=f"GEX Regime | {ticker_sel} | Blu=POS · Viola=NEG",
                                             font=dict(size=10, color="#94a3b8"), x=0),
                                  xaxis=dict(title="Giorni futuri", tickfont=dict(size=8), gridcolor="#1f2937"),
                                  yaxis=dict(title="Prezzo simulato", tickfont=dict(size=8), gridcolor="#1f2937"))
                st.plotly_chart(fig, width="stretch")
                col1, col2, col3 = st.columns(3)
                col1.metric("Regime attuale", gex_env, delta=f"ZG {zg_val:.0f}" if zg_val else "")
                col2.metric("GEX totale", f"{gex_total:+,.0f}")
                col3.metric("Spot vs ZG", f"{((spot/zg_val)-1)*100:+.1f}%" if zg_val else "N/D")
            else:
                st.caption("Dati insufficienti per heatmap regime")
        except Exception as e:
            st.caption(f"Heatmap non disponibile: {e}")

    # Exposure Profile — tabella Greeks per strike (rossa=put, verde=call)
    st.markdown(f'<div class="panel"><div class="ptitle"><span>Gamma Profile</span><span style="color:{TEXT_M};">GEX · Δ · Charm</span></div>', unsafe_allow_html=True)
    try:
        exp_df = compute_exposure_profile(combined, spot) if not combined.empty else pd.DataFrame()
        if exp_df.empty:
            st.caption("Dati insufficienti per Gamma Profile")
        else:
            near = exp_df["strike"].between(spot * 0.95, spot * 1.05)
            df_tbl = exp_df[near].copy().sort_values("strike")
            if df_tbl.empty:
                st.caption("Nessuno strike nel range ±5%")
            else:
                max_gex = max(df_tbl["gex_net"].abs().max(), 1)
                max_dex = max(df_tbl["dex_net"].abs().max(), 1)
                max_charm = max(df_tbl["charm_net"].abs().max(), 1)

                def _gex_color(v):
                    pct = min(abs(v) / max_gex, 1)
                    if v > 0:
                        return f"rgba(34,{max(int(197*(1-pct)),30)},94,{0.15+0.6*pct})"
                    elif v < 0:
                        return f"rgba(239,{max(int(68*(1-pct)),30)},{max(int(68*(1-pct)),30)},{0.15+0.6*pct})"
                    return "transparent"

                def _dex_color(v):
                    pct = min(abs(v) / max_dex, 1)
                    if v > 0:
                        return f"rgba(34,{max(int(197*(1-pct)),30)},94,{0.1+0.5*pct})"
                    elif v < 0:
                        return f"rgba(239,{max(int(68*(1-pct)),30)},{max(int(68*(1-pct)),30)},{0.1+0.5*pct})"
                    return "transparent"

                def _charm_color(v):
                    pct = min(abs(v) / max_charm, 1)
                    if v > 0:
                        return f"rgba(34,{max(int(197*(1-pct)),30)},94,{0.1+0.5*pct})"
                    elif v < 0:
                        return f"rgba(239,{max(int(68*(1-pct)),30)},{max(int(68*(1-pct)),30)},{0.1+0.5*pct})"
                    return "transparent"

                def _style(val, cfunc):
                    if val == 0:
                        return "background:transparent;color:#64748b"
                    bg = cfunc(val)
                    txt = "#22c55e" if val > 0 else "#ef4444"
                    return f"background:{bg};color:{txt};font-weight:600"

                tbl = df_tbl.copy()
                tbl.columns = [c.replace("gex_","GEX ").replace("dex_","Δ ").replace("charm_","Charm ").replace("oi_","OI ") for c in tbl.columns]
                has_charm_ok = tbl["Charm net"].abs().max() > 0

                styled = tbl.style.map(lambda v: _style(v, _gex_color), subset=["GEX net"])
                styled = styled.map(lambda v: _style(v, _dex_color), subset=["Δ net"])
                if has_charm_ok:
                    styled = styled.map(lambda v: _style(v, _charm_color), subset=["Charm net"])

                fmt = {
                    "GEX call":"{:+,.0f}","GEX put":"{:+,.0f}","GEX net":"{:+,.0f}",
                    "Δ call":"{:+,.0f}","Δ put":"{:+,.0f}","Δ net":"{:+,.0f}",
                    "OI call":"{:,.0f}","OI put":"{:,.0f}","OI total":"{:,.0f}",
                    "strike":"{:,.2f}",
                }
                if has_charm_ok:
                    fmt.update({"Charm call":"{:+,.6f}","Charm put":"{:+,.6f}","Charm net":"{:+,.6f}"})
                else:
                    tbl = tbl.drop(columns=["Charm call","Charm put","Charm net"])
                styled = styled.set_properties(**{"font-size":"10px","text-align":"center","white-space":"nowrap"})
                styled = styled.format(fmt)
                st.dataframe(styled, height=350, width="stretch")

    except Exception as e:
        st.caption(f"Gamma Profile N/D: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    # 2 - DEX (full width, sotto i due GEX)
    st.markdown(f'<div class="panel"><div class="ptitle"><span>DEX</span><span style="color:{TEXT_M};">delta</span></div>', unsafe_allow_html=True)
    if not opt.empty and "delta" in opt.columns:
        dex=compute_dex(opt,spot)
        if len(dex["strikes"])>0:
            fig=go.Figure()
            fig.add_trace(go.Bar(x=dex["strikes"],y=dex["call_dex"],marker_color=GREEN,opacity=.5,showlegend=False))
            fig.add_trace(go.Bar(x=dex["strikes"],y=-dex["put_dex"],marker_color=RED,opacity=.5,showlegend=False))
            fig.add_trace(go.Scatter(x=dex["strikes"],y=dex["dex"],mode="lines+markers",line=dict(color=GOLD,width=1.5),marker=dict(size=3,color=GOLD),showlegend=False))
            fig.add_vline(x=spot,line=dict(color="white",width=1,dash="dot"))
            fig.update_layout(height=100,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=7),hovermode="x",bargap=.05,barmode="overlay",xaxis=dict(range=xax_range,showgrid=False,visible=True,title=None,tickfont=dict(size=6),tickangle=-45,nticks=8),yaxis=dict(showgrid=False,visible=False),margin=dict(l=2,r=2,t=2,b=18))
            st.plotly_chart(fig,width="stretch",config=CHART_CFG)
            st.markdown(f'<div style="font-size:8px;color:{TEXT_M};display:flex;justify-content:space-between;"><span>CW {dex["call_wall"]:.0f}</span><span>Net {dex["total_dex"]:+,.0f}</span></div>', unsafe_allow_html=True)
    else: st.caption("N/D")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 3 - BOOKMAP
    st.markdown(f'<div class="panel"><div class="ptitle"><span>DOM Bookmap</span><span style="color:{TEXT_M};">liquidita</span></div>', unsafe_allow_html=True)
    if not opt.empty:
        dom=dom_bookmap(opt,spot)
        if len(dom["levels"])>0:
            fig=go.Figure()
            fig.add_trace(go.Bar(x=dom["levels"],y=dom["liquidity"],marker_color=[GREEN if k>spot else RED for k in dom["levels"]],opacity=.3,showlegend=False))
            fig.add_trace(go.Scatter(x=dom["levels"],y=dom["liquidity"],mode="lines",line=dict(color="rgba(255,255,255,0.1)",width=.5),showlegend=False,hoverinfo="skip"))
            if dom["big_trades"]:
                bt=dom["big_trades"][:8]; sz=np.clip(np.array([b["size"] for b in bt])*.8,4,20)
                fig.add_trace(go.Scatter(x=[b["strike"] for b in bt],y=[b["size"] for b in bt],mode="markers+text",marker=dict(size=sz,color=GOLD,symbol="circle",line=dict(width=.5,color="white"),opacity=.8),text=[f'${b["size"]:,.0f}' for b in bt],textfont=dict(size=7,color="white"),textposition="middle center",showlegend=False,hoverinfo="skip"))
            fig.add_vline(x=spot,line=dict(color="white",width=1,dash="dash"))
            fig.update_layout(height=110,margin=dict(l=2,r=2,t=2,b=18),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=7),bargap=.05,xaxis=dict(range=xax_range,showgrid=False,visible=True,tickfont=dict(size=6),tickangle=-45,nticks=8),yaxis=dict(showgrid=False,visible=True,tickfont=dict(size=6),tickprefix="$",tickformat=",.0f"))
            st.plotly_chart(fig,width="stretch",config=CHART_CFG)
    else: st.caption("N/D")
    st.markdown('</div>', unsafe_allow_html=True)

    # 4 - OI | PREMIUM affiancati
    o1,o2=st.columns(2,gap="small")
    with o1:
        st.markdown(f'<div class="panel"><div class="ptitle"><span>OI</span><span style="color:{TEXT_M};">open int.</span></div>', unsafe_allow_html=True)
        if not opt.empty:
            oi=open_interest_profile(opt,spot)
            if len(oi["strikes"])>0:
                fig=go.Figure()
                fig.add_trace(go.Bar(x=oi["strikes"],y=oi["call_oi"],marker_color=GREEN,opacity=.5,showlegend=False))
                fig.add_trace(go.Bar(x=oi["strikes"],y=oi["put_oi"],marker_color=RED,opacity=.5,showlegend=False))
                if oi.get("max_oi_strike"): fig.add_vline(x=oi["max_oi_strike"],line=dict(color=GOLD,width=1,dash="dash"),annotation_text=f"Max {oi['max_oi_strike']:.0f}",annotation_font_size=7,annotation_font_color=GOLD)
                fig.add_vline(x=spot,line=dict(color="white",width=1,dash="dot"))
                fig.update_layout(barmode="group",height=100,margin=dict(l=2,r=2,t=2,b=18),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=7),bargap=.05,xaxis=dict(range=xax_range,showgrid=False,visible=True,tickfont=dict(size=6),tickangle=-45,nticks=8),yaxis=dict(showgrid=False,visible=False))
                st.plotly_chart(fig,width="stretch",config=CHART_CFG)
        else: st.caption("N/D")
        st.markdown('</div>', unsafe_allow_html=True)
    with o2:
        st.markdown(f'<div class="panel"><div class="ptitle"><span>Premium</span><span style="color:{TEXT_M};">flow</span></div>', unsafe_allow_html=True)
        if not opt.empty:
            pf=premium_flow(opt,spot)
            if pf["by_strike"]:
                pdf=pd.DataFrame(pf["by_strike"])
                if not pdf.empty and not pdf["strike"].isna().all():
                    strike_diff=(pd.to_numeric(pdf["strike"],errors="coerce")-spot).abs()
                    if strike_diff.isna().all():
                        st.caption("Premium N/D")
                    else:
                        ai=strike_diff.idxmin()
                        fig=go.Figure()
                        fig.add_trace(go.Bar(x=pdf["strike"],y=pdf["call_premium"],marker_color=[GOLD if i==ai else GREEN for i in pdf.index],opacity=.6,showlegend=False))
                        fig.add_trace(go.Bar(x=pdf["strike"],y=pdf["put_premium"],marker_color=[ORANGE if i==ai else RED for i in pdf.index],opacity=.6,showlegend=False))
                        fig.update_layout(barmode="overlay",height=100,margin=dict(l=2,r=2,t=2,b=4),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=7),hovermode="x",xaxis=dict(showgrid=False,visible=False),yaxis=dict(showgrid=False,visible=False))
                        st.plotly_chart(fig,width="stretch",config=CHART_CFG)
                        st.markdown(f'<div style="font-size:8px;color:{TEXT_M};display:flex;justify-content:space-between;"><span>P/C {pf["ratio"]:.2f}</span><span>Put ${pf["total_put"]:,.0f}</span><span>Call ${pf["total_call"]:,.0f}</span></div>', unsafe_allow_html=True)
                else:
                    st.caption("Premium N/D")
        st.markdown('</div>', unsafe_allow_html=True)

    # 5 - GREEKS (usa dte_sel dalla sidebar)
    st.markdown(f'<div class="panel"><div class="ptitle"><span>Greeks</span><span style="color:{TEXT_M};">ATM</span></div>', unsafe_allow_html=True)
    if not combined.empty and "gamma" in combined.columns:
        dtes=sorted(combined["dte"].unique())
        dte_use=min(dtes,key=lambda d:abs(d-dte_sel))
        sub=combined[combined["dte"]==dte_use]
        if not sub.empty:
            Ka=float(sub.iloc[(sub["strike"]-spot).abs().argsort()[:1]]["strike"].iloc[0])
            gk=compute_all_greeks(spot,Ka,max(dte_use,1)/365,.05,.2,"call")
            c1,c2,c3,c4=st.columns(4); c1.metric("D",f"{gk['delta']:.3f}"); c2.metric("G",f"{gk['gamma']:.5f}"); c3.metric("V",f"{gk['vega']:.3f}"); c4.metric("T",f"{gk['theta']:.4f}")
            c1,c2,c3,c4=st.columns(4); c1.metric("Vanna",f"{gk['vanna']:.4f}"); c2.metric("Vomma",f"{gk['vomma']:.4f}"); c3.metric("Charm",f"{gk['charm']:.5f}"); c4.metric("Speed",f"{gk['speed']:.6f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # 4b - GAMMA BANDS (live per ticker)
    st.markdown(f'<div class="panel"><div class="ptitle"><span>Gamma Bands</span><span style="color:{TEXT_M};">{ticker_sel}</span></div>', unsafe_allow_html=True)
    src_lbl = "yfinance"
    gb = None
    gb_spot_use = st.session_state.get("gexbot_spot") or spot
    if gexbot_gex and len(gexbot_gex.get("strikes",[])) > 0:
        gb = gamma_bands_from_gexbot(gexbot_gex, gb_spot_use)
        src_lbl = "⚡GEXBot"
    elif not combined.empty and "gamma" in combined.columns:
        gb = gamma_bands(combined, spot)

    if gb is not None and not gb.empty:
        c_lo = gb["livello_lo"].min()
        c_hi = gb["livello_hi"].max()
        fig = go.Figure()
        border = {"^":"rgba(34,197,94,0.7)", "v":"rgba(239,68,68,0.7)", "-":"rgba(255,255,255,0.3)",
                  "▲":"rgba(34,197,94,0.7)", "▼":"rgba(239,68,68,0.7)", "●":"rgba(255,255,255,0.3)"}
        for _, r in gb.iterrows():
            fig.add_trace(go.Scatter(
                x=[r["livello_lo"], r["livello_hi"]],
                y=[0, 0],
                mode="lines+markers",
                line=dict(color=border.get(r["direzione"], "#888"), width=4),
                marker=dict(size=6, color=border.get(r["direzione"], "#888"), symbol="diamond"),
                name=r["zona"],
                hovertext=f"{r['zona']}<br>{r['livello_lo']:.0f} – {r['livello_hi']:.0f}<br>{r['nota']}",
                hoverinfo="text",
                showlegend=False
            ))
            fig.add_annotation(x=r["centro"], y=0, text=r["zona"].split(" ")[0],
                               font=dict(size=6, color=TEXT_M), showarrow=False, yshift=10)
        fig.add_vline(x=spot, line=dict(color="white", width=1.5),
                      annotation_text=f"{ticker_sel} {spot:.0f}", annotation_font_size=8)
        fig.update_layout(
            height=140,
            xaxis=dict(range=[c_lo - 5, c_hi + 5], showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                       tickfont=dict(size=7, color=TEXT_M), title=None),
            yaxis=dict(visible=False, range=[-0.5, 0.5]),
            margin=dict(l=4, r=4, t=2, b=6),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            hovermode="closest", dragmode="zoom"
        )
        st.plotly_chart(fig, width="stretch", config=CHART_CFG)
        cols = st.columns(9)
        for i, (_, r) in enumerate(gb.iterrows()):
            clr = {"^": GREEN, "v": RED, "-": TEXT_M}.get(r["direzione"], TEXT_M)
            with cols[i]:
                st.markdown(f'<div style="font-size:7px;color:{clr};text-align:center;padding:1px 0;">'
                            f'{r["zona"].split(" ")[0]}<br><b>{r["centro"]:.0f}</b></div>',
                            unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 5 - MGT-GO style panel (grafico candlestick + GEX overlay)
    with st.expander("📊 MGT-GO Style Chart", expanded=True):
        mgtgo_gex = gexbot_gex if gexbot_gex and len(gexbot_gex.get("strikes",[])) > 0 else gex_agg

        hm_data = None
        try:
            prices, days, matrix = gex_regime_heatmap(combined, spot)
            if matrix is not None:
                hm_data = (prices, days, matrix)
        except Exception:
            pass

        saved_tf = st.session_state.get("mgtgo_tf", "6M")

        render_mgtgo(
            st, ticker_sel, spot, mgtgo_gex, price_data,
            ohlc_bars=ohlc_bars,
            gex_env=gex_env,
            chart_height=st.session_state.chart_height,
            default_tf=saved_tf,
            heatmap_data=hm_data,
        )


# ═══════════════════════════════════════════════════════════════════════
# COLONNA CENTRO — Regression / Surface / Macro / GARCH / Centroid
# ═══════════════════════════════════════════════════════════════════════
with C:
    # 6 - REGRESSION
    st.markdown(f'<div class="panel"><div class="ptitle"><span>Regression + VP</span><span style="color:{TEXT_M};">89/34/13d</span></div>', unsafe_allow_html=True)
    if price_data is None:
        st.caption("Prezzi N/D (yfinance)")
    elif len(price_data) <= 50:
        st.caption(f"Prezzi insufficienti ({len(price_data)} barre)")
    else:
        xp=np.arange(len(price_data))
        vp=volume_profile(prices=price_data, option_df=combined if not combined.empty else None, spot=spot, n_bins=30)
        # Price + Regression
        fig1=go.Figure()
        fig1.add_trace(go.Scatter(x=xp,y=price_data,mode="lines",line=dict(color="white",width=1.5),showlegend=False))
        for w in [89,34,13]:
            if len(price_data)>=w:
                lr=linreg_channels(price_data,window=w); f=lr["slope"]*(xp[-w:]-w+1)+lr["intercept"]
                fig1.add_trace(go.Scatter(x=xp[-w:],y=f,mode="lines",line=dict(color=ACCENT,width=.8),showlegend=False))
                for n,c in [(1,"rgba(34,197,94,0.2)"),(2,"rgba(251,191,36,0.12)"),(3,"rgba(239,68,68,0.08)")]:
                    u=lr["channels"][f"+{n}sigma"]; lo=lr["channels"][f"-{n}sigma"]
                    fig1.add_trace(go.Scatter(x=list(xp[-w:])+list(xp[-w:][::-1]),y=list(u)+list(lo[::-1]),fill="toself",mode="none",fillcolor=c,showlegend=False,hoverinfo="skip"))
        lr=linreg_channels(price_data,window=min(89,len(price_data)))
        mf=lr["slope"]*(xp[-min(89,len(price_data)):]-min(89,len(price_data))+1)+lr["intercept"]
        ps=(price_data[-1]-mf[-1])/lr["std"]
        fig1.add_hline(y=price_data[-1],line=dict(color="white",width=.8,dash="dot"))
        fig1.add_hline(y=vp["poc"], line=dict(color=ACCENT, width=1, dash="dot"),
                       annotation_text=f"POC {vp['poc']:.0f}", annotation_font_size=7, annotation_font_color=ACCENT)
        if vp["vah"]!=vp["val"]:
            fig1.add_hrect(y0=vp["val"], y1=vp["vah"], line_width=0, fillcolor="rgba(0,210,255,0.04)")
        # Box‑Plot overlay (IQR Q25–Q75, ±2σ, ±3σ)
        bp = box_plot_stats(price_data)
        if bp:
            fig1.add_hrect(y0=bp["q25"], y1=bp["q75"], line_width=0,
                           fillcolor="rgba(34,197,94,0.08)",
                           annotation_text=f"IQR {bp['q25']:.0f}–{bp['q75']:.0f}",
                           annotation_font_size=7, annotation_font_color=GREEN)
        zs = z_score_analysis(price_data[-1], price_data)
        if zs and zs["std"] > 1:
                for s, c, ls in [(2, ORANGE, "dash"), (3, RED, "dot")]:
                    fig1.add_hline(y=zs["mean"] + s * zs["std"],
                                   line=dict(color=c, width=0.8, dash=ls))
                    fig1.add_hline(y=zs["mean"] - s * zs["std"],
                                   line=dict(color=c, width=0.8, dash=ls))
        fig1.update_layout(height=180,margin=dict(l=4,r=4,t=2,b=2),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=8),hovermode="x",xaxis=dict(showgrid=False,visible=False),yaxis=dict(showgrid=False,visible=False),dragmode="zoom")
        st.plotly_chart(fig1,width="stretch",config=CHART_CFG)
        # VP bar chart sotto
        dist=vp["distribution"]
        if dist:
            mx=max(d["volume"] for d in dist)
            vp_c=[ACCENT if d["is_va"] else "rgba(255,255,255,0.15)" for d in dist]
            fig2=go.Figure()
            fig2.add_trace(go.Bar(x=[d["volume"] for d in dist],y=[d["price"] for d in dist],
                                  orientation="h", marker_color=vp_c, width=.6,
                                  showlegend=False, opacity=.7))
            fig2.add_hline(y=spot, line=dict(color="white",width=1,dash="dash"))
            fig2.update_layout(height=70,margin=dict(l=4,r=4,t=2,b=2),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=7),xaxis=dict(showgrid=False,visible=False),yaxis=dict(showgrid=False,visible=False),dragmode="zoom",bargap=.05)
            st.plotly_chart(fig2,width="stretch",config=CHART_CFG)
        st.markdown(f'<div style="font-size:9px;color:{TEXT_M};display:flex;justify-content:space-between;"><span>POC {vp["poc"]:.0f}</span><span>VA {vp["val"]:.0f}-{vp["vah"]:.0f}</span><span>Slope {lr["slope"]:.4f}</span><span style="color:{GREEN if ps>0 else RED};">{ps:+.2f}s</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 7 - 3D SURFACE | MACRO GIP affiancati
    s1,s2=st.columns(2,gap="small")
    with s1:
        st.markdown(f'<div class="panel"><div class="ptitle"><span>3D Surface</span><span style="color:{TEXT_M};">DTExStrike</span></div>', unsafe_allow_html=True)
        if chains and len(chains)>1:
            from spx_quant import PDFExtractor, PDFCollector
            pdf_ext=PDFExtractor(); pdfs=[]
            for chain in chains:
                p=pdf_ext.extract(chain,spot)
                if p is not None: pdfs.append(p)
            if pdfs:
                K,dtes,pg=PDFCollector().collect(pdfs,spot)
                if K is not None:
                    bins=np.arange(K[0],K[-1]+50,50); bc=(bins[:-1]+bins[1:])/2
                    pg2=np.zeros((len(dtes),len(bc)))
                    for i in range(len(dtes)):
                        for j,(l,r) in enumerate(zip(np.searchsorted(K,bins[:-1]),np.searchsorted(K,bins[1:]))):
                            if r>l: pg2[i,j]=simpson(pg[i][l:r],K[l:r])*100
                    Km,Dm=np.meshgrid(bc,dtes)
                    fig=go.Figure(data=[go.Surface(x=Km,y=Dm,z=pg2,colorscale="Viridis",opacity=.85,colorbar=dict(thickness=6,len=.4,x=1.02,title=dict(text="%",font=dict(size=7))))])
                    fig.add_trace(go.Scatter3d(x=[spot]*len(dtes),y=dtes,z=[0]*len(dtes),mode="lines+markers",line=dict(color="red",width=3),marker=dict(size=3,color="red"),showlegend=False))
                    fig.update_layout(height=180,margin=dict(l=0,r=0,t=2,b=0),paper_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=7),scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),bgcolor="rgba(0,0,0,0)",camera=dict(eye=dict(x=1.5,y=-1.5,z=.8))))
                    st.plotly_chart(fig,width="stretch",config={**CHART_CFG,"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="panel"><div class="ptitle"><span>Macro GIP</span><span style="color:{TEXT_M};">Growth/Inflation</span></div>', unsafe_allow_html=True)
        gv=gip.get("G",0); iv=gip.get("I",0)
        fig=go.Figure()
        fig.add_trace(go.Bar(x=["G","I"],y=[gv,iv],marker_color=[GREEN if gv>0 else RED,ORANGE if iv>0 else "#3b82f6"],text=[f"{gv:+.3f}",f"{iv:+.3f}"],textposition="outside"))
        fig.update_layout(height=90,margin=dict(l=4,r=4,t=2,b=4),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=7),yaxis=dict(range=[-1,1],visible=False),xaxis=dict(visible=False))
        st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})
        gtag="G+" if gv>0 else "G-"; itag="I-" if iv<-.1 else "I+" if iv>.1 else "I="
        st.markdown(f'<div style="font-size:9px;color:{TEXT_M};text-align:center;">{gtag} {itag} <b style="color:{rc};">{r_str}</b> C{macro_reg.get("confidence",0):.2f}</div>', unsafe_allow_html=True)
        if shift["signals"]:
            for s in shift["signals"][:2]: st.markdown(f'<div style="font-size:8px;color:{GOLD};padding:1px 0;">{s}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 8 - GARCH MC
    st.markdown(f'<div class="panel"><div class="ptitle"><span>GARCH MC</span><span style="color:{TEXT_M};">Monte Carlo</span></div>', unsafe_allow_html=True)
    if price_data is not None and len(price_data)>100:
        paths,finals,_=garch_mc_paths(price_data,n_paths=500,n_steps=30)
        conf=monte_carlo_confidence(paths); t=np.arange(30)
        fig=go.Figure()
        for lo,hi,c in [(5,95,"rgba(0,210,255,0.06)"),(25,75,"rgba(0,210,255,0.12)")]:
            fig.add_trace(go.Scatter(x=list(t)+list(t[::-1]),y=list(conf[lo])+list(conf[hi][::-1]),fill="toself",mode="none",fillcolor=c,showlegend=False))
        idx=np.random.default_rng(42).choice(len(paths),min(30,len(paths)),replace=False)
        for i in idx: fig.add_trace(go.Scatter(x=t,y=paths[i],mode="lines",line=dict(width=.4,color="rgba(0,210,255,0.15)"),showlegend=False,hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=t,y=conf[50],mode="lines",line=dict(color=ACCENT,width=1.5),showlegend=False))
        fig.add_hline(y=price_data[-1],line=dict(color="white",width=1,dash="dash"))
        fig.update_layout(height=120,margin=dict(l=4,r=4,t=2,b=6),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=8),xaxis=dict(showgrid=False,visible=False),yaxis=dict(showgrid=False,visible=False),dragmode="zoom")
        st.plotly_chart(fig,width="stretch",config=CHART_CFG)
        pv=np.percentile(finals,[5,25,50,75,95])
        st.markdown(f'<div style="font-size:8px;color:{TEXT_M};display:flex;justify-content:space-between;"><span>P5 {pv[0]:.0f}</span><span>P25 {pv[1]:.0f}</span><span style="color:{ACCENT};">P50 {pv[2]:.0f}</span><span>P75 {pv[3]:.0f}</span><span>P95 {pv[4]:.0f}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 9 - GEX CENTROID
    st.markdown(f'<div class="panel"><div class="ptitle"><span>GEX Centroid</span><span style="color:{TEXT_M};">gamma per DTE</span></div>', unsafe_allow_html=True)
    if not combined.empty and "gamma" in combined.columns:
        cm=gex_centroid_map(combined,spot)
        if cm["centroids"]:
            ds=sorted(cm["centroids"].keys()); cents=[cm["centroids"][d] for d in ds]
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=ds,y=cents,mode="lines+markers",line=dict(color=ACCENT,width=2),marker=dict(size=6,color=ACCENT,symbol="diamond"),showlegend=False))
            fig.add_hline(y=spot,line=dict(color="white",width=1,dash="dash"))
            fig.update_layout(height=80,margin=dict(l=2,r=2,t=2,b=2),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=7),hovermode="x",xaxis=dict(showgrid=False,visible=False),yaxis=dict(showgrid=False,visible=False))
            st.plotly_chart(fig,width="stretch",config=CHART_CFG)
            nd=min(ds,key=lambda d:abs(d-dte_sel))
            st.markdown(f'<div style="font-size:8px;color:{TEXT_M};display:flex;justify-content:space-between;"><span>{int(nd)}d: {cm["centroids"][nd]:.0f}</span><span>vs SPOT: {cm["centroids"][nd]-spot:+.1f}</span></div>', unsafe_allow_html=True)
    else: st.caption("N/D")
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# COLONNA DX — Chain / PDF / Walls / Corr / MMM / Anomalies
# ═══════════════════════════════════════════════════════════════════════
with R:
    # 10 - CHAIN (usa dte_sel dalla sidebar, con delta ATM)
    st.markdown(f'<div class="panel"><div class="ptitle"><span>Options Chain</span><span style="color:{TEXT_M};">{dte_sel}d</span></div>', unsafe_allow_html=True)
    if not combined.empty and "delta" in opt.columns:
        srp=sr/100.0
        opt2=opt[(opt["strike"]>=spot*(1-srp))&(opt["strike"]<=spot*(1+srp))].copy()
        if not opt2.empty:
            type_col_c = "option_type" if "option_type" in opt2.columns else ("type" if "type" in opt2.columns else None)
            if type_col_c:
                opt2["type"] = opt2[type_col_c].str.lower().str[0]
            px=opt2.pivot_table(index="strike",columns="type",values="last_price",aggfunc="mean").reset_index()
            dl=opt2.pivot_table(index="strike",columns="type",values="delta",aggfunc="mean").reset_index()
            px.columns=["strike","call","put"]
            dl.columns=["strike","delta_c","delta_p"]
            pv=px.merge(dl,on="strike",how="left")
            atm_strike=pv.loc[(pv["strike"]-spot).abs().idxmin(),"strike"]
            pv["strike_s"]=pv["strike"].apply(lambda x:f"◀ {x:.0f}" if abs(x-atm_strike)<1 else f"{x:.0f}")
            if "implied_volatility" in opt2.columns:
                ivc=opt2.groupby("strike")["implied_volatility"].mean().reset_index()
                pv=pv.merge(ivc,on="strike",how="left")
                pv["iv"]=pv["implied_volatility"].apply(lambda x:f"{x:.1%}" if pd.notna(x) else "-")
            pv["call"]=pv["call"].apply(lambda x:f"${x:.2f}" if pd.notna(x) else "-")
            pv["put"]=pv["put"].apply(lambda x:f"${x:.2f}" if pd.notna(x) else "-")
            pv["δc"]=pv["delta_c"].apply(lambda x:f"{x:.2f}" if pd.notna(x) else "-")
            pv["δp"]=pv["delta_p"].apply(lambda x:f"{x:.2f}" if pd.notna(x) else "-")
            st.dataframe(pv[["strike_s","call","δc","put","δp","iv"]].rename(columns={"strike_s":"Strike","δc":"ΔC","δp":"ΔP"}),width="stretch",hide_index=True,height=120)
    st.markdown('</div>', unsafe_allow_html=True)

    # Volatility Skew
    st.markdown(f'<div class="panel"><div class="ptitle"><span>Volatility Skew</span><span style="color:{TEXT_M};">{dte_use}d</span></div>', unsafe_allow_html=True)
    if not opt.empty and "implied_volatility" in opt.columns and "option_type" in opt.columns:
        sk = opt[opt["implied_volatility"].notna() & (opt["implied_volatility"]>=0.05) & (opt["implied_volatility"]<=2.0)].copy()
        if not sk.empty:
            sk = sk[sk["strike"].between(spot*0.9, spot*1.1)].copy()
            if not sk.empty:
                sk["type"] = sk["option_type"].str.lower().str[0]
                calls = sk[sk["type"]=="c"].groupby("strike")["implied_volatility"].mean().reset_index()
                puts = sk[sk["type"]=="p"].groupby("strike")["implied_volatility"].mean().reset_index()
                fig_sk = go.Figure()
                fig_sk.add_trace(go.Scatter(x=calls["strike"], y=calls["implied_volatility"], mode="lines+markers", name="Call IV", line=dict(color="#22c55e", width=1.5), marker=dict(size=3)))
                fig_sk.add_trace(go.Scatter(x=puts["strike"], y=puts["implied_volatility"], mode="lines+markers", name="Put IV", line=dict(color="#ef4444", width=1.5), marker=dict(size=3)))
                fig_sk.add_vline(x=spot, line=dict(color="white", width=1, dash="dash"), annotation_text=f"SPOT {spot:.0f}", annotation_font_size=8)
                y_max = min(max(calls["implied_volatility"].max(), puts["implied_volatility"].max()) * 1.3, 1.5)
                fig_sk.update_layout(height=140, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_M, size=7), hovermode="x unified", xaxis=dict(showgrid=False, title=dict(text="Strike", font=dict(size=7)), tickfont=dict(size=6)), yaxis=dict(showgrid=False, title=dict(text="IV", font=dict(size=7)), tickfont=dict(size=6), tickformat=".0%", range=[0, y_max]), margin=dict(l=2, r=2, t=2, b=4), dragmode="zoom", legend=dict(orientation="h", y=1.1, font=dict(size=7)))
                st.plotly_chart(fig_sk, width="stretch", config=CHART_CFG)
    st.markdown('</div>', unsafe_allow_html=True)

    # 11 - PDF | WALLS affiancati
    d1,d2=st.columns(2,gap="small")
    with d1:
        st.markdown(f'<div class="panel"><div class="ptitle"><span>PDF</span><span style="color:{TEXT_M};">Breeden-Litz.</span></div>', unsafe_allow_html=True)
        if chains and len(chains)>1:
            from spx_quant import PDFExtractor, PDFCollector
            pdf_ext=PDFExtractor(); pdfs=[]
            for chain in chains:
                p=pdf_ext.extract(chain,spot)
                if p is not None: pdfs.append(p)
            if pdfs:
                K,dtes,pg=PDFCollector().collect(pdfs,spot)
                if K is not None:
                    mi=len(dtes)//2; pv=pg[mi]; ev=simpson(K*pv,K); cdf=np.cumsum(pv)*(K[1]-K[0])
                    fig=go.Figure()
                    fig.add_trace(go.Scatter(x=K,y=pv,mode="lines",fill="tozeroy",line=dict(color=ACCENT,width=1.5),showlegend=False))
                    fig.add_vline(x=spot,line=dict(color="white",width=1.5),annotation_text=f"SPOT {spot:.0f}",annotation_font_size=7)
                    fig.add_vline(x=ev,line=dict(color=GOLD,width=1,dash="dash"),annotation_text=f"EV {ev:.0f}",annotation_font_size=7)
                    for lv,clr in [(.5,"rgba(34,197,94,0.15)"),(.68,"rgba(234,179,8,0.12)"),(.9,"rgba(161,98,7,0.10)")]:
                        tl=(1-lv)/2; lk=np.interp(tl,cdf,K); uk=np.interp(1-tl,cdf,K)
                        fig.add_trace(go.Scatter(x=[lk,lk,uk,uk],y=[0,pv.max()*.95,pv.max()*.95,0],fill="toself",mode="none",fillcolor=clr,showlegend=False,hoverinfo="skip"))
                    fig.update_layout(height=100,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=7),hovermode="x unified",xaxis=dict(showgrid=False,visible=True,title=dict(text="Strike",font=dict(size=6)),tickfont=dict(size=6)),yaxis=dict(showgrid=False,visible=False),margin=dict(l=2,r=2,t=2,b=4),dragmode="zoom")
                    st.plotly_chart(fig,width="stretch",config=CHART_CFG)
        st.markdown('</div>', unsafe_allow_html=True)
    with d2:
        st.markdown(f'<div class="panel"><div class="ptitle"><span>Walls</span><span style="color:{TEXT_M};">gamma</span></div>', unsafe_allow_html=True)
        # GEXBot major walls se disponibili
        gb_walls = None
        if gexbot_ok:
            try:
                gb_m = st.session_state.gexbot.gex_per_strike(ticker_sel, weight="vol")
                if gb_m.get("major_pos") or gb_m.get("major_neg"):
                    gb_walls = gb_m
            except Exception:
                pass
        if gb_walls:
            gw=gamma_walls(opt,spot) if not opt.empty and "gamma" in opt.columns else None
            c1,c2,c3,c4=st.columns(4)
            c1.metric("PW",f"{gb_walls['major_neg']:.0f}" if gb_walls.get("major_neg") else "—")
            c2.metric("CW",f"{gb_walls['major_pos']:.0f}" if gb_walls.get("major_pos") else "—")
            c3.metric("Pg",f"{gw['put_strength']:,.0f}" if gw and gw.get('put_strength') else "—")
            c4.metric("Cg",f"{gw['call_strength']:,.0f}" if gw and gw.get('call_strength') else "—")
        elif not opt.empty and "gamma" in opt.columns:
            gw=gamma_walls(opt,spot)
            c1,c2,c3,c4=st.columns(4); c1.metric("PW",f"{gw['put_wall']:.0f}"); c2.metric("CW",f"{gw['call_wall']:.0f}"); c3.metric("Pg",f"{gw['put_strength']:,.0f}"); c4.metric("Cg",f"{gw['call_strength']:,.0f}")
            gp=gw.get("gamma_profile",())
            if len(gp)>=4 and len(gp[0])>0:
                fig=go.Figure()
                if len(gp[0])>0: fig.add_trace(go.Scatter(x=gp[0],y=-gp[1],mode="lines",fill="tozeroy",line=dict(color=RED,width=1),fillcolor="rgba(239,68,68,0.15)",showlegend=False,hoverinfo="skip"))
                if len(gp[2])>0: fig.add_trace(go.Scatter(x=gp[2],y=gp[3],mode="lines",fill="tozeroy",line=dict(color=GREEN,width=1),fillcolor="rgba(34,197,94,0.15)",showlegend=False,hoverinfo="skip"))
                fig.add_vline(x=spot,line=dict(color="white",width=.8,dash="dot"))
                fig.update_layout(height=80,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=7),xaxis=dict(range=xax_range,showgrid=False,visible=False),yaxis=dict(showgrid=False,visible=False),margin=dict(l=2,r=2,t=2,b=4),dragmode="zoom")
                st.plotly_chart(fig,width="stretch",config=CHART_CFG)
        else: st.caption("N/D")
        st.markdown('</div>', unsafe_allow_html=True)

    # 12 - VIX TERM STRUCTURE | MMM affiancati
    e1,e2=st.columns(2,gap="small")
    with e1:
        st.markdown(f'<div class="panel"><div class="ptitle"><span>VIX Term</span><span style="color:{TEXT_M};">1D · VIX · 9D · VVIX</span></div>', unsafe_allow_html=True)
        if df is not None and not df.empty:
            vix_cols=[c for c in ["VIX1D","VIX","VIX9D","VVIX"] if c in df.columns]
            if vix_cols:
                last_dates=df.index[-min(120,len(df)):]
                sub=df.loc[last_dates,vix_cols].copy()
                sub.index=sub.index.normalize()
                fig=go.Figure()
                colors={"VIX1D":"#22c55e","VIX":"#f97316","VIX9D":"#ef4444","VVIX":"#a78bfa"}
                for c in vix_cols:
                    fig.add_trace(go.Scatter(x=sub.index,y=sub[c],mode="lines",name=c,line=dict(color=colors.get(c,"#888"),width=1)))
                fig.update_layout(height=100,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=7),margin=dict(l=2,r=2,t=2,b=4),dragmode="zoom",hovermode="x unified",legend=dict(orientation="h",yanchor="bottom",y=1,xanchor="right",x=1,font=dict(size=7)))
                fig.update_xaxes(tickformat="%d %b")
                st.plotly_chart(fig,width="stretch",config=CHART_CFG)
        st.markdown('</div>', unsafe_allow_html=True)
    with e2:
        st.markdown(f'<div class="panel"><div class="ptitle"><span>Exp Move</span><span style="color:{TEXT_M};">MMM</span></div>', unsafe_allow_html=True)
        iv_est=.15
        if not opt.empty and "implied_volatility" in opt.columns and "strike" in opt.columns:
            atm_k = opt.iloc[(opt["strike"] - spot).abs().argsort()[:1]]
            if not atm_k.empty:
                atm_ivs = pd.to_numeric(atm_k["implied_volatility"], errors="coerce").dropna()
                atm_ivs = atm_ivs[(atm_ivs >= 0.05) & (atm_ivs <= 2.0)]
                if not atm_ivs.empty: iv_est = float(atm_ivs.mean())
        em=expected_move(spot,iv_est,max(dte_sel,1))
        c1,c2=st.columns(2); c1.metric("Move",f"${em['expected_move']:.0f}"); c2.metric("+/-%",f"{em['expected_move_pct']:.1f}%")
        st.markdown(f'<div style="font-size:9px;color:{TEXT_M};text-align:center;">{em["lower"]:.0f} - {em["upper"]:.0f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 13a - STATISTICAL LEARNING
    st.markdown(f'<div class="panel"><div class="ptitle"><span>Statistical Learning</span><span style="color:{TEXT_M};">Box‑Plot · Z · σ</span></div>', unsafe_allow_html=True)
    # Legge quantili B&L se gia' calcolati (da PL section piu' avanti)
    _bl_qs = {}
    _bl_trunc = {}
    if chains and spot:
        _bl_raw = bl_quantile_levels(chains, spot, target_dte=min(dte_use, 45))
        if _bl_raw:
            _bl_qs = _bl_raw.get("quantiles", {})
            _bl_trunc = _bl_raw.get("truncated", {})
        if _bl_trunc.get("right") and _bl_qs:
            for _p in [90, 95, 99]: _bl_qs.pop(_p, None)
    bl_q50 = _bl_qs.get(50, None) if _bl_qs else None
    bl_q25 = _bl_qs.get(25, None) if _bl_qs else None
    bl_q75 = _bl_qs.get(75, None) if _bl_qs else None
    # BKM implied risk-neutral moments (dalla catena opzioni)
    bkm = None
    if not combined.empty and "option_type" in combined.columns:
        bkm = implied_risk_moments(combined, spot)
        if bkm and bkm.get("variance", 0) > 0:
            sk = bkm["skewness"]
            ku = bkm["kurtosis"]
            ek = ku - 3
            sk_sym = "↗" if sk > 0.1 else "↘" if sk < -0.1 else "≈"
            ek_sym = "⚠" if ek > 0.5 else "≈"
            st.markdown(f'<div style="font-size:9px;color:{TEXT_M};display:flex;justify-content:space-between;margin-top:4px;border-top:1px solid {CARD_B};padding-top:4px;">'
                        f'<span>BKM σ: {bkm["volatility"]:.1%}</span>'
                        f'<span>Skew: {sk_sym}{abs(sk):.2f}</span>'
                        f'<span>Kurt(Exc): {ek_sym}{abs(ek):.2f}</span>'
                        f'<span>n={bkm["n_calls"]+bkm["n_puts"]}</span></div>',
                        unsafe_allow_html=True)
    if price_data is not None and len(price_data) > 20:
        bp = box_plot_stats(price_data)
        zs = z_score_analysis(price_data[-1], price_data)
        if bp:
            # Prima riga: Q50 da opzioni (se disponibile) o storico
            ref_q50 = bl_q50 if bl_q50 else bp["q50"]
            q50_delta = f"{(ref_q50/price_data[-1]-1)*100:+.1f}%" if price_data[-1] != 0 else "—"
            q50_label = f"Q50 (B&L)" if bl_q50 else "Q50 (Mediana)"
            c1, c2, c3 = st.columns(3)
            c1.metric(q50_label, f"{ref_q50:.0f}",
                      delta=q50_delta, delta_color="normal",
                      help=f"B&L Q50={bl_q50:.0f}" if bl_q50 and bl_q50 != bp["q50"] else None)
            # IQR: prioritario da opzioni
            if bl_q25 and bl_q75:
                iqr_label = f"{bl_q25:.0f}–{bl_q75:.0f}"
                iqr_delta = f"{bl_q75-bl_q25:.0f}pts"
                iqr_help = f"B&L IQR (risk-neutral) | Storico: {bp['q25']:.0f}–{bp['q75']:.0f} ({bp['iqr']:.0f}pts)"
            else:
                iqr_label = f"{bp['q25']:.0f}–{bp['q75']:.0f}"
                iqr_delta = f"{bp['iqr']:.0f}pts"
                iqr_help = f"IQR storico | B&L non disponibile"
            c2.metric("IQR", iqr_label, delta=iqr_delta, help=iqr_help, delta_color="off")
            c3.metric("Distorsione", f"{'↗' if bp['skewness']>0 else '↘'}<b>{abs(bp['skewness']):.2f}</b>" if abs(bp['skewness']) > 0.1 else "≈0",
                      help=f"Skewness storica {bp['skewness']:+.3f} | Kurtosis {bp['kurtosis']:+.2f}")
            # Seconda riga: Z-Score storico + Z-Score implicito (opzionale)
            # Soglia dinamica: se excess_kurtosis > 0, 2σ non e' piu' 95%
            zs_thresh = zs.get("cf_threshold_2s", 2.0) if zs else 2.0
            cz = RED if zs and abs(zs["z_score"]) > zs_thresh else GOLD if zs and abs(zs["z_score"]) > 1 else GREEN
            c1, c2, c3, c4 = st.columns(4)
            if zs:
                z_label = f"Z‑Score"
                z_help = f"Cornish‑Fisher adj: {zs.get('cf_threshold_2s', 2.0):.1f}σ soglia (kurt={zs.get('excess_kurtosis', 0):.1f})" if zs.get("excess_kurtosis", 0) > 0.5 else None
                c1.metric(z_label, f"{zs['z_score']:+.2f}s",
                          delta=zs['category'].upper(), delta_color="off",
                          help=z_help)
                # Z-score implicito (prezzo vs B&L Q50 / BKM σ)
                bkm_z = None
                if bl_q50 and bkm and bkm.get("volatility", 0) > 0.001:
                    bkm_sigma_daily = bkm["volatility"] / np.sqrt(252)
                    bkm_z_val = (price_data[-1] - ref_q50) / (bkm_sigma_daily * price_data[-1]) if bkm_sigma_daily * price_data[-1] > 0 else None
                    if bkm_z_val is not None:
                        bkm_z = bkm_z_val
                c2.metric("Stop 1σ", f"{zs['stop_1s']:.0f}",
                          help=f"Long SL @ {zs['stop_1s']:.0f} ({1}σ da media storica {zs['mean']:.0f})",
                          delta_color="off")
                c3.metric("Stop 2σ", f"{zs['stop_2s']:.0f}",
                          help=f"SL rinforzo @ {zs['stop_2s']:.0f} ({2}σ)",
                          delta_color="off")
                c4.metric(f"σ ({zs['std']:.1f})", f"{zs['mean']:.0f}",
                          help=f"Media storica {zs['mean']:.0f} (89d) | σ {zs['std']:.1f} | Var {zs['std']**2:.0f}")
                # Adapted VaR: prob. di toccare -1σ in sessione (solo intraday)
                if dte_use > 0 and dte_use <= 5 and zs.get("std", 0) > 0:
                    # Sigma annualizzato da rendimenti giornalieri (non da livelli prezzo)
                    _av_rets = np.diff(np.log(price_data))
                    _av_sigma = np.std(_av_rets, ddof=1) * np.sqrt(252)
                    _av = adapted_var(price_data[-1], zs["stop_1s"],
                                      _av_sigma, dte_use, drift=0.0)
                    if _av:
                        tp_col = GOLD if _av["touch_prob"] > 0.05 else TEXT_M
                        st.markdown(f'<div style="font-size:9px;color:{tp_col};text-align:center;margin-top:2px;">'
                                    f'touch 1σ: {_av["touch_prob"]:.1%} | dist: {_av["distance_pct"]:.1f}%</div>',
                                    unsafe_allow_html=True)
            # λ risk premia decomposition (Vázquez / Bergomi-Guyon)
            if bkm and bkm.get("variance", 0) > 0 and bp:
                lam = decompose_risk_premia(
                    bkm,
                    hist_skew=bp.get("skewness", 0),
                    hist_kurt=bp.get("kurtosis", 0) + 3,
                )
                if lam:
                    lc = GOLD if abs(lam.get("lam2", 0)) > 0.3 or abs(lam.get("lam3", 0)) > 0.1 else TEXT_M
                    st.markdown(f'<div style="font-size:9px;color:{lc};margin-top:4px;border-top:1px solid {CARD_B};padding-top:3px;text-align:center;">'
                                f'λ₂={lam["lam2"]:+.2f}  λ₃={lam["lam3"]:+.2f}  λ₄={lam["lam4"]:+.2f}  |  {lam["summary"]}</div>',
                                unsafe_allow_html=True)
        else:
            st.caption("Dati insufficienti per box‑plot")
    else:
        st.caption("Prezzi insufficienti (< 20 barre)")
    st.markdown('</div>', unsafe_allow_html=True)

    # 13 - ANOMALIES
    st.markdown(f'<div class="panel"><div class="ptitle"><span>Anomalies</span><span style="color:{TEXT_M};">z-score</span></div>', unsafe_allow_html=True)
    if price_data is not None and len(price_data)>30:
        anom=intraday_anomalies(pd.Series(price_data),window=20,z_threshold=2.0); z=anom["z_scores"].get("price_z",0)
        cz=RED if abs(z)>2 else GOLD if abs(z)>1 else GREEN
        st.markdown(f'<div style="text-align:center;font-size:18px;font-weight:700;color:{cz};">{z:+.2f}s</div>', unsafe_allow_html=True)
        if anom["anomalies"]:
            for a in anom["anomalies"]: st.markdown(f'<div style="font-size:8px;color:{GOLD};padding:1px 0;">{a["description"]}</div>', unsafe_allow_html=True)
        else: st.caption("Nessuna anomalia")
        # Expected Shortfall dalla densita' B&L
        try:
            _bl_es_k = np.array(_bl_raw["K"])
            _bl_es_p = np.array(_bl_raw["pdf"])
            _bl_es_c = np.array(_bl_raw.get("cdf", []))
            if len(_bl_es_c) != len(_bl_es_k):
                from scipy.integrate import simpson as _es_simp
                _bl_es_c = np.zeros_like(_bl_es_k)
                for _i in range(1, len(_bl_es_k)):
                    _bl_es_c[_i] = _es_simp(_bl_es_p[:_i+1], _bl_es_k[:_i+1])
                _bl_es_c = np.clip(_bl_es_c, 0, 1)
            _es_res = calculate_expected_shortfall(_bl_es_p, _bl_es_c, _bl_es_k, alpha=0.05)
            if _es_res:
                _es_clr = RED if _es_res["ES"] < price_data[-1] * 0.95 else GOLD
                st.markdown(f'<div style="font-size:9px;color:{_es_clr};text-align:center;border-top:1px solid {CARD_B};padding-top:4px;margin-top:4px;">'
                            f'VaR₅%: ${_es_res["VaR"]:.0f} | ES₅%: ${_es_res["ES"]:.0f}'
                            f'{" ⚠ tail risk" if _es_res["ES"] < price_data[-1] * 0.90 else ""}</div>',
                            unsafe_allow_html=True)
        except (NameError, KeyError, TypeError, Exception):
            pass
    st.markdown('</div>', unsafe_allow_html=True)

    # 13b - CONFLUENZA (magnete strutturale)
    st.markdown(f'<div class="panel"><div class="ptitle"><span>Confluenza</span><span style="color:{TEXT_M};">σ · B&L · IV · KO</span></div>', unsafe_allow_html=True)
    if spot and price_data is not None and len(price_data) > 20:
        zs_conf = z_score_analysis(price_data[-1], price_data,
                                    excess_kurtosis=abs(bkm.get("kurtosis", 3) - 3) if bkm else None)
        _bl_qs_conf = None
        if bl_q50:
            _bl_qs_conf = {50: bl_q50}
            if bl_q25: _bl_qs_conf[25] = bl_q25
            if bl_q75: _bl_qs_conf[75] = bl_q75
        _iv_conf = iv_skew_extremes(combined, spot) if not combined.empty else None
        _barriers_conf = knockout_barriers(spot, iv_est, dte_sel) if iv_est > 0 else None
        confluences = structural_confluence(
            spot,
            sigma_bands=zs_conf if zs_conf else None,
            bl_quantiles={"quantiles": _bl_qs_conf} if _bl_qs_conf else None,
            iv_extremes=_iv_conf,
            ko_barriers=_barriers_conf,
            tolerance_bp=30.0,
        )
        if confluences:
            for c in confluences[:3]:
                c_clr = GOLD if c["n_famiglie"] == 2 else RED
                c_icon = "🧲" if c["n_famiglie"] >= 3 else "⚡"
                labels = ", ".join([l["label"] for l in c["livelli"]])
                fams = ", ".join(c["famiglie"])
                st.markdown(
                    f'<div style="font-size:9px;color:{c_clr};display:flex;justify-content:space-between;'
                    f'border-left:2px solid {c_clr};padding-left:4px;margin:2px 0;">'
                    f'<span>{c_icon} {c["strike"]:.0f}</span>'
                    f'<span>{labels}</span>'
                    f'<span style="color:{TEXT_M};">{fams}</span>'
                    f'<span>{c["n_famiglie"]}fam</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption(f"Nessuna confluenza entro 30bp")
    else:
        st.caption("Dati insufficienti")
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# 3D REGIME SPACE + VALIDAZIONE (full width sotto la griglia)
# ═══════════════════════════════════════════════════════════════════════
st.divider()
col_a,col_b=st.columns(2)

with col_a:
    st.markdown(f'<div class="panel"><div class="ptitle"><span>🎯 3D Regime Space</span><span style="color:{TEXT_M};">SPX·VIX·DXY</span></div>', unsafe_allow_html=True)
    if df is not None and not df.empty and len(df)>100:
        det=RegimeDetector(); rh=det.detect_history(df)
        if rh is not None and not rh.empty:
            pd3=df[["SPX","VIX","DXY"]].loc[rh.index].copy()
            pd3["Regime"]=rh; regs=pd3["Regime"].unique()
            cm3d={"EXPANSION":"green","SLOWDOWN":"blue","REFLATION":"#f97316","STAGFLATION":"red","TRANSIZIONE":"gray"}
            fig3=go.Figure()
            for rn in regs:
                sub=pd3[pd3["Regime"]==rn]
                if len(sub)>500: sub=sub.sample(500,random_state=42)
                fig3.add_trace(go.Scatter3d(x=sub["SPX"],y=sub["VIX"],z=sub["DXY"],mode="markers",name=rn,marker=dict(size=3,color=cm3d.get(rn,"gray"),opacity=.6)))
            last=pd3.iloc[-1]
            fig3.add_trace(go.Scatter3d(x=[last["SPX"]],y=[last["VIX"]],z=[last["DXY"]],mode="markers",name=f"CORRENTE ({last['Regime']})",marker=dict(size=12,color="yellow",symbol="diamond",line=dict(width=2,color="black"))))
            fig3.update_layout(height=380,scene=dict(xaxis_title="SPX",yaxis_title="VIX",zaxis_title="DXY",bgcolor="rgba(0,0,0,0)"),margin=dict(l=0,r=0,t=4,b=0),paper_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=8))
            st.plotly_chart(fig3,width="stretch",config={"displayModeBar":False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown(f'<div class="panel"><div class="ptitle"><span>Bonferroni</span><span style="color:{TEXT_M};">validazione</span></div>', unsafe_allow_html=True)
    validator=StatisticalValidator()
    if df is not None and not df.empty:
        bonf=validator.bonferroni_correction(df) if hasattr(validator,'bonferroni_correction') else {}
        results=bonf.get("results",[])
        if results:
            bf=pd.DataFrame(results)
            has_int=bool(bf.get("interpretation",[""]).iloc[0]) if "interpretation" in bf.columns else False
            cols_show=["factor","correlation","p-value","significant"]
            bf=bf.rename(columns={"factor":"Fattore","correlation":"Corr","p_value":"p-value","significant":"Signif","interpretation":"Int."})
            if has_int: cols_show.append("Int.")
            bf["Signif"]=bf["Signif"].apply(lambda x:"[S]" if x else "[NS]")
            st.dataframe(bf[[c for c in cols_show if c in bf.columns]],width="stretch",hide_index=True,height=100)
        st.caption(f"Alpha: {bonf.get('corrected_alpha','N/D')} | Test: {bonf.get('n_tests',0)}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="panel"><div class="ptitle"><span>🏋️ Walk-Forward</span><span style="color:{TEXT_M};">stabilità</span></div>', unsafe_allow_html=True)
    if df is not None and not df.empty:
        wf=validator.walk_forward_validation(df) if hasattr(validator,'walk_forward_validation') else {}
        stab=wf.get("stability_score",0)
        st.markdown(f"**Stabilità:** {stab:.1f}%")
        if wf.get("stable_factors"): st.markdown(f"✅ STABILI: {', '.join(wf['stable_factors'])}")
        if wf.get("unstable_factors"): st.markdown(f"⚠️ INSTABILI: {', '.join(wf['unstable_factors'])}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Shift predictions
    st.markdown(f'<div class="panel"><div class="ptitle"><span>🚨 Shift Prediction</span><span style="color:{TEXT_M};">segnali</span></div>', unsafe_allow_html=True)
    ss=shift.get("signals",[])
    if ss:
        for s in ss: st.warning(s)
    else: st.info("Nessun segnale di shift")
    if shift.get("direction"): st.markdown(f"**Direzione:** {shift['direction']}")
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PROBABILISTIC LEVELS (full-width — B&L, Barrier, IV extremes, ETF)
# ═══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(f'<div style="color:{ACCENT};font-size:13px;font-weight:700;margin:4px 0;">⬡ Probabilistic Levels</div>', unsafe_allow_html=True)

# Compute levels
iv_est_pl = 0.15
if not opt.empty and "implied_volatility" in opt.columns:
    atm_k = opt.iloc[(opt["strike"] - spot).abs().argsort()[:1]]
    if not atm_k.empty:
        atm_ivs = pd.to_numeric(atm_k["implied_volatility"], errors="coerce").dropna()
        atm_ivs = atm_ivs[(atm_ivs >= 0.05) & (atm_ivs <= 2.0)]
        if not atm_ivs.empty:
            iv_est_pl = float(atm_ivs.mean())

prob = compute_probabilistic_levels(chains, spot, iv_est_pl, dte_use)
bl_quantiles = prob.get("bl_quantiles", {})
bl_qs = bl_quantiles.get("quantiles", {}) if bl_quantiles else {}
# Filtra code troncate: se Q90 ≈ Q95 ≈ Q99, escludi i percentili estremi
bl_trunc = bl_quantiles.get("truncated", {})
if bl_trunc.get("right") and bl_qs:
    for _p in [90, 95, 99]:
        bl_qs.pop(_p, None)
if bl_trunc.get("left") and bl_qs:
    bl_qs.pop(1, None)

# IV extremes
iv_extremes = iv_skew_extremes(combined if not combined.empty else opt, spot)

# ETF rebalancing
spy_ret = None
if price_data is not None and len(price_data) > 1:
    spy_ret = (price_data[-1] / price_data[-2]) - 1
dex_total_pl = None
if not opt.empty and "delta" in opt.columns:
    dex_total_pl = compute_dex(opt, spot).get("total_dex", 0)
etf_rebal = etf_rebalancing_levels(spot, spy_ret, dex_total_pl)

# Barriers
barriers = prob.get("barriers", [])

# ── Combined levels chart ──
fig_pl = go.Figure()
y_base = 0

# B&L quantiles
bl_colors = {1: TEXT_M, 5: ORANGE, 10: GOLD, 25: GREEN, 50: ACCENT,
             75: GREEN, 90: GOLD, 95: ORANGE, 99: RED}
bl_labels = {1: "Q01", 5: "Q05", 10: "Q10", 25: "Q25", 50: "Q50",
             75: "Q75", 90: "Q90", 95: "Q95", 99: "Q99"}
bl_markers = {1: "x", 5: "triangle-down", 10: "triangle-down", 25: "diamond",
              50: "star", 75: "diamond", 90: "triangle-up", 95: "triangle-up", 99: "x"}

if bl_qs:
    for pctl in sorted(bl_qs.keys()):
        k = bl_qs[pctl]
        fig_pl.add_trace(go.Scatter(
            x=[k], y=[y_base],
            mode="markers+text",
            marker=dict(size=10 if pctl == 50 else 7,
                        color=bl_colors.get(pctl, TEXT_M),
                        symbol=bl_markers.get(pctl, "circle"),
                        line=dict(width=1, color="white")),
            text=[f"{bl_labels.get(pctl, str(pctl))}"],
            textposition="top center",
            textfont=dict(size=8, color=bl_colors.get(pctl, TEXT_M)),
            name=f"B&L {bl_labels.get(pctl, str(pctl))}",
            hovertext=f"B&L {bl_labels.get(pctl, str(pctl))}: {k:.1f}",
            hoverinfo="text",
            showlegend=False,
        ))

# Barriers (KO levels)
for b in barriers:
    clr = RED if b["direction"] == "down" else GREEN
    sym = "triangle-down" if b["direction"] == "down" else "triangle-up"
    fig_pl.add_trace(go.Scatter(
        x=[b["strike"]], y=[y_base],
        mode="markers+text",
        marker=dict(size=6, color=clr, symbol=sym,
                    line=dict(width=0.5, color="white")),
        text=[f"{b['sigma_distance']:.1f}σ"],
        textposition="bottom center",
        textfont=dict(size=7, color=clr),
        name=f"KO {b['ratio']:.1%}",
        hovertext=f"KO {b['ratio']:.1%} b={b['sigma_distance']:.2f}σ touch={b['touch_prob']:.1%}",
        hoverinfo="text",
        showlegend=False,
    ))

# IV extremes
if iv_extremes.get("basic_high"):
    fig_pl.add_trace(go.Scatter(
        x=[iv_extremes["basic_high"]], y=[y_base],
        mode="markers+text",
        marker=dict(size=8, color="#a78bfa", symbol="diamond",
                    line=dict(width=0.5, color="white")),
        text=["IV↑"],
        textposition="top center",
        textfont=dict(size=7, color="#a78bfa"),
        name="IV max strike",
        hovertext=f"IVmax @ {iv_extremes['basic_high']:.0f} ({iv_extremes.get('basic_high_iv', 0):.1%})",
        hoverinfo="text",
        showlegend=False,
    ))
if iv_extremes.get("basic_low"):
    fig_pl.add_trace(go.Scatter(
        x=[iv_extremes["basic_low"]], y=[y_base],
        mode="markers+text",
        marker=dict(size=8, color="#a78bfa", symbol="diamond",
                    line=dict(width=0.5, color="white")),
        text=["IV↓"],
        textposition="bottom center",
        textfont=dict(size=7, color="#a78bfa"),
        name="IV min strike",
        hovertext=f"IVmin @ {iv_extremes['basic_low']:.0f} ({iv_extremes.get('basic_low_iv', 0):.1%})",
        hoverinfo="text",
        showlegend=False,
    ))

# Spot reference
fig_pl.add_vline(x=spot, line=dict(color="white", width=1.5, dash="dash"),
                 annotation_text=f"Spot {spot:.0f}", annotation_font_size=8)

x_min = spot * 0.88
x_max = spot * 1.12
fig_pl.update_layout(
    height=90,
    xaxis=dict(range=[x_min, x_max], showgrid=True, gridcolor="rgba(255,255,255,0.05)",
               tickfont=dict(size=7, color=TEXT_M), title=None),
    yaxis=dict(visible=False, range=[-0.5, 0.5]),
    margin=dict(l=4, r=4, t=2, b=6),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    hovermode="closest", dragmode="zoom",
)
st.plotly_chart(fig_pl, width="stretch", config=CHART_CFG)

# ── Summary metrics ──
mcols = st.columns(6, gap="small")
with mcols[0]:
    if bl_qs:
        p5 = bl_qs.get(5, spot)
        p95 = bl_qs.get(95, bl_qs.get(75, spot))
        st.metric("B&L Range", f"{p5:.0f}-{p95:.0f}",
                  delta=f"{(p95-p5)/spot*100:.1f}%", delta_color="off")
    else: st.metric("B&L Range", "N/D")
with mcols[1]:
    if barriers:
        down_b = [b for b in barriers if b["direction"] == "down"]
        up_b = [b for b in barriers if b["direction"] == "up"]
        max_down = down_b[-1]["strike"] if down_b else spot
        min_up = up_b[0]["strike"] if up_b else spot
        st.metric("KO Range", f"{max_down:.0f}-{min_up:.0f}",
                  delta=f"{(min_up-max_down)/spot*100:.1f}%", delta_color="off")
    else: st.metric("KO Range", "N/D")
with mcols[2]:
    iv_flow = etf_rebal.get("total_flow_bn", 0)
    iv_dir = etf_rebal.get("net_direction", "neutral")
    st.metric("ETF Leva Flow", f"${iv_flow:+.3f}B",
              delta=iv_dir.upper(), delta_color="off")
with mcols[3]:
    if bl_qs:
        q50 = bl_qs.get(50, spot)
        st.metric("B&L Q50", f"{q50:.0f}",
                  delta=f"{(q50/spot-1)*100:+.1f}%",
                  delta_color="normal")
    else: st.metric("B&L Q50", "N/D")
with mcols[4]:
    if iv_extremes.get("interp_high"):
        st.metric("IVmax@", f"{iv_extremes['interp_high']:.0f}",
                  delta=f"↑{iv_extremes.get('interp_high_iv', 0):.1%}",
                  help="Strike con IV massima (tipicamente lato put/OTM)",
                  delta_color="off")
    elif iv_extremes.get("basic_high"):
        st.metric("IVmax@", f"{iv_extremes['basic_high']:.0f}",
                  delta=f"↑{iv_extremes.get('basic_high_iv', 0):.1%}",
                  help="Strike con IV massima (±5%)",
                  delta_color="off")
    else: st.metric("IVmax@", "N/D")
with mcols[5]:
    if iv_extremes.get("interp_low"):
        st.metric("IVmin@", f"{iv_extremes['interp_low']:.0f}",
                  delta=f"↓{iv_extremes.get('interp_low_iv', 0):.1%}",
                  help="Strike con IV minima (tipicamente lato call)",
                  delta_color="off")
    elif iv_extremes.get("basic_low"):
        st.metric("IVmin@", f"{iv_extremes['basic_low']:.0f}",
                  delta=f"↓{iv_extremes.get('basic_low_iv', 0):.1%}",
                  help="Strike con IV minima (±5%)",
                  delta_color="off")
    else: st.metric("IVmin@", "N/D")

# ── Detail expander ──
with st.expander("📋 Dettaglio livelli", expanded=False):
    pl_rows = []

    # B&L quantiles
    if bl_qs:
        bl_desc = "Breeden-Litzenberger risk-neutral density"
        if bl_trunc.get("right"):
            bl_desc += " [coda dx troncata]"
        if bl_trunc.get("left"):
            bl_desc += " [coda sx troncata]"
        bl_row = {"Famiglia": "B&L Risk-Neutral", "Tipo": "Quantile",
                  "Descrizione": bl_desc,
                  "Livelli": " | ".join(f"{bl_labels.get(p)}={bl_qs[p]:.0f}" for p in sorted(bl_qs.keys())),
                  "Fonte": f"{bl_quantiles.get('dte', '?')}d PDF"}
        pl_rows.append(bl_row)

    # KO barriers
    for b in barriers:
        d = "⬇" if b["direction"] == "down" else "⬆"
        pl_rows.append({
            "Famiglia": "Barrier KO",
            "Tipo": f"Knock-Out {d}",
            "Descrizione": f"b={b['sigma_distance']:.2f}σ touch={b['touch_prob']:.1%}",
            "Livelli": f"{b['strike']:.0f} ({b['ratio']:.1%})",
            "Fonte": f"BS {b['sigma_distance']:.2f}σ",
        })

        # IV extremes (naming: IVmax@ = strike con IV piu' alta, di solito put OTM)
    if iv_extremes.get("basic_high"):
        pl_rows.append({
            "Famiglia": "IV Skew",
            "Tipo": "IVmax ±5%",
            "Descrizione": f"Strike con IV max {iv_extremes.get('basic_high_iv', 0):.1%} (tipicamente put)",
            "Livelli": f"{iv_extremes['basic_high']:.0f}",
            "Fonte": "IV skew ATM ±5%",
        })
    if iv_extremes.get("basic_low"):
        pl_rows.append({
            "Famiglia": "IV Skew",
            "Tipo": "IVmin ±5%",
            "Descrizione": f"Strike con IV min {iv_extremes.get('basic_low_iv', 0):.1%} (tipicamente call)",
            "Livelli": f"{iv_extremes['basic_low']:.0f}",
            "Fonte": "IV skew ATM ±5%",
        })
    if iv_extremes.get("interp_high"):
        pl_rows.append({
            "Famiglia": "IV Skew",
            "Tipo": "IVmax spline",


            "Descrizione": f"IV max spline: {iv_extremes.get('interp_high_iv', 0):.1%}",
            "Livelli": f"{iv_extremes['interp_high']:.0f}",
            "Fonte": "CubicSpline IV",
        })
    if iv_extremes.get("interp_low"):
        pl_rows.append({
            "Famiglia": "IV Skew",
            "Tipo": "Interpolated Low",
            "Descrizione": f"IV min spline: {iv_extremes.get('interp_low_iv', 0):.1%}",
            "Livelli": f"{iv_extremes['interp_low']:.0f}",
            "Fonte": "CubicSpline IV",
        })

    if etf_rebal.get("total_flow_bn", 0) != 0:
        pl_rows.append({
            "Famiglia": "Mechanical",
            "Tipo": "ETF Leva Rebalancing",
            "Descrizione": f"Net flow: ${etf_rebal['total_flow_bn']:+.3f}B",
            "Livelli": f"{'BUY' if etf_rebal['total_flow_bn'] > 0 else 'SELL'} ${abs(etf_rebal['total_flow_bn']):.2f}B",
            "Fonte": f"SPY ret {spy_ret*100:.2f}%" if spy_ret else "—",
        })

    if pl_rows:
        df_pl = pd.DataFrame(pl_rows)
        st.dataframe(df_pl, width="stretch", hide_index=True, height=min(40 + 35 * len(pl_rows), 300))
    else:
        st.caption("Dati insufficienti per livelli probabilistici.")


# ═══════════════════════════════════════════════════════════════════════
# STORICO GIP
# ═══════════════════════════════════════════════════════════════════════
with st.expander("📈 Storico GIP + γ-ENV (ultimi 30 giorni)", expanded=False):
    hist_df=db.history(days=30)
    if not hist_df.empty:
        hist_df["timestamp"]=pd.to_datetime(hist_df["timestamp"])
        fig_h=go.Figure()
        fig_h.add_trace(go.Scatter(x=hist_df["timestamp"],y=hist_df["g_score"],mode="lines+markers",name="G-Score",line=dict(color="#00cc66")))
        fig_h.add_trace(go.Scatter(x=hist_df["timestamp"],y=hist_df["i_score"],mode="lines+markers",name="I-Score",line=dict(color="#ff8800")))
        fig_h.add_trace(go.Scatter(x=hist_df["timestamp"],y=hist_df["shift_prob"],mode="lines+markers",name="Shift",line=dict(color="#ff4444",dash="dot"),yaxis="y2"))
        if "gex_env" in hist_df.columns:
            env_num = hist_df["gex_env"].map({"POS": 1, "NEG": -1}).fillna(0)
            fig_h.add_trace(go.Scatter(x=hist_df["timestamp"],y=env_num,mode="markers",name="γ-ENV",marker=dict(size=4,color=["#22c55e" if v==1 else "#ef4444" for v in env_num]),yaxis="y2"))
        if "vvr" in hist_df.columns:
            fig_h.add_trace(go.Scatter(x=hist_df["timestamp"],y=hist_df["vvr"],mode="lines",name="VVIX/VIX",line=dict(color="#a78bfa",width=1,dash="dot"),yaxis="y2"))
        fig_h.update_layout(height=250,yaxis=dict(title="G/I",range=[-1.2,1.2]),yaxis2=dict(title="Shift / γ-ENV / VVR",overlaying="y",side="right",range=[-1.5,1.5]),hovermode="x unified",margin=dict(l=10,r=10,t=10,b=30),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_M,size=10))
        st.plotly_chart(fig_h,width="stretch",config={"displayModeBar":False})
        cols_show = ["timestamp","regime","g_score","i_score","bias","gex_env","vvr","vix","wti","copper","dxy"]
        cols_show = [c for c in cols_show if c in hist_df.columns]
        st.dataframe(hist_df[cols_show].tail(10).style.format({"g_score":"{:+.3f}","i_score":"{:+.3f}","vix":"{:.1f}","wti":"${:.1f}","copper":"${:.2f}","dxy":"{:.1f}"}),width="stretch",hide_index=True)
        st.markdown("**γ-ENV Flip History** (VVR antecedente il flip)")
        try:
            with sqlite3.connect(str(DB_PATH)) as c:
                flip_df=pd.read_sql("""
                    SELECT timestamp,regime,prev_env,gex_env AS new_env,gex_total,vvr_prev,vvr
                    FROM (
                        SELECT *,LAG(gex_env) OVER (ORDER BY timestamp) AS prev_env,
                                LAG(vvr) OVER (ORDER BY timestamp) AS vvr_prev
                        FROM macro_history
                    ) WHERE gex_env IS NOT NULL AND prev_env IS NOT NULL
                      AND gex_env != prev_env
                    ORDER BY timestamp DESC LIMIT 20
                """,c)
            if not flip_df.empty:
                st.dataframe(flip_df,width="stretch",hide_index=True)
            else: st.caption("Nessun flip registrato.")
        except: st.caption("Query non disponibile.")
    else: st.info("Ancora nessuno storico.")

# ═══════════════════════════════════════════════════════════════════════
# EOD SNAPSHOT (salva ultimo stato del giorno per il morning briefing)
# ═══════════════════════════════════════════════════════════════════════
try:
    _cw = None
    _pw = None
    if gb_walls:
        _cw = gb_walls.get("major_pos")
        _pw = gb_walls.get("major_neg")
    elif 'gw' in dir() and gw:
        _cw = gw.get("call_wall") or gw.get("major_pos")
        _pw = gw.get("put_wall") or gw.get("major_neg")
    _skew = signals.get("skew_level") if isinstance(signals, dict) else None
    _em_up = em.get("upper") if 'em' in dir() and em else None
    _em_down = em.get("lower") if 'em' in dir() and em else None
    _g = gip.get("G", 0) if isinstance(gip, dict) else 0
    _i = gip.get("I", 0) if isinstance(gip, dict) else 0
    with sqlite3.connect(str(DB_PATH)) as _c:
        save_eod_snapshot(_c, datetime.now().strftime("%Y-%m-%d"), ticker_sel,
                          spot, gex_env, gex_zg, gex_total, vvr_log,
                          _cw, _pw, _skew, _em_up, _em_down,
                          r_str, _g, _i, shift_p, vx)
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════════════
# AUTO-REFRESH
# ═══════════════════════════════════════════════════════════════════════
if auto:
    time.sleep(refresh_sec)
    st.cache_data.clear()
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════
st.session_state["_init_done"] = True
st.divider()
st.caption(f"⬡ Maze Capital Terminal v5.0 · {datetime.now():%Y-%m-%d %H:%M:%S} · {ticker_sel} · refresh: {'auto ogni ' + str(refresh_sec) + 's' if auto else 'manuale ⟳'} · DB: {DB_PATH} · Dati: ⚡GEXBot (GEX real-time) + yfinance (Greeks/Chain/PDF)")
