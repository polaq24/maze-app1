"""
Macro Quant Dashboard — Streamlit
====================================
Interfaccia web interattiva per il Macro Breakdown GIP.
Include: SQLite logging, audio alerts, esportazione PDF, calendario economico.
"""

import sys, os, sqlite3, time, io, csv
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from macro_quant import (
    Config, DataFetcher, CorrelationAnalyzer,
    RegimeDetector, RegimeShiftPredictor, StatisticalValidator,
    MacroBreakdownReport, Visualizer, MacroQuantSystem,
    Regime, Bias,
)

st.set_page_config(
    page_title="Macro Quant Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CONFIG
# =============================================================================

DASH_DIR = Path(__file__).parent
DB_PATH = DASH_DIR / "macro_history.db"

# =============================================================================
# SQLITE — Storico giornaliero GIP
# =============================================================================

class MacroHistoryDB:
    """Registra ogni scan in SQLite per tracciare evoluzione GIP."""

    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    regime TEXT,
                    g_score REAL,
                    i_score REAL,
                    confidence REAL,
                    bias TEXT,
                    shift_prob REAL,
                    vix REAL,
                    wti REAL,
                    copper REAL,
                    dxy REAL,
                    skew REAL,
                    vix_term TEXT,
                    UNIQUE(timestamp)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    type TEXT,
                    message TEXT,
                    acknowledged INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def log_scan(self, regime: Dict, bias: Bias, shift: Dict, signals: Dict):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gip = regime.get("gip", {})
        with self._conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO macro_history
                (timestamp, regime, g_score, i_score, confidence, bias,
                 shift_prob, vix, wti, copper, dxy, skew, vix_term)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                ts, regime["regime"].value,
                round(gip.get("G", 0), 4),
                round(gip.get("I", 0), 4),
                round(regime.get("confidence", 0), 2),
                bias.value,
                round(shift.get("shift_probability", 0), 3),
                round(signals.get("vix_level", 0), 1),
                round(signals.get("wti_level", 0), 1),
                round(signals.get("copper_level", 0), 2),
                round(signals.get("dxy_level", 0), 1),
                round(signals.get("skew_level", 0), 0),
                signals.get("vix_term", ""),
            ))
            conn.commit()

    def history(self, days: int = 30) -> pd.DataFrame:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        with self._conn() as conn:
            df = pd.read_sql(f"""
                SELECT * FROM macro_history
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            """, conn, params=(cutoff,))
        return df

    def log_alert(self, alert_type: str, message: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO alerts (timestamp, type, message) VALUES (?,?,?)",
                (ts, alert_type, message),
            )
            conn.commit()


# =============================================================================
# AUDIO ALERTS
# =============================================================================

class AudioAlerter:
    """Avvisi sonori su Windows via winsound."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._last_alert = None

    def alert_shift(self, prob: float):
        if not self.enabled or prob < 0.5:
            return
        # Evita alert ripetitivi (max 1 ogni 30 min)
        if self._last_alert and (datetime.now() - self._last_alert).seconds < 1800:
            return
        self._last_alert = datetime.now()
        try:
            import winsound
            # Frequenza: 880Hz, durata: 500ms, 3 ripetizioni
            for _ in range(3):
                winsound.Beep(880, 400)
                time.sleep(0.15)
        except Exception:
            pass

    def alert_event(self):
        """Beep singolo per eventi macro imminenti."""
        try:
            import winsound
            winsound.Beep(660, 300)
        except Exception:
            pass


# =============================================================================
# CALENDARIO ECONOMICO
# =============================================================================

class EconomicCalendar:
    """
    Calendario eventi macro con sourcing multiplo:
    1. ForexFactory (via requests + BeautifulSoup)
    2. Fallback: eventi noti del mese corrente
    """

    MONTHS_IT = {
        "Jan": "Gen", "Feb": "Feb", "Mar": "Mar", "Apr": "Apr",
        "May": "Mag", "Jun": "Giu", "Jul": "Lug", "Aug": "Ago",
        "Sep": "Set", "Oct": "Ott", "Nov": "Nov", "Dec": "Dic",
    }

    # Eventi ad alto impatto — pattern da cercare nel titolo
    HIGH_IMPACT_EVENTS = [
        "NFP", "Non Farm", "Payrolls", "Employment",
        "CPI", "Consumer Price",
        "FOMC", "Fed Decision", "Interest Rate",
        "GDP", "Gross Domestic",
        "PPI", "Producer Price",
        "Retail Sales",
        "ISM", "PMI",
        "Michigan", "Unemployment Rate",
        "JOLTS", "ADP",
    ]

    def fetch(self) -> pd.DataFrame:
        """Recupera eventi macro della settimana corrente."""
        # Prova sourcing live
        events = self._scrape_forexfactory()
        if events is None:
            events = self._scrape_investing()
        if events is None:
            events = self._fallback_events()
        return events

    def _scrape_forexfactory(self) -> Optional[pd.DataFrame]:
        """Scraping ForexFactory calendar (gratuito, no API key)."""
        try:
            import requests
            from bs4 import BeautifulSoup

            today = datetime.now()
            # FF mostra la settimana corrente
            url = "https://www.forexfactory.com/calendar"
            headers = {
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 Chrome/120.0.0.0"),
            }
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.select("tr.calendar__row")
            events = []
            current_date = today.date()

            for row in rows:
                date_cell = row.select_one("td.calendar__date")
                if date_cell:
                    date_text = date_cell.text.strip()
                    for eng, it in self.MONTHS_IT.items():
                        date_text = date_text.replace(eng, it)
                    try:
                        current_date = datetime.strptime(date_text, "%a %b %d, %Y").date()
                    except ValueError:
                        try:
                            current_date = datetime.strptime(date_text, "%b %d, %Y").date()
                        except ValueError:
                            continue

                time_cell = row.select_one("td.calendar__time")
                event_time_str = time_cell.text.strip() if time_cell else ""
                if event_time_str in ("", "All Day", "Day"):
                    continue

                title_cell = row.select_one("td.calendar__event")
                if not title_cell:
                    continue
                title = title_cell.text.strip()
                impact_span = row.select_one("td.calendar__impact span")
                impact = impact_span["title"] if impact_span and impact_span.get("title") else ""

                # Solo eventi ad alto impatto
                if "High" not in impact:
                    continue

                # Parsing ora
                event_dt = None
                try:
                    hour, minute = event_time_str.split(":")
                    ampm = "AM" if "AM" in minute else "PM" if "PM" in minute else ""
                    minute = minute.replace("AM", "").replace("PM", "").strip()
                    event_dt = datetime.strptime(
                        f"{current_date} {hour}:{minute} {ampm}",
                        "%Y-%m-%d %I:%M %p"
                    )
                except Exception:
                    continue

                events.append({
                    "datetime": event_dt,
                    "title": title,
                    "impact": "High",
                    "source": "ForexFactory",
                })

            if events:
                return pd.DataFrame(events).sort_values("datetime")

        except Exception as e:
            print(f"FF scrape error: {e}")
        return None

    def _scrape_investing(self) -> Optional[pd.DataFrame]:
        """Fallback: Investing.com economic calendar."""
        try:
            import requests
            from bs4 import BeautifulSoup

            headers = {
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 Chrome/120.0.0.0"),
                "X-Requested-With": "XMLHttpRequest",
            }
            today = datetime.now()
            data = {
                "country[]": "72",  # US
                "timeZone": "5",
                "dateFrom": today.strftime("%Y-%m-%d"),
                "dateTo": (today + timedelta(days=7)).strftime("%Y-%m-%d"),
                "currentTab": "calendar",
                "limit_from": 0,
            }
            resp = requests.post(
                "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData",
                headers=headers, data=data, timeout=10,
            )
            if resp.status_code != 200:
                return None

            raw = resp.json()
            soup = BeautifulSoup(raw.get("data", ""), "html.parser")
            rows = soup.select("tr.js-event-item")
            events = []

            for row in rows:
                impact_el = row.select_one("td.impact")
                impact = "Low"
                if impact_el:
                    icons = impact_el.select("i")
                    if len(icons) >= 3:
                        impact = "High"
                    elif len(icons) == 2:
                        impact = "Medium"

                if impact != "High":
                    continue

                title_el = row.select_one("td.earnCalEvent")
                if not title_el:
                    continue
                title = title_el.text.strip()

                time_el = row.select_one("td.time")
                if not time_el:
                    continue
                time_str = time_el.text.strip()

                date_el = row.select_one("td.date")
                date_str = date_el.text.strip() if date_el else today.strftime("%Y-%m-%d")

                try:
                    event_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                except ValueError:
                    try:
                        event_dt = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M")
                    except ValueError:
                        continue

                events.append({
                    "datetime": event_dt,
                    "title": title,
                    "impact": impact,
                    "source": "Investing.com",
                })

            if events:
                return pd.DataFrame(events).sort_values("datetime")

        except Exception as e:
            print(f"Investing scrape error: {e}")
        return None

    def _fallback_events(self) -> pd.DataFrame:
        """Fallback: eventi noti del mese corrente da pattern."""
        import calendar as cal_mod
        today = datetime.now()
        rows = []

        # Terzo venerdi del mese = OpEx
        cal = cal_mod.monthcalendar(today.year, today.month)
        third_friday = None
        for week in cal:
            if week[cal_mod.FRIDAY] != 0:
                if third_friday is None:
                    third_friday = week[cal_mod.FRIDAY]
                else:
                    third_friday = week[cal_mod.FRIDAY]
                    break
        if third_friday:
            opex = today.replace(day=third_friday, hour=8, minute=30)
            rows.append({"datetime": opex, "title": "Options Expiry (OpEx)",
                         "impact": "High", "source": "Calendario"})

        # Primo venerdi = NFP (approssimazione)
        first_friday = None
        for week in cal:
            if week[cal_mod.FRIDAY] != 0:
                first_friday = week[cal_mod.FRIDAY]
                break
        if first_friday:
            nfp = today.replace(day=first_friday, hour=8, minute=30)
            rows.append({"datetime": nfp, "title": "NFP (Employment Situation)",
                         "impact": "High", "source": "Calendario"})

        # CPI meta' mese
        cpi_day = 14 if today.month in (2, 4, 6, 8, 10, 12) else 15
        cpi = today.replace(day=min(cpi_day, cal_mod.monthrange(today.year, today.month)[1]),
                           hour=8, minute=30)
        rows.append({"datetime": cpi, "title": "CPI (Consumer Price Index)",
                     "impact": "High", "source": "Calendario"})

        # FOMC — 8 incontri all'anno, approssimazione
        fomc_dates = {
            1: 29, 3: 19, 5: 7, 6: 18, 7: 30, 9: 17, 11: 7, 12: 17,
        }
        if today.month in fomc_dates:
            fomc = today.replace(day=fomc_dates[today.month], hour=14, minute=0)
            rows.append({"datetime": fomc, "title": "FOMC Decision",
                         "impact": "High", "source": "Calendario"})

        return pd.DataFrame(rows).sort_values("datetime") if rows else pd.DataFrame()

    def get_upcoming(self, hours: int = 48) -> pd.DataFrame:
        """Eventi nelle prossime N ore."""
        df = self.fetch()
        if df.empty:
            return df
        now = datetime.now()
        cutoff = now + timedelta(hours=hours)
        mask = (df["datetime"] >= now) & (df["datetime"] <= cutoff)
        return df[mask].sort_values("datetime")

    def imminent_events(self, minutes: int = 30) -> pd.DataFrame:
        """Eventi entro N minuti."""
        df = self.get_upcoming(hours=2)
        if df.empty:
            return df
        now = datetime.now()
        cutoff = now + timedelta(minutes=minutes)
        return df[df["datetime"] <= cutoff]


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("📊 Macro Quant")
st.sidebar.caption("GIP Regime Detection · Multi-Asset")

frequency = st.sidebar.radio("Frequenza", ["DAILY", "WEEKLY"], index=0)
auto_refresh = st.sidebar.checkbox("Auto-refresh ogni 5 min", value=False)
audio_on = st.sidebar.checkbox("Alert sonori", value=True)

if st.sidebar.button("🔄 Refresh Data", type="primary", use_container_width=True):
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Fattori monitorati")
st.sidebar.markdown("""
- **SPX** — S&P 500 Index
- **VIX** — Volatility Index
- **DXY** — US Dollar Index
- **COPPER** — Copper Futures
- **WTI** — Crude Oil Futures
- **SKEW** — CBOE Skew Index
""")

# Pulsante esportazione
st.sidebar.divider()
st.sidebar.subheader("Esporta")
export_format = st.sidebar.selectbox("Formato", ["HTML", "CSV storico"], index=0)
if st.sidebar.button("📥 Scarica Report", use_container_width=True):
    st.sidebar.success("Report generato — vedi sezione sotto")

st.sidebar.divider()
st.sidebar.caption(f"Ultimo aggiornamento: {datetime.now():%H:%M:%S}")

# =============================================================================
# MAIN
# =============================================================================

st.title("📈 Macro Quant — GIP Regime Dashboard")
st.markdown("Analisi multi-asset per determinare il regime macro **Growth–Inflation–Policy**.")

# Init
db = MacroHistoryDB()
alerter = AudioAlerter(enabled=audio_on)
calendar = EconomicCalendar()

# --- Run system ---
with st.spinner("Scaricamento dati e analisi in corso..."):
    system = MacroQuantSystem()
    report, charts, regime, bias, shift = system.run(frequency=frequency)

    signals = regime.get("signals", {})
    gip = regime.get("gip", {})
    r = regime.get("regime", Regime.TRANSITION)
    shift_prob = shift.get("shift_probability", 0)

    # Log SQLite
    db.log_scan(regime, bias, shift, signals)

    # Audio alert per shift
    alerter.alert_shift(shift_prob)
    if shift_prob > 0.5:
        db.log_alert("REGIME_SHIFT", f"Shift prob: {shift_prob:.0%} — regime attuale: {r.value}")

# =============================================================================
# TOP BANNER — Regime
# =============================================================================

regime_colors = {
    Regime.EXPANSION: "#00cc66",
    Regime.SLOWDOWN: "#3399ff",
    Regime.REFLATION: "#ff8800",
    Regime.STAGFLATION: "#ff3333",
    Regime.TRANSITION: "#999999",
}
bg = regime_colors.get(r, "#999999")
shift_color = "#ff4444" if shift_prob > 0.5 else "#ffaa00" if shift_prob > 0.3 else "#888"

st.markdown(
    f"""
    <div style="
        background: {bg};
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 14px; opacity: 0.9;">REGIME GIP</span><br>
                <span style="font-size: 36px; font-weight: 700;">{r.value}</span>
            </div>
            <div style="text-align: center;">
                <span style="font-size: 14px; opacity: 0.9;">CONFIDENZA</span><br>
                <span style="font-size: 28px; font-weight: 600;">{regime.get("confidence", 0):.1f}</span>
            </div>
            <div style="text-align: center;">
                <span style="font-size: 14px; opacity: 0.9;">BIAS</span><br>
                <span style="font-size: 28px; font-weight: 600;">{bias.value}</span>
            </div>
            <div style="text-align: center;">
                <span style="font-size: 14px; opacity: 0.9;">SHIFT 🚨</span><br>
                <span style="font-size: 28px; font-weight: 600; color: {shift_color};">{shift_prob*100:.0f}%</span>
            </div>
            <div style="text-align: right; font-size: 13px; opacity: 0.9;">
                {frequency}<br>
                {datetime.now():%Y-%m-%d %H:%M}
            </div>
        </div>
        {"<div style='margin-top: 8px; font-weight: 600;'>🚨 REGIME SHIFT — preparare piano di contingenza</div>" if shift_prob > 0.5 else ""}
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# CALENDARIO ECONOMICO
# =============================================================================

st.subheader("📅 Eventi Macro Imminenti")
with st.spinner("Caricamento calendario..."):
    econ_df = calendar.get_upcoming(hours=48)
    imminent_df = calendar.imminent_events(minutes=30)

if not econ_df.empty:
    econ_df["datetime"] = pd.to_datetime(econ_df["datetime"])
    econ_df["Orario"] = econ_df["datetime"].dt.strftime("%a %d %b %H:%M")
    econ_df["Tra"] = econ_df["datetime"].apply(
        lambda x: f"{int((x - datetime.now()).total_seconds() // 60)}min"
        if x > datetime.now() else "IN CORSO"
    )

    now = datetime.now()
    imminent_mask = (econ_df["datetime"] <= now + timedelta(minutes=30)) & (econ_df["datetime"] > now)

    display_df = econ_df[["Orario", "title", "impact", "Tra", "source"]].copy()
    display_df.columns = ["Orario", "Evento", "Impatto", "Tra", "Fonte"]

    def highlight_imminent(row):
        if row.name in econ_df.index[imminent_mask]:
            return ["background-color: #ff4444; color: white"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_imminent, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    if not imminent_df.empty:
        st.error(f"⚠️ **{len(imminent_df)} evento/i entro 30min!** — Nessun trade prima/dopo (dalla tua SOP)")
        alerter.alert_event()
        for _, ev in imminent_df.iterrows():
            st.warning(f"🚨 {ev['title']} — tra {int((ev['datetime'] - datetime.now()).total_seconds() // 60)}min")
            db.log_alert("EVENT_IMMINENT", f"{ev['title']} tra {int((ev['datetime'] - datetime.now()).total_seconds() // 60)}min")

elif econ_df.empty:
    st.info("Nessun evento ad alto impatto nelle prossime 48h.")

else:
    st.info("Nessun evento nelle prossime 48h.")

# =============================================================================
# STORICO GIP
# =============================================================================

with st.expander("📈 Storico GIP (ultimi 30 giorni)", expanded=False):
    hist_df = db.history(days=30)
    if not hist_df.empty:
        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist_df["timestamp"], y=hist_df["g_score"],
            mode="lines+markers", name="G-Score",
            line=dict(color="#00cc66"),
        ))
        fig_hist.add_trace(go.Scatter(
            x=hist_df["timestamp"], y=hist_df["i_score"],
            mode="lines+markers", name="I-Score",
            line=dict(color="#ff8800"),
        ))
        fig_hist.add_trace(go.Scatter(
            x=hist_df["timestamp"], y=hist_df["shift_prob"],
            mode="lines+markers", name="Shift Prob",
            line=dict(color="#ff4444", dash="dot"),
            yaxis="y2",
        ))

        # Regime background
        regimes_hist = hist_df["regime"].unique()
        colors_hist = {"EXPANSION": "rgba(0,204,102,0.1)", "SLOWDOWN": "rgba(51,153,255,0.1)",
                       "REFLATION": "rgba(255,136,0,0.1)", "STAGFLATION": "rgba(255,51,51,0.1)",
                       "TRANSIZIONE": "rgba(153,153,153,0.1)"}
        for i in range(len(hist_df)):
            hist_df.at[hist_df.index[i], "_color"] = colors_hist.get(hist_df.iloc[i]["regime"], "rgba(0,0,0,0)")

        fig_hist.update_layout(
            height=300,
            yaxis=dict(title="G/I Score", range=[-1.2, 1.2]),
            yaxis2=dict(title="Shift Prob", overlaying="y", side="right", range=[0, 1]),
            hovermode="x unified",
            margin=dict(l=10, r=10, t=10, b=30),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Tabella storico
        st.dataframe(
            hist_df[["timestamp", "regime", "g_score", "i_score", "bias",
                     "vix", "wti", "copper", "dxy"]].tail(10)
            .style.format({
                "g_score": "{:+.3f}", "i_score": "{:+.3f}",
                "vix": "{:.1f}", "wti": "${:.1f}",
                "copper": "${:.2f}", "dxy": "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": "Data", "regime": "Regime",
                "g_score": "G", "i_score": "I", "bias": "Bias",
                "vix": "VIX", "wti": "WTI", "copper": "Cu", "dxy": "DXY",
            },
        )
    else:
        st.info("Ancora nessuno storico. Fai partire alcuni scan per popolarlo.")

# =============================================================================
# ROW 1 — GIP Scores + Prezzi
# =============================================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 GIP Scores")
    g_val = gip.get("G", 0)
    i_val = gip.get("I", 0)

    fig_gip = go.Figure()
    fig_gip.add_trace(go.Bar(
        x=["Growth (G)", "Inflation (I)"],
        y=[g_val, i_val],
        marker_color=["#00cc66" if g_val > 0 else "#ff3333",
                      "#ff8800" if i_val > 0 else "#3399ff"],
        text=[f"{g_val:+.3f}", f"{i_val:+.3f}"],
        textposition="outside",
    ))
    fig_gip.update_layout(
        height=250,
        yaxis=dict(range=[-1, 1], title="Score"),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=30),
    )
    st.plotly_chart(fig_gip, use_container_width=True)

    g_tag = "G↑" if g_val > 0 else "G↓"
    i_tag = "I↓" if i_val < -0.1 else "I↑" if i_val > 0.1 else "I→"
    st.markdown(f"**Posizione GIP:** {g_tag} {i_tag} → **{r.value}**")

with col2:
    st.subheader("💰 Prezzi Correnti")
    raw = system.data_fetcher.fetch_all()
    full_df = DataFetcher.align_series(raw)
    price_rows = []
    all_cols = ["SPX", "VIX", "DXY", "COPPER", "WTI", "SKEW"]
    for col in all_cols:
        if col in full_df.columns:
            try:
                v = float(full_df[col].iloc[-1])
                fmt = "{:.2f}"
                if col in ("VIX",): fmt = "{:.1f}"
                elif col in ("DXY",): fmt = "{:.1f}"
                elif col == "SKEW": fmt = "{:.0f}"
                elif col == "SPX": fmt = "{:.2f}"
                price_rows.append({"Asset": col, "Valore": fmt.format(v)})
            except (ValueError, IndexError, TypeError):
                pass
    st.dataframe(
        pd.DataFrame(price_rows),
        use_container_width=True,
        hide_index=True,
        column_config={"Asset": "Asset", "Valore": "Valore"},
    )
    st.markdown(f"**VIX Term:** {signals.get('vix_term', 'N/D')} | "
                f"**SKEW:** {signals.get('skew_signal', 'N/D')}")

# =============================================================================
# ROW 2 — Heatmaps
# =============================================================================

st.subheader("🔥 Heatmap Correlazioni")
col1, col2 = st.columns(2)

with col1:
    st.caption("Correlazioni correnti (rendimenti)")
    if not full_df.empty:
        corr = full_df.pct_change().dropna().corr()
        fig_hm = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:+.3f}" for v in row] for row in corr.values],
            texttemplate="%{text}", textfont={"size": 10},
        ))
        fig_hm.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=30))
        st.plotly_chart(fig_hm, use_container_width=True)

with col2:
    st.caption("Evoluzione correlazione SPX vs fattori (rolling 63gg)")
    if not full_df.empty and len(full_df) > 63:
        returns = full_df.pct_change().dropna()
        factors = [c for c in returns.columns if c != "SPX"]
        roll_data = {}
        for f in factors:
            roll_data[f] = returns["SPX"].rolling(63).corr(returns[f])
        roll_df = pd.DataFrame(roll_data).dropna()
        if not roll_df.empty:
            fig_roll = go.Figure(data=go.Heatmap(
                z=roll_df.T.values, x=roll_df.index.tolist(), y=factors,
                colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
            ))
            fig_roll.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=30))
            st.plotly_chart(fig_roll, use_container_width=True)

# =============================================================================
# ROW 3 — 3D + Z-Score
# =============================================================================

st.subheader("🎯 Regime Space & Anomalie")
col1, col2 = st.columns(2)

with col1:
    st.caption("3D Macro Regime Space (SPX · VIX · DXY)")
    if not full_df.empty and len(full_df) > 100:
        detector = RegimeDetector()
        regime_hist = detector.detect_history(full_df)
        if not regime_hist.empty:
            plot_df = full_df[["SPX", "VIX", "DXY"]].loc[regime_hist.index].copy()
            plot_df["Regime"] = regime_hist
            reg_names = plot_df["Regime"].unique()
            cmap3d = {"EXPANSION": "green", "SLOWDOWN": "blue",
                      "REFLATION": "orange", "STAGFLATION": "red", "TRANSIZIONE": "gray"}
            fig3d = go.Figure()
            for rn in reg_names:
                sub = plot_df[plot_df["Regime"] == rn]
                if len(sub) > 500:
                    sub = sub.sample(500, random_state=42)
                fig3d.add_trace(go.Scatter3d(
                    x=sub["SPX"], y=sub["VIX"], z=sub["DXY"],
                    mode="markers", name=rn,
                    marker=dict(size=3, color=cmap3d.get(rn, "gray"), opacity=0.6),
                ))
            last = plot_df.iloc[-1]
            fig3d.add_trace(go.Scatter3d(
                x=[last["SPX"]], y=[last["VIX"]], z=[last["DXY"]],
                mode="markers", name=f"CORRENTE ({last['Regime']})",
                marker=dict(size=12, color="yellow", symbol="diamond",
                            line=dict(width=2, color="black")),
            ))
            fig3d.update_layout(
                height=450,
                scene=dict(xaxis_title="SPX", yaxis_title="VIX", zaxis_title="DXY"),
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig3d, use_container_width=True)

with col2:
    st.caption("Z-Score: anomalie storiche per asset")
    if not full_df.empty and len(full_df) > 252:
        zs = full_df.apply(lambda x: (x - x.rolling(252).mean()) / x.rolling(252).std())
        zs = zs.dropna()
        if not zs.empty:
            fig_z = go.Figure(data=go.Heatmap(
                z=zs.T.values, x=zs.index.tolist(), y=zs.columns.tolist(),
                colorscale="RdYlGn", zmid=0, zmin=-3, zmax=3,
            ))
            fig_z.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=30))
            st.plotly_chart(fig_z, use_container_width=True)

# =============================================================================
# ROW 4 — Validazione Statistica
# =============================================================================

st.subheader("✅ Validazione Statistica")
col1, col2 = st.columns(2)

with col1:
    st.caption("Bonferroni Correction")
    validator = StatisticalValidator()
    bonf = validator.bonferroni_correction(full_df) if not full_df.empty else {}
    results = bonf.get("results", [])
    if results:
        bf_df = pd.DataFrame(results)
        bf_df["significant"] = bf_df["significant"].apply(lambda x: "✅ [S]" if x else "❌ [NS]")
        bf_df = bf_df.rename(columns={
            "factor": "Fattore", "correlation": "Corr", "p_value": "p-value",
            "significant": "Signif", "interpretation": "Interpretazione",
        })
        st.dataframe(bf_df[["Fattore", "Corr", "p-value", "Signif", "Interpretazione"]],
                     use_container_width=True, hide_index=True)
    st.caption(f"Alpha corretto: {bonf.get('corrected_alpha', 'N/D')} | Test: {bonf.get('n_tests', 0)}")

with col2:
    st.caption("Walk-Forward Validation")
    wf = validator.walk_forward_validation(full_df) if not full_df.empty else {}
    stab = wf.get("stability_score", 0)
    st.markdown(f"**Stabilità:** {stab:.1f}%")
    if wf.get("stable_factors"):
        st.markdown(f"✅ Fattori STABILI: {', '.join(wf['stable_factors'])}")
    if wf.get("unstable_factors"):
        st.markdown(f"⚠️ Fattori INSTABILI: {', '.join(wf['unstable_factors'])}")
    fold_corrs = wf.get("fold_correlations", {})
    if fold_corrs:
        fold_df = pd.DataFrame(fold_corrs)
        fold_df.index = [f"Fold {i+1}" for i in range(len(fold_df))]
        st.dataframe(fold_df.style.format("{:+.3f}"), use_container_width=True)

# =============================================================================
# ROW 5 — Shift Prediction
# =============================================================================

st.subheader("🚨 Regime Shift Prediction")
shift_signals = shift.get("signals", [])
if shift_signals:
    for s in shift_signals:
        st.warning(s)
else:
    st.info("Nessun segnale di shift imminente.")
if shift.get("direction"):
    st.markdown(f"**Direzione prevista:** {shift['direction']}")

# =============================================================================
# ESPORTAZIONE
# =============================================================================

st.divider()
st.subheader("📥 Esporta Report")

col_a, col_b = st.columns(2)

with col_a:
    if st.button("📄 Scarica HTML Report", use_container_width=True):
        html_parts = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>Macro Quant Report</title>",
            "<style>body{font-family:system-ui;max-width:900px;margin:40px auto;padding:20px}",
            "h1{color:#333}.regime-banner{padding:20px;border-radius:12px;color:#fff;font-size:24px;font-weight:700}",
            "table{width:100%;border-collapse:collapse;margin:20px 0}",
            "td,th{border:1px solid #ddd;padding:8px;text-align:left}",
            "th{background:#f5f5f5}.badge{display:inline-block;padding:4px 12px;border-radius:20px;color:#fff;font-size:13px}</style></head><body>",
            f"<h1>Macro Quant Report — {frequency}</h1>",
            f"<p>{datetime.now():%Y-%m-%d %H:%M}</p>",
            f"<div class='regime-banner' style='background:{bg};'>",
            f"Regime: {r.value} | Bias: {bias.value} | Shift: {shift_prob*100:.0f}%</div>",
            "<h2>GIP Scores</h2>",
            f"<p>G-Score: {gip.get('G', 0):+.3f}<br>I-Score: {gip.get('I', 0):+.3f}</p>",
            "<h2>Prezzi Correnti</h2><table><tr><th>Asset</th><th>Valore</th></tr>",
        ]
        for pr in price_rows:
            html_parts.append(f"<tr><td>{pr['Asset']}</td><td>{pr['Valore']}</td></tr>")
        html_parts.append("</table>")

        html_parts.append("<h2>Segnale Trading</h2><pre>")
        signal_text = f"""REGIME: {r.value}
BIAS: {bias.value}
G_SCORE: {gip.get('G', 0):+.3f}
I_SCORE: {gip.get('I', 0):+.3f}
VIX: {signals.get('vix_level', 0):.1f}
WTI: ${signals.get('wti_level', 0):.1f}
COPPER: ${signals.get('copper_level', 0):.2f}
DXY: {signals.get('dxy_level', 0):.1f}
SKEW: {signals.get('skew_level', 0):.0f}
SHIFT_PROB: {shift_prob*100:.0f}%
"""
        html_parts.append(signal_text)
        html_parts.append("</pre></body></html>")

        html_content = "\n".join(html_parts)
        st.download_button(
            label="💾 Salva HTML",
            data=html_content,
            file_name=f"macro_report_{datetime.now():%Y%m%d_%H%M}.html",
            mime="text/html",
            use_container_width=True,
        )

with col_b:
    if st.button("📊 Scarica CSV Storico", use_container_width=True):
        hist_csv = db.history(days=90)
        if not hist_csv.empty:
            csv_buffer = io.StringIO()
            hist_csv.to_csv(csv_buffer, index=False)
            st.download_button(
                label="💾 Salva CSV",
                data=csv_buffer.getvalue(),
                file_name=f"macro_history_{datetime.now():%Y%m%d}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Nessuno storico disponibile.")

# =============================================================================
# AUTO-REFRESH
# =============================================================================

if auto_refresh:
    st.toast("Auto-refresh attivo — prossimo refresh tra 5 min", icon="⏰")
    time.sleep(300)
    st.rerun()

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption(f"Macro Quant Dashboard · DB: {DB_PATH} · Ultimo refresh: {datetime.now():%H:%M:%S}")
