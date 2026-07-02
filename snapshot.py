"""Genera report snapshot dello stato corrente del terminale.
Uso: python snapshot.py [--force] [--now]
Legge da macro_history.db e salva report .md in 2nd-brain\.claude\
"""

import sqlite3, argparse, sys, json
from pathlib import Path
from datetime import datetime

DB = Path(__file__).parent / "macro_history.db"
OUT = Path(__file__).parent

def fmt_num(v, dec=1):
    if v is None: return "—"
    try: return f"{float(v):,.{dec}f}"
    except: return str(v)

def fmt_signed(v, dec=1):
    if v is None: return "—"
    try: return f"{float(v):+,.{dec}f}"
    except: return str(v)

def fmt_pct(v):
    if v is None: return "—"
    try: return f"{float(v)*100:.0f}%"
    except: return str(v)

def emoji_regime(r):
    return {"EXPANSION":"🟢","SLOWDOWN":"🔵","REFLATION":"🟠","STAGFLATION":"🔴","TRANSIZIONE":"⚪"}.get(str(r).upper(),"⚪")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="forza sovrascrittura")
    ap.add_argument("--now", action="store_true", help="genera report immediato")
    args = ap.parse_args()
    if not args.force and not args.now:
        print("Usa: python snapshot.py --force --now")
        sys.exit(1)

    if not DB.exists():
        print(f"❌ DB non trovato: {DB}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Ultimo eod_snapshot
    snap = None
    try:
        c.execute("SELECT * FROM eod_snapshots ORDER BY date DESC LIMIT 1")
        snap = c.fetchone()
    except: pass

    # Ultima riga macro_history
    hist = None
    try:
        c.execute("SELECT * FROM macro_history ORDER BY timestamp DESC LIMIT 1")
        hist = c.fetchone()
    except: pass

    # Storico ultimi 7 giorni macro
    trend = []
    try:
        c.execute("SELECT timestamp,regime,g_score,i_score,shift_prob,vix FROM macro_history ORDER BY timestamp DESC LIMIT 7")
        trend = [dict(r) for r in c.fetchall()]
    except: pass

    conn.close()

    ts = datetime.now()
    ts_str = ts.strftime("%Y-%m-%d_%H%M")

    if snap:
        r = dict(snap)
    else:
        r = {}

    lines = []
    lines.append(f"# ⬡ Maze Capital — Snapshot Report")
    lines.append(f"**Generato:** {ts.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 📊 Riepilogo")
    lines.append(f"")
    if snap:
        lines.append(f"| Metrica | Valore |")
        lines.append(f"|---------|--------|")
        lines.append(f"| **Ticker** | {r.get('ticker','—')} |")
        lines.append(f"| **Spot** | {fmt_num(r.get('spot'),2)} |")
        lines.append(f"| **Regime** | {emoji_regime(r.get('regime'))} {r.get('regime','—')} |")
        lines.append(f"| **GEX totale** | {fmt_signed(r.get('total_gex'),0)} |")
        lines.append(f"| **Zero Gamma** | {fmt_num(r.get('zero_gamma'),1)} |")
        lines.append(f"| **Call Wall** | {fmt_num(r.get('call_wall'),1)} |")
        lines.append(f"| **Put Wall** | {fmt_num(r.get('put_wall'),1)} |")
        lines.append(f"| **G-Score** | {fmt_signed(r.get('g_score'),3)} |")
        lines.append(f"| **I-Score** | {fmt_signed(r.get('i_score'),3)} |")
        lines.append(f"| **Shift Prob** | {fmt_pct(r.get('shift_prob'))} |")
        lines.append(f"| **VIX** | {fmt_num(r.get('vix'),1)} |")
        lines.append(f"| **Gamma Env** | {r.get('gamma_env','—')} |")
        lines.append(f"| **Expected Move ↑** | {fmt_num(r.get('em_up'),1)} |")
        lines.append(f"| **Expected Move ↓** | {fmt_num(r.get('em_down'),1)} |")
        if r.get('skew') is not None:
            lines.append(f"| **Skew** | {fmt_num(r.get('skew'),0)} |")
        lines.append(f"")
    else:
        lines.append(f"_Nessun snapshot disponibile. Avvia il terminale prima di generare un report._")
        lines.append(f"")

    if hist:
        h = dict(hist)
        lines.append(f"## 📈 Storico GIP (ultimo)")
        lines.append(f"")
        lines.append(f"| Timestamp | Regime | G | I | Shift | VIX |")
        lines.append(f"|-----------|--------|---|---|-------|-----|")
        lines.append(f"| {h.get('timestamp','—')} | {h.get('regime','—')} | {fmt_signed(h.get('g_score'),3)} | {fmt_signed(h.get('i_score'),3)} | {fmt_pct(h.get('shift_prob'))} | {fmt_num(h.get('vix'),1)} |")
        lines.append(f"")

    if trend:
        lines.append(f"### Trend 7 giorni")
        lines.append(f"")
        lines.append(f"| Data | Regime | G | I | Shift | VIX |")
        lines.append(f"|------|--------|---|---|-------|-----|")
        for t in trend:
            lines.append(f"| {t.get('timestamp','—')[:10]} | {t.get('regime','—')} | {fmt_signed(t.get('g_score'),3)} | {fmt_signed(t.get('i_score'),3)} | {fmt_pct(t.get('shift_prob'))} | {fmt_num(t.get('vix'),1)} |")
        lines.append(f"")

    lines.append(f"---")
    lines.append(f"_Report generato da Maze Capital Terminal v5.0_")

    fname = f"snapshot_{ts_str}.md"
    fpath = OUT / fname
    fpath.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Report salvato: {fpath}")

if __name__ == "__main__":
    main()
