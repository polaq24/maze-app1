"""MGT-GO style panel — Candlestick + GEX Regime Heatmap sfondo + fix TF persist."""
import numpy as np
import json
from datetime import datetime


def _lw_bars(prices_hist: np.ndarray) -> list:
    """Generate fake OHLC bars from closing prices for demo."""
    n = len(prices_hist)
    if n == 0:
        return []
    from datetime import timedelta
    base = datetime.now() - timedelta(days=n)
    bars = []
    for i in range(n):
        p = float(prices_hist[i])
        bars.append({
            "time": int((base + timedelta(days=i)).timestamp()),
            "open":  p * (1 + np.random.uniform(-0.003, 0.003)),
            "high":  p * (1 + np.random.uniform(0, 0.005)),
            "low":   p * (1 - np.random.uniform(0, 0.005)),
            "close": p,
        })
    return bars


def generate_mgtgo_html(
    ticker: str,
    spot: float,
    zerogamma: float,
    major_pos: float | None,
    major_neg: float | None,
    gex_strikes: list,
    gex_values: list,
    gex_total: float,
    bars: list | None = None,
    gex_env: str = "POS",
    default_tf: str = "6M",
    heatmap_prices: list | None = None,
    heatmap_days: list | None = None,
    heatmap_matrix: list | None = None,
) -> str:

    down_color     = "#a855f7" if gex_env == "NEG" else "#ef4444"
    neg_gex_color  = "rgba(168,85,247,0.4)" if gex_env == "NEG" else "rgba(239,68,68,0.4)"
    pw_color       = "#a855f7" if gex_env == "NEG" else "#ef4444"
    env_badge_col  = "#a855f7" if gex_env == "NEG" else "#22c55e"

    lines = [
        {"price": spot,      "color": "rgba(255,255,255,0.9)", "title": f"{ticker} {spot:.0f}"},
        {"price": zerogamma, "color": "#fbbf24",               "title": f"ZG {zerogamma:.0f}"},
    ]
    if major_pos:
        lines.append({"price": major_pos, "color": "#22c55e", "title": f"CW {major_pos:.0f}"})
    if major_neg:
        lines.append({"price": major_neg, "color": pw_color,  "title": f"PW {major_neg:.0f}"})

    lines_json  = json.dumps(lines)
    gex_json    = json.dumps([{"strike": s, "gex": g} for s, g in zip(gex_strikes, gex_values)])
    bars_json   = json.dumps(bars) if bars else "null"

    hm_prices = json.dumps(heatmap_prices or [])
    hm_days   = json.dumps(heatmap_days   or [])
    hm_matrix = json.dumps(heatmap_matrix or [])

    tf_buttons = ""
    for tf in ["5m", "15m", "30m", "1h", "4h", "1D", "1W", "1M", "3M", "6M"]:
        active_style = "color:#00d2ff;border-color:#00d2ff;" if tf == default_tf else ""
        tf_buttons += (
            f'<button onclick="setTF(\'{tf}\')" id="btn_{tf}" '
            f'style="background:#1f2937;color:#64748b;border:1px solid #374151;'
            f'border-radius:3px;padding:1px 6px;font-size:9px;cursor:pointer;{active_style}">'
            f'{tf}</button>'
        )

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0e17;color:#e2e8f0;font-family:-apple-system,system-ui,sans-serif;
      height:100vh;display:flex;flex-direction:column;overflow:hidden}}
#header{{display:flex;justify-content:space-between;align-items:center;
         padding:4px 12px;background:#111827;border-bottom:1px solid #1f2937;flex-shrink:0}}
#header .tkr{{color:#00d2ff;font-weight:700;font-size:15px}}
#header .info{{display:flex;gap:14px;font-size:10px;color:#64748b;align-items:center}}
#header .info span{{color:#e2e8f0;font-weight:600}}
#tf-bar{{display:flex;gap:4px;align-items:center;margin-left:8px}}
#wrap{{flex:1;position:relative;min-height:0}}
#chart{{position:absolute;top:0;left:0;width:100%;height:100%}}
#heatmap-canvas{{position:absolute;top:0;left:0;width:100%;height:100%;z-index:1;pointer-events:none}}
#footer{{display:flex;justify-content:space-between;padding:2px 12px;font-size:9px;
         color:#64748b;background:#111827;border-top:1px solid #1f2937;flex-shrink:0}}
.badge{{display:inline-block;background:#1f2937;padding:1px 6px;border-radius:3px;font-size:9px}}
.env-badge{{background:{env_badge_col}22;color:{env_badge_col};
            border:1px solid {env_badge_col};padding:1px 7px;border-radius:3px;font-size:9px;font-weight:700}}
</style>
<script src="https://unpkg.com/lightweight-charts@4.2.1/dist/lightweight-charts.standalone.production.js"></script>
</head>
<body>
<div id="header">
  <div style="display:flex;align-items:center;gap:8px">
    <span class="tkr">⬡ {ticker}</span>
    <span class="badge">⚡ GEXBot</span>
    <span class="env-badge">γ {gex_env}</span>
  </div>
  <div class="info">
    <div>ΣGEX <span style="color:#fbbf24;">{gex_total:+,.0f}</span></div>
    <div>ZG <span style="color:#fbbf24;">{zerogamma:.0f}</span></div>
    <div>Spot <span style="color:white;">{spot:.2f}</span></div>
    <div id="tf-bar">{tf_buttons}</div>
    <div id="clock" style="color:#64748b;font-size:9px;"></div>
  </div>
</div>

<div id="wrap">
  <canvas id="heatmap-canvas" style="position:absolute;top:0;left:0;width:100%;height:100%;z-index:1;pointer-events:none"></canvas>
  <div id="chart" style="position:absolute;top:0;left:0;width:100%;height:100%;z-index:2"></div>
</div>

<div id="footer">
  <span>Viola=NEG · Blu=POS · Bianco=ZeroGamma</span>
  <span>MAZE CAPITAL · {datetime.now():%Y-%m-%d %H:%M:%S}</span>
</div>

<script>
(function tick(){{
  document.getElementById('clock').textContent = new Date().toLocaleTimeString('it-IT');
  setTimeout(tick, 1000);
}})();

const lines    = {lines_json};
const gexData  = {gex_json};
const realBars = {bars_json};
const hmPrices = {hm_prices};
const hmDays   = {hm_days};
const hmMatrix = {hm_matrix};

let bars;
if (realBars && realBars.length > 0) {{
  bars = realBars;
}} else {{
  bars = [];
  let px = {spot};
  const nDays = 180;
  const now = Math.floor(Date.now()/1000);
  for (let i = nDays; i >= 0; i--) {{
    const t = now - i*86400;
    const chg = (Math.random()-0.48)*px*0.002;
    const o=px, c=px+chg;
    bars.push({{time:t, open:o, high:Math.max(o,c)*(1+Math.random()*0.003),
                low:Math.min(o,c)*(1-Math.random()*0.003), close:c}});
    px=c;
  }}
}}
const allBars = bars.slice();

// ── Heatmap canvas ─────────────────────────────────────────────────────────
const canvas  = document.getElementById('heatmap-canvas');
const ctx     = canvas.getContext('2d');

function gexColor(val, maxAbs) {{
  if (maxAbs === 0) return 'rgba(255,255,255,0.05)';
  const n = Math.max(-1, Math.min(1, val/maxAbs));
  if (n > 0) {{
    const a = Math.pow(n, 0.6) * 0.55;
    return `rgba(59,130,246,${{a.toFixed(3)}})`;
  }} else {{
    const a = Math.pow(-n, 0.6) * 0.55;
    return `rgba(168,85,247,${{a.toFixed(3)}})`;
  }}
}}

function drawHeatmap() {{
  if (!hmPrices.length || !hmMatrix.length) return;
  const w = canvas.width, h = canvas.height;
  if (w === 0 || h === 0) return;
  ctx.clearRect(0, 0, w, h);

  // Try priceToCoordinate, fallback a Y manuale
  const mid = hmPrices[Math.floor(hmPrices.length/2)];
  let useBuiltin = false;
  try {{ useBuiltin = chart.priceToCoordinate(mid) !== null; }} catch(e) {{}}

  const topMargin = 0.05 * h;
  const bottomMargin = 0.15 * h;
  const scaleH = h - topMargin - bottomMargin;

  const vRange = chart.timeScale().getVisibleRange();
  let minPx, maxPx;
  if (vRange) {{
    const visBars = allBars.filter(b => b.time >= vRange.from && b.time <= vRange.to);
    if (visBars.length > 0) {{
      maxPx = Math.max(...visBars.map(b => b.high));
      minPx = Math.min(...visBars.map(b => b.low));
    }}
  }}
  if (minPx === undefined) {{ minPx = Math.min(...hmPrices); maxPx = Math.max(...hmPrices); }}
  const pad = (maxPx - minPx) * 0.05;
  const prMin = minPx - pad, prMax = maxPx + pad;

  const yCoords = hmPrices.map(p => {{
    if (useBuiltin) {{
      try {{ const y = chart.priceToCoordinate(p); if (y !== null) return y; }} catch(e) {{}}
    }}
    return topMargin + scaleH * (1 - (p - prMin) / (prMax - prMin));
  }});

  const cellH = hmPrices.length > 1
    ? Math.abs(yCoords[1] - yCoords[0]) * 1.3
    : 25;

  const col0 = hmMatrix.map(r => r[0]);
  const maxAbs = Math.max(...col0.map(Math.abs), 1);

  hmPrices.forEach((p, i) => {{
    const y = yCoords[i];
    if (y === null || y === undefined || !isFinite(y)) return;
    ctx.fillStyle = gexColor(hmMatrix[i][0], maxAbs);
    ctx.fillRect(0, Math.floor(y - cellH/2), w, Math.ceil(cellH + 1));
  }});

  // Zero gamma line
  try {{
    let zgY;
    if (useBuiltin) {{
      const y = chart.priceToCoordinate({zerogamma});
      if (y !== null) zgY = y;
    }}
    if (zgY === undefined) {{
      zgY = topMargin + scaleH * (1 - ({zerogamma} - prMin) / (prMax - prMin));
    }}
    if (zgY && isFinite(zgY)) {{
      ctx.strokeStyle = '#fbbf2480';
      ctx.lineWidth = 1;
      ctx.setLineDash([4,4]);
      ctx.beginPath();
      ctx.moveTo(0, zgY);
      ctx.lineTo(w, zgY);
      ctx.stroke();
      ctx.setLineDash([]);
    }}
  }} catch(e) {{}}
}}

// ── LightweightCharts ──────────────────────────────────────────────────────
const chartEl = document.getElementById('chart');
const chart = LightweightCharts.createChart(chartEl, {{
  layout:  {{ background:{{type:'solid',color:'transparent'}}, textColor:'#64748b', fontSize:10 }},
  grid:    {{ vertLines:{{color:'#1f293740'}}, horzLines:{{color:'#1f293740'}} }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  rightPriceScale: {{ borderColor:'#1f2937', scaleMargins:{{top:0.05,bottom:0.15}} }},
  timeScale: {{ borderColor:'#1f2937', timeVisible:true }},
  handleScroll: true,
  handleScale:  true,
}});

const candleSeries = chart.addCandlestickSeries({{
  upColor:        '#22c55e',
  downColor:      '{down_color}',
  borderUpColor:  '#22c55e',
  borderDownColor:'{down_color}',
  wickUpColor:    '#22c55e',
  wickDownColor:  '{down_color}',
}});

lines.forEach(l => {{
  candleSeries.createPriceLine({{
    price: l.price, color: l.color, lineWidth:1,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    axisLabelVisible:true, title: l.title,
  }});
}});

const gexSeries = chart.addHistogramSeries({{
  priceFormat:{{type:'volume'}}, priceScaleId:'gex',
}});
chart.priceScale('gex').applyOptions({{
  scaleMargins:{{top:0.75,bottom:0}}, visible:false,
}});
const lastTime = allBars[allBars.length-1].time;
gexSeries.setData(gexData.map(d => ({{
  time: lastTime,
  value: Math.abs(d.gex),
  color: d.gex > 0 ? 'rgba(34,197,94,0.35)' : '{neg_gex_color}',
}})));

// ── TF con bar-count approach ──────────────────────────────────────────────
const tfSec = {{'5m':300,'15m':900,'30m':1800,'1h':3600,'4h':14400,
               '1D':86400,'1W':604800,'1M':2592000,'3M':7776000,'6M':15552000}};
const avgSpacing = (allBars[allBars.length-1].time - allBars[0].time) / (allBars.length - 1);

function setTF(tf) {{
  sessionStorage.setItem('mgtgo_tf', tf);
  document.querySelectorAll('[id^="btn_"]').forEach(b => {{
    const active = b.id === 'btn_'+tf;
    b.style.color       = active ? '#00d2ff' : '#64748b';
    b.style.borderColor = active ? '#00d2ff' : '#374151';
  }});
  candleSeries.setData(allBars);
  const nBars = Math.max(2, Math.round(tfSec[tf] / avgSpacing));
  const fromIdx = Math.max(0, allBars.length - nBars);
  chart.timeScale().setVisibleRange({{
    from: allBars[fromIdx].time,
    to: allBars[allBars.length-1].time,
  }});
  setTimeout(drawHeatmap, 150);
}}

const savedTF = sessionStorage.getItem('mgtgo_tf') || '{default_tf}';
candleSeries.setData(allBars);
setTF(savedTF);

// ── Heatmap ridisegnata su scroll/zoom ─────────────────────────────────────
chart.timeScale().subscribeVisibleLogicalRangeChange(() => setTimeout(drawHeatmap, 50));
setTimeout(drawHeatmap, 500);
setTimeout(drawHeatmap, 1000);
setTimeout(drawHeatmap, 2000);

// ── Resize ─────────────────────────────────────────────────────────────────
function onResize() {{
  const wrap = document.getElementById('wrap');
  const w = wrap.clientWidth, h = wrap.clientHeight;
  canvas.width  = w;
  canvas.height = h;
  chart.applyOptions({{width:w, height:h}});
  drawHeatmap();
}}
window.addEventListener('resize', onResize);
setTimeout(onResize, 100);
</script>
</body>
</html>"""


def render_mgtgo(
    st,
    ticker: str,
    spot: float,
    gexbot_gex: dict | None,
    price_data: np.ndarray | None,
    ohlc_bars: list | None = None,
    gex_env: str = "POS",
    chart_height: int = 600,
    default_tf: str = "6M",
    heatmap_data: tuple | None = None,
):
    if not gexbot_gex or len(gexbot_gex.get("strikes", [])) == 0:
        st.caption("GEXBot N/D — MGT-GO panel non disponibile")
        return

    strikes   = [int(s)   for s in gexbot_gex["strikes"]]
    values    = [float(v) for v in gexbot_gex["gex"]]
    zg        = float(gexbot_gex.get("zero_gamma", spot))
    total     = float(gexbot_gex.get("total_gex", 0))
    major_pos = gexbot_gex.get("major_pos")
    major_neg = gexbot_gex.get("major_neg")

    bars = ohlc_bars or (_lw_bars(price_data) if price_data is not None else None)

    hm_prices = hm_days = hm_matrix = None
    if heatmap_data:
        hm_prices, hm_days, hm_mat = heatmap_data
        if hm_mat is not None:
            hm_prices = hm_prices.tolist() if hasattr(hm_prices, 'tolist') else list(hm_prices)
            hm_days   = hm_days.tolist()   if hasattr(hm_days,   'tolist') else list(hm_days)
            hm_matrix = hm_mat.tolist()    if hasattr(hm_mat,    'tolist') else list(hm_mat)

    html = generate_mgtgo_html(
        ticker=ticker, spot=spot, zerogamma=zg,
        major_pos=major_pos, major_neg=major_neg,
        gex_strikes=strikes, gex_values=values,
        gex_total=total, bars=bars,
        gex_env=gex_env,
        default_tf=default_tf,
        heatmap_prices=hm_prices,
        heatmap_days=hm_days,
        heatmap_matrix=hm_matrix,
    )
    st.components.v1.html(html, height=chart_height, scrolling=False)
