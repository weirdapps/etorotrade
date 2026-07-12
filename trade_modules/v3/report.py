"""v3 factor report renderer — refined editorial financial style.

``render_report(scores, meta) -> str`` returns a full standalone HTML
document meant to be read in a browser: a luxury research-note layout with
Fraunces / IBM Plex typography, a light editorial palette, responsive
CSS-grid cards, diverging factor z-bars and conviction meters.

``compute_regime(index_close) -> (label, detail)`` is a pure helper the
driver script feeds with a fetched index series; kept network-free here.
"""

from __future__ import annotations

import html as _html

import numpy as np
import pandas as pd

# --- editorial palette (light; the user's) ---------------------------------
INK = "#16181d"  # primary ink
INK2 = "#4a4e57"  # secondary ink
MUTED = "#9aa0a8"  # muted labels
LINE = "#e7e6e2"  # hairline
WARM = "#faf9f7"  # whisper-warm section tone
ACCENT = "#123b3a"  # deep ink-teal structural accent
BULL = "#2d6a4f"
BULL_BAR = "#bfe3cd"
BEAR = "#b3402f"
BEAR_BAR = "#f2c4bb"
TRACK = "#efeeea"  # neutral z-bar track

# Bar scaling: metric z clamps at ±2.5; conviction meter uses a robust
# per-frame scale (90th pct of |conviction|) so typical names read strongly
# and rare outliers simply saturate. Clamped to a sane [floor, cap] band.
_Z_METRIC = 2.5
_CONV_FLOOR = 1.75
_CONV_CAP = 3.0

# Display grouping of metrics: (group label, [(feature_key, metric label), ...]).
# A z-score column ``{key}_z`` is shown when present; else the raw value only.
DISPLAY_GROUPS = [
    (
        "Value",
        [
            ("pe_trailing", "P/E TTM"),
            ("pe_forward", "P/E Fwd"),
            ("ps_sector", "P/S"),
            ("pb", "P/B"),
            ("ev_ebitda", "EV/EBITDA"),
            ("peg", "PEG"),
        ],
    ),
    (
        "Quality",
        [
            ("roe", "ROE"),
            ("roa", "ROA"),
            ("gross_margin", "Gross Mgn"),
            ("op_margin", "Op Mgn"),
            ("fcf", "FCF Yld"),
            ("current_ratio", "Current Ratio"),
            ("de", "Debt/Eq"),
        ],
    ),
    (
        "Momentum",
        [
            ("mom_12_1", "12-1 Mom"),
            ("price_perf", "Price Perf"),
            ("pct_52w_high", "% 52w High"),
        ],
    ),
    ("Low-vol", [("beta", "Beta"), ("realized_vol", "Realized Vol")]),
    ("Strength", [("short_interest", "Short Int"), ("target_dispersion", "Target Disp")]),
    (
        "Analyst",
        [
            ("analyst_mom", "Analyst Mom"),
            ("upside", "Upside"),
            ("buy_pct", "Buy %"),
            ("target_high", "Tgt High"),
            ("target_low", "Tgt Low"),
        ],
    ),
    ("Size-Liq", [("cap", "Mkt Cap"), ("avg_volume", "Avg Vol"), ("adv_usd", "ADV")]),
    ("Income", [("div_yield", "Div Yield")]),
]

# Group label -> the cluster z column shown beside the group header (when present).
GROUP_CLUSTER = {
    "Value": "value_z",
    "Quality": "quality_z",
    "Momentum": "momentum_z",
    "Low-vol": "lowvol_z",
    "Strength": "strength_z",
}

# Formatting hints.
_PCT_FRACTION = {
    "roa",
    "gross_margin",
    "op_margin",
    "mom_12_1",
    "realized_vol",
    "target_dispersion",
}
_PCT_NUMBER = {"upside", "buy_pct", "fcf", "div_yield"}
_MONEY = {"price", "target_high", "target_low"}
_HUMAN = {"cap", "adv_usd", "avg_volume"}


# --- small helpers ----------------------------------------------------------
def _esc(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return _html.escape(str(v))


def _isnan(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)
    except (TypeError, ValueError):
        return False


def _human_money(v: float) -> str:
    a = abs(v)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{v / div:.1f}{suf}"
    return f"{v:.0f}"


def _fmt(key: str, v) -> str:
    if _isnan(v):
        return "–"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return _esc(v)
    if key in _MONEY:
        return f"${v:,.2f}"
    if key in _HUMAN:
        prefix = "" if key == "avg_volume" else "$"
        return prefix + _human_money(v)
    if key in _PCT_FRACTION:
        return f"{v * 100:.1f}%"
    if key in _PCT_NUMBER:
        return f"{v:.1f}%"
    return f"{v:.2f}"


def _z_color(z) -> str:
    if _isnan(z):
        return MUTED
    return BULL if z > 0 else (BEAR if z < 0 else MUTED)


def _znum(z) -> str:
    return "·" if _isnan(z) else f"{z:+.2f}"


def _tilt(conv: float, is_portfolio: bool) -> tuple[str, str]:
    """Return (label, css_class) for the tilt badge."""
    if _isnan(conv):
        return ("n/a", "tilt-flat")
    if is_portfolio:
        if conv > 0.5:
            return ("ADD", "tilt-bull")
        if conv < -0.5:
            return ("TRIM", "tilt-bear")
        return ("HOLD", "tilt-flat")
    if conv > 0.5:
        return ("BUY-WATCH", "tilt-bull")
    if conv < -0.5:
        return ("PASS", "tilt-bear")
    return ("WATCH", "tilt-flat")


def _conv_scale(scores: pd.DataFrame) -> float:
    """Robust symmetric scale for conviction meters.

    Uses the 90th percentile of |conviction| (clamped to [floor, cap]) so the
    bulk of names fill a meaningful portion of the meter and a lone extreme
    (its numeric value still shown) saturates rather than crushing everything.
    """
    c = pd.to_numeric(scores.get("conviction", pd.Series(dtype=float)), errors="coerce").abs()
    c = c.dropna()
    if c.empty:
        return _CONV_FLOOR
    p90 = float(np.nanpercentile(c, 90))
    return min(_CONV_CAP, max(_CONV_FLOOR, p90))


def _bar(z, zmax: float, cls: str) -> str:
    """A diverging horizontal bar centered at 0 (green right / red left).

    ``cls`` is the CSS base class (``zbar`` or ``meter``); the fill spans up
    to half the track (50%) at ``|z| >= zmax``. NaN renders the bare track.
    """
    if _isnan(z) or zmax <= 0:
        return f'<span class="{cls}"></span>'
    frac = min(abs(float(z)) / zmax, 1.0)
    w = frac * 50.0
    side = "pos" if z >= 0 else "neg"
    return f'<span class="{cls}"><span class="{cls}-fill {cls}-{side}" style="width:{w:.1f}%;"></span></span>'


# --- regime (pure; script supplies the fetched index series) ----------------
def compute_regime(index_close: pd.Series) -> tuple[str, str]:
    """Classify the market regime from an index close series.

    RISK_ON  = above 200dma and realized-vol percentile < 60
    RISK_OFF = below 200dma and realized-vol percentile > 60
    else NEUTRAL. Short/empty series -> NEUTRAL.
    """
    s = pd.to_numeric(pd.Series(index_close), errors="coerce").dropna()
    if len(s) < 200:
        return ("NEUTRAL", "insufficient index history")
    last = float(s.iloc[-1])
    ma200 = float(s.rolling(200).mean().iloc[-1])
    above = last > ma200
    rv = s.pct_change().rolling(21).std().dropna()
    if len(rv) >= 2:
        pctile = float((rv <= rv.iloc[-1]).mean() * 100.0)
    else:
        pctile = 50.0
    if above and pctile < 60:
        label = "RISK_ON"
    elif not above and pctile > 60:
        label = "RISK_OFF"
    else:
        label = "NEUTRAL"
    detail = f"SPX {'above' if above else 'below'} 200dma · realized-vol {pctile:.0f}th pct"
    return (label, detail)


# --- card + section builders ------------------------------------------------
def _unit(row: pd.Series, cols, key: str, label: str) -> str:
    """One metric tile.

    Scored metrics (a ``{key}_z`` column exists) show label + value (mono) +
    a tiny diverging z-bar and numeric z. Reference metrics with no z column
    (Mkt Cap, Avg Vol, ADV, Div Yield, Tgt High/Low) show only label + value,
    so context data reads distinctly from scored factors.
    """
    raw = row.get(key, np.nan) if key in cols else np.nan
    val = _fmt(key, raw)
    zkey = f"{key}_z"
    if zkey not in cols:
        return (
            f'<div class="unit unit-ctx">'
            f'<div class="unit-l">{label}</div>'
            f'<div class="unit-v">{val}</div>'
            f"</div>"
        )
    z = row.get(zkey, np.nan)
    return (
        f'<div class="unit">'
        f'<div class="unit-l">{label}</div>'
        f'<div class="unit-v">{val}</div>'
        f'<div class="unit-z">{_bar(z, _Z_METRIC, "zbar")}'
        f'<span class="znum" style="color:{_z_color(z)};">{_znum(z)}</span></div>'
        f"</div>"
    )


def _group(row: pd.Series, cols, label: str, metrics) -> str:
    cluster_key = GROUP_CLUSTER.get(label)
    cz_html = ""
    if cluster_key:
        cz = row.get(cluster_key, np.nan)
        cz_html = f'<span class="grp-z" style="color:{_z_color(cz)};">{_znum(cz)}</span>'
    units = "".join(_unit(row, cols, k, lbl) for k, lbl in metrics)
    return (
        f'<div class="grp">'
        f'<div class="grp-head"><span class="grp-name">{label}</span>{cz_html}</div>'
        f'<div class="units">{units}</div>'
        f"</div>"
    )


def _card(tkr: str, row: pd.Series, cols, conv_scale: float, delay: float) -> str:
    conv = row.get("conviction", np.nan)
    rank = row.get("rank", np.nan)
    is_port = bool(row.get("is_portfolio", False))
    name = _esc(row.get("name", "")) or "&nbsp;"
    sector = _esc(row.get("sector", "")) or "–"
    price = _fmt("price", row.get("price", np.nan))
    conv_txt = "·" if _isnan(conv) else f"{conv:+.2f}"
    rank_txt = "–" if _isnan(rank) else f"#{int(rank)}"
    tilt_label, tilt_cls = _tilt(conv, is_port)

    groups = "".join(_group(row, cols, gl, ms) for gl, ms in DISPLAY_GROUPS)

    return (
        f'<article class="card" style="animation-delay:{delay:.2f}s;">'
        f'<div class="card-top">'
        f'<div class="id">'
        f'<div class="tkr">{_esc(tkr)}</div>'
        f'<div class="name">{name}</div>'
        f'<div class="id-meta"><span class="chip">{sector}</span>'
        f'<span class="price">{price}</span></div>'
        f"</div>"
        f'<div class="verdict">'
        f'<div class="conv" style="color:{_z_color(conv)};">{conv_txt}</div>'
        f'<div class="conv-l">Conviction · {rank_txt}</div>'
        f'<div class="tilt {tilt_cls}">{tilt_label}</div>'
        f"</div>"
        f"</div>"
        f'<div class="meter-wrap">{_bar(conv, conv_scale, "meter")}</div>'
        f'<div class="groups">{groups}</div>'
        f"</article>"
    )


def _section_head(numeral: str, label: str, sub: str) -> str:
    return (
        f'<div class="section-head">'
        f'<span class="sec-num">{numeral}</span>'
        f'<div class="sec-txt"><div class="sec-title">{label}</div>'
        f'<div class="sec-sub">{sub}</div></div>'
        f"</div>"
    )


def _summary(scores: pd.DataFrame, conv_scale: float) -> str:
    ordered = scores.sort_values("conviction", ascending=False)
    top = ordered.head(5)
    bottom = ordered.tail(5).iloc[::-1]

    def _rows(frame: pd.DataFrame) -> str:
        out = []
        for tkr, r in frame.iterrows():
            conv = r.get("conviction", np.nan)
            rank = r.get("rank", np.nan)
            rank_s = "–" if _isnan(rank) else f"{int(rank)}"
            sector_s = _esc(r.get("sector", "")) or "–"
            out.append(
                f'<div class="sum-row">'
                f'<span class="sum-rank">{rank_s}</span>'
                f'<span class="sum-tkr">{_esc(tkr)}</span>'
                f'<span class="sum-sector">{sector_s}</span>'
                f'<span class="sum-bar">{_bar(conv, conv_scale, "meter")}</span>'
                f'<span class="sum-conv" style="color:{_z_color(conv)};">{_znum(conv)}</span>'
                f"</div>"
            )
        return "".join(out)

    return (
        f'<div class="summary">'
        f'<div class="sum-col">'
        f'<div class="sum-head"><span class="dot dot-bull"></span>Highest conviction</div>'
        f"{_rows(top)}</div>"
        f'<div class="sum-col">'
        f'<div class="sum-head"><span class="dot dot-bear"></span>Lowest conviction</div>'
        f"{_rows(bottom)}</div>"
        f"</div>"
    )


def _cards_section(
    scores: pd.DataFrame, is_port: bool, numeral: str, label: str, sub: str, conv_scale: float
) -> str:
    cols = scores.columns
    subset = scores[scores.get("is_portfolio", pd.Series(False, index=scores.index)) == is_port]
    subset = subset.sort_values("conviction", ascending=False)
    if subset.empty:
        body = '<div class="empty">No names in this group.</div>'
    else:
        cards = []
        for i, (t, r) in enumerate(subset.iterrows()):
            cards.append(_card(t, r, cols, conv_scale, min(i * 0.05, 0.4)))
        body = f'<div class="cards">{"".join(cards)}</div>'
    return _section_head(numeral, label, sub) + body


# --- stylesheet -------------------------------------------------------------
def _stylesheet() -> str:
    return """
:root{
  --canvas:#ffffff;--warm:#faf9f7;--ink:#16181d;--ink2:#4a4e57;--muted:#9aa0a8;
  --line:#e7e6e2;--accent:#123b3a;--track:#ecebe4;
  --bull:#2d6a4f;--bull-bar:#bfe3cd;--bull-fill:#e9f5ee;
  --bear:#b3402f;--bear-bar:#f2c4bb;--bear-fill:#fbeeec;
  --serif:'Fraunces',Georgia,'Times New Roman',serif;
  --sans:'IBM Plex Sans',system-ui,sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,'SFMono-Regular',monospace;
}
*{box-sizing:border-box;}
html{-webkit-text-size-adjust:100%;}
body{margin:0;background:var(--canvas);color:var(--ink2);font-family:var(--sans);
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased;
  text-rendering:optimizeLegibility;}
.wrap{max-width:1180px;margin:0 auto;padding:0 40px 72px;}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums;}
.eyebrow{font-family:var(--sans);font-size:10.5px;font-weight:600;letter-spacing:1.3px;
  text-transform:uppercase;color:var(--muted);}

/* masthead */
.masthead{padding:56px 0 26px;}
.mast-eyebrow{font-size:10.5px;font-weight:600;letter-spacing:1.6px;text-transform:uppercase;
  color:var(--accent);}
.mast-title{font-family:var(--serif);font-optical-sizing:auto;font-weight:400;
  font-size:52px;line-height:1.02;letter-spacing:-.6px;color:var(--ink);margin:12px 0 0;}
.mast-rule{width:60px;height:2px;background:var(--accent);margin:22px 0 20px;}
.standfirst{font-family:var(--serif);font-optical-sizing:auto;font-style:italic;font-weight:400;
  font-size:19px;line-height:1.5;color:var(--ink2);margin:0;max-width:760px;}
.mast-meta{display:flex;flex-wrap:wrap;align-items:center;gap:10px 18px;margin-top:24px;
  padding-top:20px;border-top:1px solid var(--line);}
.regime{display:inline-flex;align-items:center;gap:7px;font-family:var(--sans);font-size:11px;
  font-weight:600;letter-spacing:.6px;text-transform:uppercase;padding:5px 12px;border-radius:100px;
  border:1px solid var(--line);background:var(--warm);color:var(--ink2);}
.regime .dot{width:7px;height:7px;border-radius:50%;background:var(--muted);}
.regime-on{background:var(--bull-fill);border-color:var(--bull-bar);color:var(--bull);}
.regime-on .dot{background:var(--bull);}
.regime-off{background:var(--bear-fill);border-color:var(--bear-bar);color:var(--bear);}
.regime-off .dot{background:var(--bear);}
.mast-detail{font-size:12.5px;color:var(--ink2);}
.mast-stats{margin-left:auto;font-family:var(--mono);font-size:11.5px;color:var(--muted);
  font-variant-numeric:tabular-nums;letter-spacing:.2px;}

/* section headers */
.section-head{display:flex;align-items:baseline;gap:18px;margin:52px 0 22px;
  padding-bottom:14px;border-bottom:1px solid var(--line);}
.sec-num{font-family:var(--serif);font-optical-sizing:auto;font-weight:300;font-size:40px;
  line-height:.8;color:var(--accent);min-width:44px;}
.sec-title{font-family:var(--serif);font-weight:500;font-size:24px;letter-spacing:-.3px;
  color:var(--ink);}
.sec-sub{font-size:12.5px;color:var(--muted);margin-top:3px;}

/* summary band */
.summary{display:grid;grid-template-columns:1fr;gap:18px;background:var(--warm);
  border:1px solid var(--line);border-radius:8px;padding:22px 24px;}
.sum-head{display:flex;align-items:center;gap:8px;font-size:10.5px;font-weight:600;
  letter-spacing:1.3px;text-transform:uppercase;color:var(--ink2);margin-bottom:12px;}
.sum-head .dot{width:8px;height:8px;border-radius:50%;}
.dot-bull{background:var(--bull);}
.dot-bear{background:var(--bear);}
.sum-row{display:grid;grid-template-columns:22px 66px minmax(0,1fr) 112px 52px;align-items:center;
  gap:10px;padding:7px 0;border-top:1px solid var(--line);}
.sum-col .sum-row:first-of-type{border-top:none;}
.sum-rank{font-family:var(--mono);font-size:11px;color:var(--muted);text-align:right;
  font-variant-numeric:tabular-nums;}
.sum-tkr{font-family:var(--serif);font-weight:600;font-size:15px;color:var(--ink);
  letter-spacing:-.2px;}
.sum-sector{font-size:11.5px;color:var(--muted);white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;}
.sum-conv{font-family:var(--mono);font-size:13px;font-weight:500;text-align:right;
  font-variant-numeric:tabular-nums;}

/* cards */
.cards{display:grid;grid-template-columns:1fr;gap:20px;}
.card{background:var(--canvas);border:1px solid var(--line);border-radius:10px;padding:24px 26px;
  box-shadow:0 1px 2px rgba(22,24,29,.04);transition:transform .25s ease,box-shadow .25s ease,
  border-color .25s ease;animation:rise .6s cubic-bezier(.2,.7,.2,1) both;}
.card:hover{transform:translateY(-3px);box-shadow:0 14px 34px rgba(22,24,29,.09);
  border-color:#dad8d2;}
.card-top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;}
.tkr{font-family:var(--serif);font-optical-sizing:auto;font-weight:600;font-size:28px;
  line-height:1;letter-spacing:-.4px;color:var(--ink);}
.name{font-size:13px;color:var(--ink2);margin-top:5px;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;max-width:280px;}
.id-meta{display:flex;align-items:center;gap:10px;margin-top:10px;}
.chip{font-size:9.5px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;
  color:var(--ink2);background:var(--warm);border:1px solid var(--line);border-radius:100px;
  padding:3px 9px;white-space:nowrap;}
.price{font-family:var(--mono);font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums;}
.verdict{text-align:right;flex-shrink:0;}
.conv{font-family:var(--mono);font-weight:500;font-size:32px;line-height:1;letter-spacing:-1px;
  font-variant-numeric:tabular-nums;}
.conv-l{font-size:9.5px;font-weight:600;letter-spacing:1px;text-transform:uppercase;
  color:var(--muted);margin-top:6px;}
.tilt{display:inline-block;margin-top:9px;font-size:10px;font-weight:600;letter-spacing:1px;
  text-transform:uppercase;padding:4px 11px;border-radius:100px;border:1px solid var(--line);}
.tilt-bull{color:var(--bull);background:var(--bull-fill);border-color:var(--bull-bar);}
.tilt-bear{color:var(--bear);background:var(--bear-fill);border-color:var(--bear-bar);}
.tilt-flat{color:var(--ink2);background:var(--warm);}
.meter-wrap{margin:20px 0 4px;}

/* factor groups + metric tiles */
.groups{margin-top:8px;}
.grp{padding:15px 0 3px;border-top:1px solid var(--line);}
.grp-head{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:11px;}
.grp-name{font-size:10px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;
  color:var(--ink);}
.grp-z{font-family:var(--mono);font-size:11.5px;font-weight:500;font-variant-numeric:tabular-nums;}
.units{display:grid;grid-template-columns:repeat(auto-fill,minmax(94px,1fr));gap:14px 16px;
  align-items:start;}
.unit-l{font-size:9px;font-weight:600;letter-spacing:.5px;text-transform:uppercase;
  color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.unit-v{font-family:var(--mono);font-size:13.5px;font-weight:500;color:var(--ink);margin-top:3px;
  font-variant-numeric:tabular-nums;letter-spacing:-.2px;}
.unit-z{display:flex;align-items:center;gap:6px;margin-top:6px;}
.znum{font-family:var(--mono);font-size:10px;font-weight:500;font-variant-numeric:tabular-nums;
  min-width:30px;}

/* diverging bars */
.zbar,.meter{position:relative;display:inline-block;background:var(--track);border-radius:100px;
  vertical-align:middle;overflow:hidden;}
.zbar{width:100%;max-width:52px;height:5px;}
.meter{width:100%;height:8px;}
.zbar::before,.meter::before{content:"";position:absolute;left:calc(50% - .5px);top:0;bottom:0;
  width:1px;background:rgba(22,24,29,.16);z-index:1;}
.zbar-fill,.meter-fill{position:absolute;top:0;bottom:0;z-index:0;}
.zbar-pos,.meter-pos{left:50%;background:var(--bull-bar);}
.zbar-neg,.meter-neg{right:50%;background:var(--bear-bar);}
.meter-pos{background:linear-gradient(90deg,#cdead9,var(--bull));}
.meter-neg{background:linear-gradient(270deg,#f4cec6,var(--bear));}

.empty{color:var(--muted);font-size:13px;padding:12px 0;}

/* footer */
.footer{margin-top:56px;padding-top:24px;border-top:2px solid var(--ink);}
.footer .eyebrow{margin-bottom:10px;}
.method{font-size:11.5px;color:var(--muted);line-height:1.7;max-width:820px;}
.method b{color:var(--ink2);font-weight:600;}
.foot-stamp{margin-top:16px;font-family:var(--mono);font-size:11px;color:var(--muted);
  font-variant-numeric:tabular-nums;}

@keyframes rise{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:none;}}

@media (min-width:1100px){
  .cards{grid-template-columns:1fr 1fr;}
  .summary{grid-template-columns:1fr 1fr;gap:36px;}
}
@media (max-width:680px){
  .wrap{padding:0 20px 56px;}
  .mast-title{font-size:38px;}
  .mast-stats{margin-left:0;width:100%;}
  .card{padding:20px;}
}
@media (prefers-reduced-motion:reduce){
  .card{animation:none;}
  .card:hover{transform:none;}
}
""".strip()


# --- top-level --------------------------------------------------------------
def render_report(scores: pd.DataFrame, meta: dict) -> str:
    meta = meta or {}
    n_port = meta.get("n_portfolio", int(scores.get("is_portfolio", pd.Series(dtype=bool)).sum()))
    n_cand = meta.get("n_candidates", 0)
    date = _esc(meta.get("date", ""))
    regime = _esc(meta.get("regime", "NEUTRAL"))
    regime_detail = _esc(meta.get("regime_detail", ""))
    system_read = _esc(meta.get("system_read", ""))
    priced = _esc(meta.get("priced", ""))
    enriched = _esc(meta.get("enriched", ""))
    generated = _esc(meta.get("generated_utc", ""))

    regime_cls = {"RISK_ON": "regime-on", "RISK_OFF": "regime-off"}.get(
        str(meta.get("regime")), "regime-neutral"
    )
    conv_scale = _conv_scale(scores)

    masthead = (
        f'<header class="masthead">'
        f'<div class="mast-eyebrow">Trading Model v3 · Factor Report</div>'
        f'<h1 class="mast-title">Factor Snapshot</h1>'
        f'<div class="mast-rule"></div>'
        f'<p class="standfirst">{system_read}</p>'
        f'<div class="mast-meta">'
        f'<span class="regime {regime_cls}"><span class="dot"></span>{regime}</span>'
        f'<span class="mast-detail">{regime_detail}</span>'
        f'<span class="mast-stats">{date} · Portfolio {n_port} · Candidates {n_cand} · '
        f"Priced {priced} / Enriched {enriched}</span>"
        f"</div></header>"
    )

    summary = _section_head(
        "I", "Conviction extremes", "Ranked across the full universe · long vs. avoid"
    ) + _summary(scores, conv_scale)
    port_sec = _cards_section(
        scores, True, "II", "Portfolio", "Current holdings · tilt = ADD / HOLD / TRIM", conv_scale
    )
    cand_sec = _cards_section(
        scores,
        False,
        "III",
        "Candidates",
        "Buy-signal watchlist · tilt = BUY-WATCH / WATCH / PASS",
        conv_scale,
    )

    footer = (
        f'<footer class="footer">'
        f'<div class="eyebrow">Methodology</div>'
        f'<div class="method">'
        f"Each metric is winsorized (1/99) and cross-sectionally z-scored; low-is-good "
        f"metrics (valuation, leverage, beta, realized-vol, short interest, target dispersion) "
        f"are negated so a high z is always attractive. Metric z-scores are sector-neutral "
        f"(demeaned within GICS sector). Five cluster z-scores (Value, Quality, Momentum, "
        f"Low-vol, Strength) combine into <b>conviction</b> with a near-equal weighting: Value "
        f"plus Quality jointly capped at ~55%, renormalized per row over the clusters actually "
        f"present. Conviction is re-z-scored cross-sectionally; rank 1 = best. Bars are centered "
        f"at 0: green extends right (attractive), red extends left (unattractive).<br><br>"
        f"This is a <b>shadow / decision-support</b> snapshot, not yet capital-authorized. The "
        f"naive price-spine (12-1 momentum plus low-vol) showed no standalone out-of-sample edge "
        f"in validation, so treat single-factor tilts with caution and weigh the full-cluster "
        f"conviction over any one number."
        f"</div>"
        f'<div class="foot-stamp">Generated {generated} · decision-support, not investment advice.</div>'
        f"</footer>"
    )

    body = f'<div class="wrap">{masthead}{summary}{port_sec}{cand_sec}{footer}</div>'

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Trading Model v3 · Factor Report</title>"
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        "family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,500;0,9..144,600;"
        "1,9..144,400&family=IBM+Plex+Mono:wght@400;500;600&"
        'family=IBM+Plex+Sans:wght@400;500;600&display=swap">'
        f"<style>{_stylesheet()}</style>"
        "</head>"
        f"<body>{body}</body></html>"
    )
