"""v3 factor report renderer — refined editorial financial style.

``render_report(scores, meta) -> str`` returns a full standalone HTML
document meant to be read in a browser: a luxury research-note layout with
Fraunces / IBM Plex typography, a light editorial palette, an at-a-glance
conviction heatmap of the whole scored universe, and curated full-factor
cards (diverging z-bars, conviction meters) for the names that matter.

``compute_regime(index_close) -> (label, detail)`` is a pure helper the
driver script feeds with a fetched index series; kept network-free here.
"""

from __future__ import annotations

import html as _html

import numpy as np
import pandas as pd

# --- editorial palette (Python-side colors used in helper functions) --------
# INK, INK2, LINE, WARM, ACCENT, BULL_BAR, BEAR_BAR, TRACK appear only as
# CSS custom-property literals inside _stylesheet(); they are not referenced
# from Python code and are not defined here.  MUTED / BULL / BEAR are used
# in the Python helpers below (_z_color, _heat_bg, _heat_cell).
MUTED = "#9aa0a8"  # muted labels
BULL = "#2d6a4f"
BEAR = "#b3402f"

# Bar scaling: metric z clamps at ±2.5; conviction meter uses a robust
# per-frame scale (90th pct of |conviction|) so typical names read strongly
# and rare outliers simply saturate. Clamped to a sane [floor, cap] band.
_Z_METRIC = 2.5
_Z_HEAT = 2.0  # cluster-z at which a heatmap cell reaches full tint
_CONV_FLOOR = 1.75
_CONV_CAP = 3.0

# Curation defaults: how many candidate cards to render in detail.
_MAX_LONG_CARDS = 12
_MAX_AVOID_CARDS = 6

# The five scoring clusters, shown as the overview heatmap's heat-strip.
CLUSTER_CELLS = [
    ("value_z", "Value"),
    ("quality_z", "Quality"),
    ("momentum_z", "Mom"),
    ("lowvol_z", "Low-vol"),
    ("strength_z", "Strength"),
]

# Six cluster z-scores used in the collapsed card summary strip (all six clusters
# including Growth, which the combiner computes even when its weight is 0).
_CARD_CLUSTERS = [
    ("value_z", "Value"),
    ("quality_z", "Quality"),
    ("momentum_z", "Momentum"),
    ("growth_z", "Growth"),
    ("lowvol_z", "Low-vol"),
    ("strength_z", "Strength"),
]

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


def _heat_bg(z) -> str:
    """RGBA tint for a heatmap cell: bull-green (pos) / bear-red (neg), the
    opacity ramping with |z| (clamped at ``_Z_HEAT``). Same two hues as the
    rest of the palette, so near-zero cells recede toward the paper."""
    a = min(abs(float(z)) / _Z_HEAT, 1.0)
    alpha = 0.10 + a * 0.50
    rgb = "45,106,79" if z >= 0 else "179,64,47"
    return f"rgba({rgb},{alpha:.2f})"


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


# --- Phase 5D formatters (never emit the literal "None") --------------------
_AMBER = "#96601a"  # warm ochre for TRIM / soft-warn (stays within the light palette)


def _pct1(v) -> str:
    """A fraction as a 1-decimal percent (0.182 -> "18.2%"); NaN/None -> "n/a"."""
    return "n/a" if _isnan(v) else f"{float(v) * 100:.1f}%"


def _pct0(v) -> str:
    """A fraction as a whole percent (0.9 -> "90%"); NaN/None -> "n/a"."""
    return "n/a" if _isnan(v) else f"{float(v) * 100:.0f}%"


def _pp(v) -> str:
    """A fractional delta as signed percentage points (0.023 -> "+2.3 pp")."""
    return "n/a" if _isnan(v) else f"{float(v) * 100:+.1f} pp"


def _num1(v) -> str:
    return "n/a" if _isnan(v) else f"{float(v):.1f}"


def _num2(v) -> str:
    return "n/a" if _isnan(v) else f"{float(v):.2f}"


def _money(v) -> str:
    return "n/a" if _isnan(v) else f"${float(v):,.2f}"


def _usd_signed(v) -> str:
    """Signed whole-dollar delta ("+$1,234" / "-$980"); NaN/None -> "" (omitted)."""
    if _isnan(v):
        return ""
    v = float(v)
    return f"{'+' if v >= 0 else '-'}${abs(v):,.0f}"


# Suggested-action groups: (ACTION, css-suffix, one-line hint). Rendered in this
# order; BUY/ADD carry entry/SL/TP tiles.
_ACTION_GROUPS = [
    ("BUY", "buy", "Open new position"),
    ("ADD", "add", "Increase weight"),
    ("TRIM", "trim", "Reduce weight"),
    ("SELL", "sell", "Close position"),
    ("HOLD", "hold", "No change"),
]
_LEVEL_ACTIONS = {"BUY", "ADD"}

# Binding-flag / gate-lever -> short human chip label.
_FLAG_LABELS = {
    "cvar": "CVaR over budget",
    "net_beta": "net-beta out of band",
    "effective_bets": "low breadth",
    "deployment_capped": "deployment cap-limited",
    "name_cap": "name cap",
    "sector_cap": "sector cap",
    "usd_bloc_cap": "USD-bloc cap",
    "vol_over_target": "vol over target",
}
_LEVER_LABELS = {
    "caps": "gate: caps",
    "tail_deweight": "gate: tail de-weight",
    "gross_cut": "gate: gross cut",
}

# Section numerals; render_report assigns them left-to-right so the legacy
# Overview / Portfolio / Candidates stay I / II / III when no new sections lead.
_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]


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
    rv = s.pct_change(fill_method=None).rolling(21).std().dropna()
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


# --- overview heatmap -------------------------------------------------------
def _heat_cell(label: str, z) -> str:
    """One cluster heat-cell: color-only, |z|-scaled tint; value on hover."""
    if _isnan(z):
        return f'<span class="hm-cell nan" title="{label}: no data">·</span>'
    return (
        f'<span class="hm-cell" style="background:{_heat_bg(z)};" '
        f'title="{label}: {float(z):+.2f}"></span>'
    )


def _overview(scores: pd.DataFrame, conv_scale: float) -> str:
    """A scannable, ranked table of the FULL scored universe.

    rank · ticker · sector · conviction (numeric + diverging bar) · a compact
    strip of the five cluster z's as diverging heat-cells. Portfolio names
    carry a small accent dot. This is the report's centerpiece.
    """
    ordered = scores.sort_values("conviction", ascending=False, na_position="last")

    head = (
        '<div class="hm-head">'
        "<span></span>"
        '<span class="hm-h right">#</span>'
        '<span class="hm-h">Ticker</span>'
        '<span class="hm-h">Sector</span>'
        '<span class="hm-h hm-name">Name</span>'
        '<span class="hm-h right">Conv</span>'
        '<span class="hm-barcell"></span>'
        '<div class="hm-cells">'
        + "".join(f'<span class="hm-h center">{lbl}</span>' for _, lbl in CLUSTER_CELLS)
        + "</div></div>"
    )

    rows = []
    for tkr, r in ordered.iterrows():
        conv = r.get("conviction", np.nan)
        rank = r.get("rank", np.nan)
        is_port = bool(r.get("is_portfolio", False))
        dot = (
            '<span class="hm-dot pf" title="Portfolio holding"></span>'
            if is_port
            else '<span class="hm-dot"></span>'
        )
        rank_s = "–" if _isnan(rank) else str(int(rank))
        sector_s = _esc(r.get("sector", "")) or "–"
        name_s = _esc(r.get("name", ""))
        conv_s = "·" if _isnan(conv) else f"{conv:+.2f}"
        cells = "".join(_heat_cell(lbl, r.get(k, np.nan)) for k, lbl in CLUSTER_CELLS)
        rows.append(
            '<div class="hm-row">'
            f"{dot}"
            f'<span class="hm-rank">{rank_s}</span>'
            f'<span class="hm-tkr">{_esc(tkr)}</span>'
            f'<span class="hm-sector">{sector_s}</span>'
            f'<span class="hm-name">{name_s}</span>'
            f'<span class="hm-conv" style="color:{_z_color(conv)};">{conv_s}</span>'
            f'<span class="hm-barcell">{_bar(conv, conv_scale, "meter")}</span>'
            f'<div class="hm-cells">{cells}</div>'
            "</div>"
        )

    return f'<div class="hm">{head}<div class="hm-body">{"".join(rows)}</div></div>'


# --- card + section builders ------------------------------------------------
def _unit(row: pd.Series, cols, key: str, label: str) -> str:
    """One metric row inside a factor group.

    Scored metrics (a ``{key}_z`` column exists) show label + value (mono) on
    one line, then a full-width diverging z-bar and numeric z beneath. Reference
    metrics with no z column (Mkt Cap, Avg Vol, ADV, Div Yield, Tgt High/Low)
    show only label + value, so context data reads distinctly from scored ones.
    """
    raw = row.get(key, np.nan) if key in cols else np.nan
    val = _fmt(key, raw)
    top = (
        f'<div class="unit-row"><span class="unit-l">{label}</span>'
        f'<span class="unit-v">{val}</span></div>'
    )
    zkey = f"{key}_z"
    if zkey not in cols:
        return f'<div class="unit unit-ctx">{top}</div>'
    z = row.get(zkey, np.nan)
    return (
        f'<div class="unit">{top}'
        f'<div class="unit-z">{_bar(z, _Z_METRIC, "zbar")}'
        f'<span class="znum" style="color:{_z_color(z)};">{_znum(z)}</span></div>'
        f"</div>"
    )


def _group(row: pd.Series, cols, label: str, metrics) -> str:
    """One factor group: header (label + cluster z) over its stacked metrics.

    Rendered as a single cell of the card's responsive group grid.
    """
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


def _cluster_strip(row) -> str:
    """Compact six-cluster z-score row for the always-visible card header.

    Each chip: label + a small magnitude bar (width proportional to |z|, capped
    at 2.5) + the signed z value.  Green for positive z, red for negative.
    ``row`` may be None (ghost rows not in the scores frame); every chip then
    renders as the neutral "·" placeholder.
    """
    chips = []
    for key, label in _CARD_CLUSTERS:
        z = row.get(key, np.nan) if row is not None else np.nan
        if _isnan(z):
            chips.append(
                f'<span class="cz-chip cz-nan">'
                f'<span class="cz-label">{label}</span>'
                f'<span class="cz-bar"></span>'
                f'<span class="cz-val">&#xB7;</span>'
                f"</span>"
            )
        else:
            z = float(z)
            pct = min(abs(z) / 2.5, 1.0) * 100.0
            col = BULL if z > 0 else BEAR
            chips.append(
                f'<span class="cz-chip">'
                f'<span class="cz-label">{label}</span>'
                f'<span class="cz-bar">'
                f'<span class="cz-fill" style="width:{pct:.0f}%;background:{col};"></span>'
                f"</span>"
                f'<span class="cz-val" style="color:{col};">{z:+.1f}</span>'
                f"</span>"
            )
    return f'<div class="cluster-strip">{"".join(chips)}</div>'


def _levels_block(row: pd.Series) -> str:
    """Trade Levels tiles: Entry / Stop / Target / R:R (vol-scaled).

    Values come from the ``entry``/``stop_loss``/``take_profit``/``rr`` feature
    columns. Entry falls back to price when present; degenerate levels (missing
    vol) render "n/a" rather than a dash.
    """
    entry = row.get("entry", np.nan)
    if _isnan(entry):
        entry = row.get("price", np.nan)
    stop = row.get("stop_loss", np.nan)
    target = row.get("take_profit", np.nan)
    rr = row.get("rr", np.nan)

    def _px(v) -> str:
        return "n/a" if _isnan(v) else f"${float(v):,.2f}"

    rr_s = "n/a" if _isnan(rr) else f"{float(rr):.2f}"
    tiles = (
        f'<div class="lvl"><div class="lvl-l">Entry</div><div class="lvl-v">{_px(entry)}</div></div>'
        f'<div class="lvl lvl-stop"><div class="lvl-l">Stop</div>'
        f'<div class="lvl-v">{_px(stop)}</div></div>'
        f'<div class="lvl lvl-tgt"><div class="lvl-l">Target</div>'
        f'<div class="lvl-v">{_px(target)}</div></div>'
        f'<div class="lvl"><div class="lvl-l">R:R</div><div class="lvl-v">{rr_s}</div></div>'
    )
    return (
        f'<div class="levels">'
        f'<div class="levels-head">Trade Levels</div>'
        f'<div class="levels-tiles">{tiles}</div>'
        f'<div class="levels-note">Vol-scaled: monthly ~2σ stop / 3σ target.</div>'
        f"</div>"
    )


def _card(tkr: str, row: pd.Series, cols, conv_scale: float, delay: float) -> str:
    """One full-width card: a balanced top band over a responsive factor grid.

    Top band, left: ticker + name + sector chip. Right: price, conviction
    figure + diverging meter, tilt badge, and the Trade Levels tiles. Body: the
    eight factor groups as a container-query grid that reflows 4 / 2 / 1 columns
    with the card's own width.
    """
    conv = row.get("conviction", np.nan)
    rank = row.get("rank", np.nan)
    is_port = bool(row.get("is_portfolio", False))
    name = _esc(row.get("name", "")) or "&nbsp;"
    sector = _esc(row.get("sector", "")) or "n/a"
    desc_raw = row.get("description", "")
    desc_text = (
        ""
        if (desc_raw is None or (isinstance(desc_raw, float) and pd.isna(desc_raw)))
        else str(desc_raw).strip()
    )
    desc_html = f'<div class="desc">{_esc(desc_text)}</div>' if desc_text else ""
    price = _fmt("price", row.get("price", np.nan))
    conv_txt = "·" if _isnan(conv) else f"{conv:+.2f}"
    rank_txt = "n/a" if _isnan(rank) else f"#{int(rank)}"
    tilt_label, tilt_cls = _tilt(conv, is_port)
    pf_tag = '<span class="pf-tag" title="Portfolio holding">PF</span>' if is_port else ""

    groups = "".join(_group(row, cols, gl, ms) for gl, ms in DISPLAY_GROUPS)

    return (
        f'<article class="card" style="animation-delay:{delay:.2f}s;">'
        f'<div class="card-top">'
        f'<div class="id">'
        f'<div class="tkr">{_esc(tkr)}{pf_tag}</div>'
        f'<div class="name">{name}</div>'
        f"{desc_html}"
        f'<div class="id-meta"><span class="chip">{sector}</span></div>'
        f"</div>"
        f'<div class="verdict-top">'
        f'<div class="price-block"><div class="vk">Price</div>'
        f'<div class="price">{price}</div></div>'
        f'<div class="conv-block">'
        f'<div class="conv" style="color:{_z_color(conv)};">{conv_txt}</div>'
        f'<div class="conv-l">Conviction · {rank_txt}</div>'
        f'<div class="tilt {tilt_cls}">{tilt_label}</div>'
        f"</div>"
        f"</div>"
        f"</div>"
        f'<div class="card-strip">'
        f'<div class="meter-col"><div class="vk">Conviction vs peers</div>'
        f'<div class="meter-wrap">{_bar(conv, conv_scale, "meter")}</div></div>'
        f"{_levels_block(row)}"
        f"</div>"
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


# --- Phase 5D: executive summary line + exec/risk panel + suggested actions --
def _exec_summary_line(meta: dict, portfolio, actions, conditioning) -> str:
    """A single scannable line at the very top: regime, deployment, BUY/SELL, vol.

    Renders whatever is available; omits any item whose source (portfolio /
    actions / conditioning) was not supplied.
    """
    cond = conditioning or {}
    items: list[str] = []
    regime = str(meta.get("regime") or cond.get("regime") or "").strip()
    if regime:
        items.append(f"{_esc(regime.upper())} regime")
    if portfolio is not None:
        items.append(
            f"{_pct0(portfolio.get('gross'))} deployed / {_pct0(portfolio.get('cash'))} cash"
        )
    elif cond.get("final_deployment") is not None:
        items.append(f"{_pct0(cond.get('final_deployment'))} deployment")
    if actions is not None:
        n_buy = sum(1 for a in actions if a.get("action") == "BUY")
        n_sell = sum(1 for a in actions if a.get("action") == "SELL")
        items.append(f"{n_buy} BUY / {n_sell} SELL")
    if portfolio is not None:
        gate = (portfolio.get("diagnostics") or {}).get("gate") or {}
        vol, ceil = gate.get("vol_after"), gate.get("vol_ceiling")
        if not _isnan(vol):
            tail = f" (ceiling {_pct0(ceil)})" if not _isnan(ceil) else ""
            items.append(f"vol {_pct1(vol)}{tail}")
    if not items:
        return ""
    inner = '<span class="es-sep">·</span>'.join(f'<span class="es-item">{i}</span>' for i in items)
    return f'<div class="exec-summary">{inner}</div>'


def _stat(label: str, value: str, sub: str = "", tone: str = "") -> str:
    """One executive stat tile: label over a big value with an optional sub-note.

    ``tone`` in {"", "bull", "bear", "warn"} colors the value within the palette.
    """
    tone_cls = f" stat--{tone}" if tone else ""
    sub_html = f'<div class="stat-sub">{_esc(sub)}</div>' if sub else ""
    return (
        f'<div class="stat{tone_cls}">'
        f'<div class="stat-l">{_esc(label)}</div>'
        f'<div class="stat-v">{_esc(value)}</div>{sub_html}</div>'
    )


def _social_grid(social: dict) -> str:
    """PI performance + social tiles (the eToro account panel) for Section I.

    Reads the ``social`` block from the live-account snapshot. Gains and win
    ratio arrive already as percentages (3.4 = 3.4%), so they are formatted
    directly, not through _pct1 (which assumes fractions). Returns "" when the
    block is absent so the report degrades gracefully.
    """
    if not social:
        return ""

    def _g(v) -> str:  # signed 1-dp percent for a value already in percent units
        return "n/a" if _isnan(v) else f"{float(v):+.1f}%"

    def _tone(v) -> str:
        return "" if _isnan(v) else ("bull" if float(v) >= 0 else "bear")

    copiers, baseline = social.get("copiers"), social.get("baseline_copiers")
    cop_val = "n/a" if _isnan(copiers) else f"{int(copiers):,}"
    cop_sub = ""
    if not _isnan(copiers) and not _isnan(baseline):
        cop_sub = f"{int(copiers) - int(baseline):+d} vs last wk"
    risk = social.get("risk_score")
    risk_val = "n/a" if _isnan(risk) else f"{int(risk)}/10"
    win = social.get("win_ratio")
    win_val = "n/a" if _isnan(win) else f"{float(win):.1f}%"
    trades = social.get("trades_ytd")
    win_sub = "" if _isnan(trades) else f"{int(trades)} trades"
    ua, shorts = social.get("unique_assets"), social.get("shorts")
    ua_val = "n/a" if _isnan(ua) else f"{int(ua)}"
    ua_sub = "" if _isnan(shorts) else f"{int(shorts)} short"
    op = social.get("open_positions")
    op_val = "n/a" if _isnan(op) else f"{int(op)}"

    tiles = [
        _stat(
            "Today", _g(social.get("daily_gain")), "eToro daily", _tone(social.get("daily_gain"))
        ),
        _stat("Month to date", _g(social.get("gain_mtd")), "MTD", _tone(social.get("gain_mtd"))),
        _stat(
            "Year to date",
            _g(social.get("gain_ytd")),
            "eToro CurrYear",
            _tone(social.get("gain_ytd")),
        ),
        _stat("Copiers", cop_val, cop_sub),
        _stat("Risk score", risk_val, "eToro 1-10"),
        _stat("Win ratio", win_val, win_sub),
        _stat("Unique assets", ua_val, ua_sub),
        _stat("Open positions", op_val, "incl. lots"),
    ]
    return f'<div class="stat-grid social-grid">{"".join(tiles)}</div>'


def _exec_panel(portfolio: dict, conditioning, meta: dict) -> str:
    """The executive / risk panel: deployment + risk stat tiles, sector exposures
    and any binding cap / gate flags. Reads the build_portfolio result dict
    (``diagnostics`` incl. ``diagnostics['gate']``) and the resolve_deployment
    dial diagnostics. Degenerate inputs (missing gate) render "n/a" gracefully.
    """
    cond = conditioning or {}
    diag = portfolio.get("diagnostics") or {}
    gate = diag.get("gate") or {}
    binding = diag.get("binding") or {}

    gross = portfolio.get("gross")
    cash = portfolio.get("cash")
    usd_bloc = portfolio.get("usd_bloc")
    sector_exp = portfolio.get("sector_exposures") or {}

    regime = str(meta.get("regime") or cond.get("regime") or "").strip().upper() or "n/a"
    final_dep = cond.get("final_deployment")
    base_dep = cond.get("base_deployment")

    vol_after = gate.get("vol_after")
    vol_ceiling = gate.get("vol_ceiling")
    net_beta = gate.get("net_beta", diag.get("net_beta"))
    band = gate.get("net_beta_band") or diag.get("net_beta_band") or (0.3, 1.1)
    net_beta_out = gate.get("net_beta_out")
    eff = gate.get("effective_bets", diag.get("effective_bets"))
    min_eff = gate.get("min_effective_bets")
    caps_ok = gate.get("caps_ok")
    cvar_rb = diag.get("cvar_95_risk_book")
    # Prefer the post-gate value when a lever fired (gate["cvar_after"] reflects the
    # gated vol, keeping this tile consistent with the "Portfolio vol" tile which also
    # reads the post-gate gate["vol_after"]). Falls back to the pre-gate
    # diag["cvar_95_deployed"] when no gate dict is present.
    cvar_dep = (
        gate.get("cvar_after")
        if gate.get("cvar_after") is not None
        else diag.get("cvar_95_deployed")
    )

    lo, hi = float(band[0]), float(band[1])
    vol_tone = (
        ""
        if (_isnan(vol_after) or _isnan(vol_ceiling))
        else ("bull" if float(vol_after) <= float(vol_ceiling) + 1e-9 else "bear")
    )
    if _isnan(net_beta):
        beta_tone = ""
    elif net_beta_out is True or not (lo <= float(net_beta) <= hi):
        beta_tone = "warn"
    else:
        beta_tone = "bull"
    eff_tone = (
        ""
        if _isnan(eff)
        else ("warn" if (not _isnan(min_eff) and float(eff) < float(min_eff)) else "bull")
    )
    # Concentration caps: OK/BREACH + the BINDING axis (highest utilization vs its
    # own cap), NOT always USD-bloc. Previously the sub always showed USD-bloc, so a
    # name/sector breach read as "22% breached". Limits come from meta["caps"].
    _caps_cfg = meta.get("caps") or {}
    max_region = diag.get("max_region")
    _axes: list = []  # (label, value, cap_limit)
    if not _isnan(gate.get("max_name")) and _caps_cfg.get("name"):
        _axes.append(("name", float(gate["max_name"]), float(_caps_cfg["name"])))
    if not _isnan(gate.get("max_sector")) and _caps_cfg.get("sector"):
        _axes.append(("sector", float(gate["max_sector"]), float(_caps_cfg["sector"])))
    # usd_bloc is NAV-basis (absolute); convert to book-basis (/gross) so it ranks on
    # the same basis as name/sector (gate, of book) and region (/gross) — I2 fix.
    _usd_book = (
        float(usd_bloc) / float(gross)
        if (not _isnan(usd_bloc) and not _isnan(gross) and float(gross) > 0)
        else usd_bloc
    )
    if not _isnan(_usd_book) and _caps_cfg.get("usd_bloc"):
        _axes.append(("USD-bloc", float(_usd_book), float(_caps_cfg["usd_bloc"])))
    if not _isnan(max_region) and _caps_cfg.get("region"):
        _axes.append(("region", float(max_region), float(_caps_cfg["region"])))
    if _axes:
        # 0.5pp tolerance = half a display unit (whole-percent rounding). A name at
        # 10.03% vs a 10% cap displays as "10% of 10%" and must NOT read as BREACH;
        # only a visible overage (e.g. 11% of 10%) flags. Absorbs gate slack + noise.
        _breach = any(v > c + 0.005 for _, v, c in _axes)
        _bind = max(_axes, key=lambda a: (a[1] / a[2]) if a[2] > 0 else 0.0)
        caps_val = "BREACH" if _breach else "OK"
        caps_tone = "bear" if _breach else "bull"
        caps_sub = f"{_bind[0]} {_pct0(_bind[1])} of {_pct0(_bind[2])}"
    else:  # no cap config -> fall back to the gate boolean + USD-bloc
        caps_val = "n/a" if caps_ok is None else ("OK" if caps_ok else "BREACH")
        caps_tone = "" if caps_ok is None else ("bull" if caps_ok else "bear")
        caps_sub = f"USD-bloc {_pct0(usd_bloc)}"

    regime_sub = ""
    if not _isnan(final_dep):
        regime_sub = f"target {_pct0(final_dep)}"
        if not _isnan(base_dep) and abs(float(base_dep) - float(final_dep)) > 1e-9:
            regime_sub = f"base {_pct0(base_dep)} to {_pct0(final_dep)}"

    tiles = [
        _stat("Regime", regime, regime_sub),
        _stat("Deployment", _pct0(gross), f"{_pct0(cash)} cash"),
        _stat("Portfolio vol", _pct1(vol_after), f"ceiling {_pct0(vol_ceiling)}", vol_tone),
        _stat("Net beta", _num2(net_beta), f"band {lo:.1f} to {hi:.1f}", beta_tone),
        _stat("CVaR-95 risk book", _pct1(cvar_rb), "fully invested"),
        _stat("CVaR-95 deployed", _pct1(cvar_dep), "at deployment"),
        _stat(
            "Effective bets",
            _num1(eff),
            ("min " + _num1(min_eff)) if not _isnan(min_eff) else "diversification",
            eff_tone,
        ),
        _stat("Concentration caps", caps_val, caps_sub, caps_tone),
    ]
    grid = f'<div class="stat-grid">{"".join(tiles)}</div>'

    sec_items = sorted(
        ((str(k), float(v)) for k, v in sector_exp.items() if float(v) > 1e-6),
        key=lambda kv: -kv[1],
    )[:5]
    sec_html = ""
    if sec_items:
        chips = "".join(
            f'<span class="sec-chip"><b>{_esc(k.title())}</b> {_pct1(v)}</span>'
            for k, v in sec_items
        )
        sec_html = f'<div class="exec-sub"><span class="exec-sub-l">Top sectors</span>{chips}</div>'

    flags = [_FLAG_LABELS[k] for k, on in binding.items() if on and k in _FLAG_LABELS]
    for lev in gate.get("levers_fired") or []:
        if lev in _LEVER_LABELS:
            flags.append(_LEVER_LABELS[lev])
    if gate.get("gross_cut") and _LEVER_LABELS["gross_cut"] not in flags:
        flags.append(_LEVER_LABELS["gross_cut"])
    # Polymarket dial: surface the signal when present. Advisory (no book effect)
    # while the tilt is 0 (shadow phase); shows the applied pp when live.
    pm_sig = cond.get("polymarket_signal")
    if pm_sig is not None:
        pm_tilt = float(cond.get("polymarket_tilt") or 0.0)
        if cond.get("polymarket_active") and abs(pm_tilt) > 1e-9:
            flags.append(f"PM {float(pm_sig):+.2f} to {pm_tilt * 100:+.0f}pp")
        else:
            flags.append(f"PM {float(pm_sig):+.2f} advisory")
    seen: set = set()
    flags = [f for f in flags if not (f in seen or seen.add(f))]
    if flags:
        chips = "".join(
            f'<span class="flag-chip{" lever" if str(f).startswith("gate:") else ""}">'
            f"{_esc(f)}</span>"
            for f in flags
        )
        flag_html = f'<div class="exec-flags"><span class="exec-sub-l">Flags</span>{chips}</div>'
    else:
        flag_html = (
            '<div class="exec-flags"><span class="exec-sub-l">Flags</span>'
            '<span class="flag-none">none binding</span></div>'
        )

    acct = meta.get("account") or {}
    account_html = ""
    if acct:
        pnl_usd = acct.get("unrealized_pnl")
        pnl_pct = acct.get("profit_pct")
        pnl_s = _usd_signed(pnl_usd) or "n/a"
        pnl_sub = f"{float(pnl_pct):+.1f}%" if not _isnan(pnl_pct) else ""
        pnl_tone = ("bull" if float(pnl_usd) >= 0 else "bear") if not _isnan(pnl_usd) else ""
        account_html = (
            '<div class="stat-grid acct-grid">'
            + _stat("Total Value", _money(acct.get("total_equity")), "portfolio NAV")
            + _stat("Unrealized P&L", pnl_s, pnl_sub, pnl_tone)
            + _stat("Cash", _money(acct.get("available")), "available")
            + _stat("Invested", _money(acct.get("invested_cost")), "at cost")
            + "</div>"
        )
    social_html = _social_grid(meta.get("social") or {})
    return f'<div class="exec-panel">{account_html}{social_html}{grid}{sec_html}{flag_html}</div>'


def _action_row(a: dict) -> str:
    """One suggested-action row: id + conviction + current→target(+delta,$) + levels."""
    tkr = _esc(a.get("ticker") or "")
    name = _esc(a.get("name") or "")
    conv = a.get("conviction")
    conv_s = "·" if _isnan(conv) else f"{conv:+.2f}"
    cur, tgt, delta = a.get("current_pct"), a.get("target_pct"), a.get("delta_pct")
    usd = _usd_signed(a.get("delta_usd"))
    usd_html = f'<span class="act-usd">{usd}</span>' if usd else ""
    name_html = f'<div class="act-name">{name}</div>' if name else ""
    pnl = a.get("pnl")
    pnl_html = ""
    if pnl is not None:
        pct, cv = a.get("pnl_pct"), a.get("current_value")
        pcol = "#2d6a4f" if float(pnl) >= 0 else "#b3402f"
        pct_s = f" ({pct:+.1f}%)" if pct is not None else ""
        val_s = f" · {_money(cv)}" if cv is not None else ""
        pnl_html = (
            f'<span class="act-pnl" style="color:{pcol};">'
            f"P/L {_usd_signed(pnl)}{pct_s}{val_s}</span>"
        )
    move = (
        f'<span class="from">{_pct1(cur)}</span> '
        f'<span class="to">→ {_pct1(tgt)}</span> '
        f'<span class="act-dpp">({_pp(delta)})</span>{usd_html}{pnl_html}'
    )
    levels_html = ""
    if a.get("action") in _LEVEL_ACTIONS:
        levels_html = (
            f'<span class="act-lvl">E {_money(a.get("price"))}</span>'
            f'<span class="act-lvl sl">SL {_money(a.get("stop_loss"))}</span>'
            f'<span class="act-lvl tp">TP {_money(a.get("take_profit"))}</span>'
        )
    return (
        f'<div class="act-row">'
        f'<div class="act-id"><div class="act-tkr">{tkr}</div>{name_html}</div>'
        f'<div class="act-conv" style="color:{_z_color(conv)};">{conv_s}</div>'
        f'<div class="act-move">{move}</div>'
        f'<div class="act-levels">{levels_html}</div>'
        f"</div>"
    )


def _action_group(action: str, cls: str, hint: str, rows: list) -> str:
    if not rows:
        return ""
    body = "".join(_action_row(a) for a in rows)
    n = len(rows)
    count = f"{n} name" + ("" if n == 1 else "s")
    return (
        f'<div class="act-grp act-grp--{cls}">'
        f'<div class="act-grp-head"><span class="act-tag">{action}</span>'
        f'<span class="act-hint">{_esc(hint)}</span>'
        f'<span class="act-count">{count}</span></div>'
        f'<div class="act-rows">{body}</div>'
        f"</div>"
    )


def _actions_block(actions: list, approx_current: bool = False) -> str:
    """Decision-support action groups (BUY / ADD / TRIM / SELL / HOLD).

    Empty groups are omitted. When ``approx_current`` is set (no live account),
    a clear note flags that current weights are an equal-split approximation.
    """
    note = '<div class="act-note">Decision-support: you decide, not auto-executed.'
    if approx_current:
        note += (
            ' <span class="act-note-warn">Current weights are approximate '
            "(equal-split; no live account file supplied).</span>"
        )
    note += "</div>"
    groups = "".join(
        _action_group(act, cls, hint, [a for a in actions if a.get("action") == act])
        for act, cls, hint in _ACTION_GROUPS
    )
    return f'<div class="actions-block">{note}{groups}</div>'


def _card_grid(frame: pd.DataFrame, cols, conv_scale: float) -> str:
    cards = [
        _card(t, r, cols, conv_scale, min(i * 0.04, 0.36))
        for i, (t, r) in enumerate(frame.iterrows())
    ]
    return f'<div class="cards">{"".join(cards)}</div>'


# --- action deep-dive cards -------------------------------------------------


def _action_card(a: dict, row, cols, conv_scale: float, delay: float) -> str:
    """Full deep-dive card for one suggested action (replaces compact _action_row)."""
    tkr = _esc(a.get("ticker") or "")
    action = a.get("action") or ""
    act_cls = {"BUY": "buy", "ADD": "add", "TRIM": "trim", "SELL": "sell"}.get(action, "hold")
    conv = a.get("conviction")
    conv_s = "·" if _isnan(conv) else f"{conv:+.2f}"
    cur, tgt, delta = a.get("current_pct"), a.get("target_pct"), a.get("delta_pct")
    usd = _usd_signed(a.get("delta_usd"))
    if row is not None:
        name = _esc(row.get("name", "")) or "&nbsp;"
        sector = _esc(row.get("sector", "")) or "n/a"
        desc_raw = row.get("description", "")
        desc_text = (
            ""
            if (desc_raw is None or (isinstance(desc_raw, float) and pd.isna(desc_raw)))
            else str(desc_raw).strip()
        )
        desc_html = f'<div class="desc">{_esc(desc_text)}</div>' if desc_text else ""
        groups = "".join(_group(row, cols, gl, ms) for gl, ms in DISPLAY_GROUPS)
    else:
        name = "&nbsp;"
        sector = "n/a"
        desc_html = ""
        groups = ""
    pnl = a.get("pnl")
    pnl_html = ""
    if pnl is not None:
        pct, cv = a.get("pnl_pct"), a.get("current_value")
        pcol = "#2d6a4f" if float(pnl) >= 0 else "#b3402f"
        pct_s = f" ({pct:+.1f}%)" if pct is not None else ""
        val_s = f" · {_money(cv)}" if cv is not None else ""
        pnl_html = (
            f'<span class="act-pnl" style="color:{pcol};">'
            f"P/L {_usd_signed(pnl)}{pct_s}{val_s}</span>"
        )
    move = (
        f'<span class="from">{_pct1(cur)}</span>'
        f' <span class="to">&#8594; {_pct1(tgt)}</span>'
        f' <span class="act-dpp">({_pp(delta)})</span>'
        + (f'<span class="act-usd"> {usd}</span>' if usd else "")
        + pnl_html
    )
    levels_html = _levels_block(row) if (action in _LEVEL_ACTIONS and row is not None) else ""
    strip_html = f'<div class="ac-strip">{levels_html}</div>' if levels_html else ""
    cluster_strip_html = _cluster_strip(row)
    inner_groups = f'<div class="groups">{groups}</div>' if groups else ""
    groups_html = (
        f'<details class="card-more">'
        f'<summary class="card-more-h">All factors'
        f' <span class="chev">&#9662;</span></summary>'
        f"{inner_groups}"
        f"</details>"
        if inner_groups
        else ""
    )
    return (
        f'<article class="card card--action" style="animation-delay:{delay:.2f}s;">'
        f'<div class="ac-head">'
        f'<div class="ac-id">'
        f'<div class="ac-tkr">{tkr}</div>'
        f'<div class="ac-name">{name}</div>'
        f"{desc_html}"
        f'<div class="id-meta"><span class="chip">{sector}</span></div>'
        f"</div>"
        f'<div class="ac-verdict">'
        f'<span class="act-tag act-tag--{act_cls}">{_esc(action)}</span>'
        f'<div class="ac-conv" style="color:{_z_color(conv)};">{conv_s}</div>'
        f'<div class="ac-move">{move}</div>'
        f"</div>"
        f"</div>"
        f"{strip_html}"
        f"{cluster_strip_html}"
        f"{groups_html}"
        f"</article>"
    )


def _action_group_cards(action, cls, hint, rows, scores, cols, conv_scale):
    """Render one action bucket (BUY / ADD / TRIM / SELL / HOLD) as deep-dive cards."""
    if not rows:
        return ""
    idx = scores.index if scores is not None else pd.Index([])
    cards_html = "".join(
        _action_card(
            a,
            scores.loc[a["ticker"]] if (a.get("ticker") and a["ticker"] in idx) else None,
            cols,
            conv_scale,
            i * 0.03,
        )
        for i, a in enumerate(rows)
    )
    n = len(rows)
    count = f"{n} name" + ("" if n == 1 else "s")
    return (
        f'<div class="act-grp act-grp--{cls}">'
        f'<div class="act-grp-head">'
        f'<span class="act-tag act-tag--{cls}">{action}</span>'
        f'<span class="act-hint">{_esc(hint)}</span>'
        f'<span class="act-count">{count}</span>'
        f"</div>"
        f'<div class="act-cards">{cards_html}</div>'
        f"</div>"
    )


def _actions_block_cards(actions, scores, conv_scale: float, approx_current: bool = False) -> str:
    """Render all action buckets as deep-dive cards with a one-line summary header."""
    cols = scores.columns if scores is not None else []
    counts = {
        act: sum(1 for a in actions if a.get("action") == act) for act, _, _ in _ACTION_GROUPS
    }
    parts = [f"{counts[act]} {act}" for act, _, _ in _ACTION_GROUPS if counts[act] > 0]
    summary_html = f'<div class="act-summary">{" · ".join(parts)}</div>' if parts else ""
    note = '<div class="act-note">Decision-support: you decide, not auto-executed.'
    if approx_current:
        note += (
            ' <span class="act-note-warn">Current weights are approximate '
            "(equal-split; no live account file supplied).</span>"
        )
    note += "</div>"
    groups = "".join(
        _action_group_cards(
            act,
            cls,
            hint,
            [a for a in actions if a.get("action") == act],
            scores,
            cols,
            conv_scale,
        )
        for act, cls, hint in _ACTION_GROUPS
    )
    return f'<div class="actions-block">{summary_html}{note}{groups}</div>'


def _portfolio_section(scores: pd.DataFrame, conv_scale: float, numeral: str = "II") -> str:
    cols = scores.columns
    mask = scores.get("is_portfolio", pd.Series(False, index=scores.index)) == True  # noqa: E712
    port = scores[mask].sort_values("conviction", ascending=False, na_position="last")
    sub = f"Current holdings · {len(port)} name(s) · tilt = ADD / HOLD / TRIM"
    head = _section_head(numeral, "Portfolio", sub)
    if port.empty:
        return head + '<div class="empty">No portfolio holdings in this run.</div>'
    return head + _card_grid(port, cols, conv_scale)


def _candidates_section(
    scores: pd.DataFrame, conv_scale: float, max_long: int, max_avoid: int, numeral: str = "III"
) -> str:
    cols = scores.columns
    mask = scores.get("is_portfolio", pd.Series(False, index=scores.index)) == False  # noqa: E712
    cand = scores[mask].sort_values("conviction", ascending=False, na_position="last")
    total = len(cand)
    if cand.empty:
        return (
            _section_head(numeral, "Candidates", "Buy-signal watchlist")
            + '<div class="empty">No candidates in this run.</div>'
        )

    longs = cand.head(max_long)
    remaining = cand.drop(longs.index)
    avoids = remaining[remaining["conviction"].notna()].tail(max_avoid).iloc[::-1]
    shown = len(longs) + len(avoids)

    sub = (
        f"Buy-signal watchlist · showing {shown} of {total} in detail · "
        f"top {len(longs)} by conviction"
        + (f" and bottom {len(avoids)}" if len(avoids) else "")
        + " · tilt = BUY-WATCH / WATCH / PASS"
    )
    head = _section_head(numeral, "Candidates", sub)
    body = _card_grid(longs, cols, conv_scale)
    if len(avoids):
        body += '<div class="cards-divider"><span>Lowest conviction · potential avoids</span></div>'
        body += _card_grid(avoids, cols, conv_scale)
    return head + body


# --- portfolio allocation bars ----------------------------------------------
def _allocation_bars(allocations: dict | None) -> str:
    """Portfolio allocation breakdown: Geography / Asset Type / Sector.

    Renders a responsive 3-column grid of horizontal thick bars (one group per
    column). Returns empty string when allocations is falsy or all three groups
    are empty after filtering.
    """
    if not allocations:
        return ""

    def _group(title: str, data: dict) -> str:
        if not data:
            return ""
        rows = []
        for label, v in data.items():
            pct = float(v)
            w = round(pct * 100)
            rows.append(
                f'<div class="alloc-row">'
                f'<div class="alloc-lbl">{_esc(label)}</div>'
                f'<div class="alloc-track">'
                f'<div class="alloc-fill" style="width:{w}%;"></div>'
                f"</div>"
                f'<div class="alloc-pct">{w}%</div>'
                f"</div>"
            )
        return (
            f'<div class="alloc-grp">'
            f'<div class="alloc-grp-title">{_esc(title)}</div>'
            f"{''.join(rows)}"
            f"</div>"
        )

    geo = _group("Geography", allocations.get("geography") or {})
    at = _group("Asset Type", allocations.get("asset_type") or {})
    sec = _group("Sector", allocations.get("sector") or {})

    if not (geo or at or sec):
        return ""

    return (
        f'<div class="alloc-section">'
        f'<div class="alloc-heading">Portfolio Allocation</div>'
        f'<div class="alloc-grid">{geo}{at}{sec}</div>'
        f"</div>"
    )


# --- stylesheet -------------------------------------------------------------
def _stylesheet() -> str:
    return """
:root{
  --canvas:#ffffff;--warm:#faf9f7;--ink:#16181d;--ink2:#4a4e57;--muted:#9aa0a8;
  --line:#e7e6e2;--accent:#123b3a;--track:#ecebe4;--hover:#f4f2ee;
  --bull:#2d6a4f;--bull-bar:#bfe3cd;--bull-fill:#e9f5ee;
  --bear:#b3402f;--bear-bar:#f2c4bb;--bear-fill:#fbeeec;
  --serif:'Hanken Grotesk',system-ui,sans-serif;
  --sans:'Hanken Grotesk',system-ui,sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,'SFMono-Regular',monospace;
}
*{box-sizing:border-box;}
html{-webkit-text-size-adjust:100%;}
body{margin:0;background:var(--canvas);color:var(--ink2);font-family:var(--sans);
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased;
  text-rendering:optimizeLegibility;}
.wrap{max-width:1168px;margin:0 auto;padding:0 44px 90px;}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums;}
.eyebrow{font-family:var(--sans);font-size:10.5px;font-weight:600;letter-spacing:1.3px;
  text-transform:uppercase;color:var(--muted);}

/* masthead */
.masthead{padding:66px 0 30px;}
.mast-eyebrow{font-size:10.5px;font-weight:600;letter-spacing:2px;text-transform:uppercase;
  color:var(--accent);}
.mast-title{font-family:var(--serif);font-optical-sizing:auto;font-weight:400;
  font-size:64px;line-height:1;letter-spacing:-1px;color:var(--ink);margin:14px 0 0;}
.mast-rule{width:80px;height:3px;background:var(--accent);margin:26px 0 22px;}
.standfirst{font-family:var(--serif);font-optical-sizing:auto;font-weight:400;
  font-size:20px;line-height:1.5;color:var(--ink2);margin:0;}
.mast-meta{display:flex;flex-wrap:wrap;align-items:center;gap:10px 18px;margin-top:28px;
  padding-top:22px;border-top:1px solid var(--line);}
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
.section-head{display:flex;align-items:baseline;gap:20px;margin:64px 0 24px;
  padding-bottom:16px;border-bottom:1px solid var(--line);}
.sec-num{font-family:var(--serif);font-optical-sizing:auto;font-weight:300;font-size:44px;
  line-height:.8;color:var(--accent);min-width:48px;}
.sec-title{font-family:var(--serif);font-weight:500;font-size:25px;letter-spacing:-.3px;
  color:var(--ink);}
.sec-sub{font-size:12.5px;color:var(--muted);margin-top:4px;}

/* overview heatmap */
.hm{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--canvas);
  box-shadow:0 1px 2px rgba(22,24,29,.04);}
.hm-head,.hm-row{display:grid;
  grid-template-columns:14px 30px 74px minmax(100px,1.5fr) minmax(80px,1fr) 52px 96px 250px;
  align-items:center;gap:14px;padding:0 20px;}
.hm-head{height:40px;background:var(--warm);border-bottom:1px solid var(--line);}
.hm-row{min-height:34px;border-top:1px solid var(--line);}
.hm-body .hm-row:first-child{border-top:none;}
.hm-body .hm-row:nth-child(even){background:var(--warm);}
.hm-row:hover{background:var(--hover);}
.hm-h{font-size:9px;font-weight:600;letter-spacing:.7px;text-transform:uppercase;color:var(--muted);}
.hm-h.center{text-align:center;}
.hm-h.right{text-align:right;}
.hm-dot{width:7px;height:7px;border-radius:50%;}
.hm-dot.pf{background:var(--accent);}
.hm-rank{font-family:var(--mono);font-size:11px;color:var(--muted);text-align:right;
  font-variant-numeric:tabular-nums;}
.hm-tkr{font-family:var(--serif);font-weight:600;font-size:15.5px;color:var(--ink);letter-spacing:-.2px;}
.hm-sector{font-size:11.5px;color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.hm-name{font-size:11px;color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.hm-conv{font-family:var(--mono);font-size:12.5px;font-weight:500;text-align:right;
  font-variant-numeric:tabular-nums;}
.hm-cells{display:grid;grid-template-columns:repeat(5,1fr);gap:5px;align-items:center;}
.hm-cell{height:22px;border-radius:4px;display:flex;align-items:center;justify-content:center;
  font-family:var(--mono);font-size:10px;color:transparent;}
.hm-cell.nan{background:transparent;color:var(--muted);}
.excluded-note{margin-top:15px;font-size:11.5px;font-style:italic;color:var(--muted);letter-spacing:.2px;}

/* cards: always one full-width card per row, on every device */
.cards{display:grid;grid-template-columns:1fr;gap:32px;max-width:1080px;margin:0 auto;}
.card{position:relative;background:var(--canvas);border:1px solid var(--line);border-radius:14px;
  padding:30px 32px 34px;overflow:hidden;container-type:inline-size;
  box-shadow:0 1px 2px rgba(22,24,29,.04);transition:transform .25s ease,box-shadow .25s ease,
  border-color .25s ease;animation:rise .6s cubic-bezier(.2,.7,.2,1) both;}
.card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--accent),#3a7d6b 55%,rgba(18,59,58,0));opacity:.55;}
.card:hover{transform:translateY(-3px);box-shadow:0 18px 44px rgba(22,24,29,.11);
  border-color:#dad8d2;}

/* card header: identity (left) + verdict (right) */
.card-top{display:flex;align-items:flex-start;justify-content:space-between;gap:22px 40px;}
.id{min-width:0;}
.tkr{font-family:var(--serif);font-optical-sizing:auto;font-weight:600;font-size:34px;line-height:1;
  letter-spacing:-.5px;color:var(--ink);display:flex;align-items:baseline;gap:11px;}
.pf-tag{font-family:var(--sans);font-size:9px;font-weight:600;letter-spacing:1px;color:var(--accent);
  background:var(--warm);border:1px solid var(--line);border-radius:100px;padding:2px 7px;}
.name{font-size:13.5px;color:var(--ink2);margin-top:8px;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;}
.desc{font-size:12px;color:var(--muted);margin-top:5px;line-height:1.45;overflow:hidden;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;max-width:480px;}
.id-meta{display:flex;align-items:center;gap:10px;margin-top:12px;}
.chip{font-size:9.5px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;color:var(--ink2);
  background:var(--warm);border:1px solid var(--line);border-radius:100px;padding:3px 10px;
  white-space:nowrap;}
.verdict-top{display:flex;align-items:flex-start;gap:26px;flex-shrink:0;}
.price-block{text-align:left;}
.vk{font-size:9px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:var(--muted);}
.price{font-family:var(--mono);font-size:16px;font-weight:500;color:var(--ink);margin-top:6px;
  font-variant-numeric:tabular-nums;letter-spacing:-.2px;}
.conv-block{text-align:right;}
.conv{font-family:var(--mono);font-weight:500;font-size:38px;line-height:.9;letter-spacing:-1.2px;
  font-variant-numeric:tabular-nums;}
.conv-l{font-size:9px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:var(--muted);
  margin-top:7px;}
.tilt{display:inline-block;margin-top:11px;font-size:10px;font-weight:600;letter-spacing:1px;
  text-transform:uppercase;padding:4px 11px;border-radius:100px;border:1px solid var(--line);}
.tilt-bull{color:var(--bull);background:var(--bull-fill);border-color:var(--bull-bar);}
.tilt-bear{color:var(--bear);background:var(--bear-fill);border-color:var(--bear-bar);}
.tilt-flat{color:var(--ink2);background:var(--warm);}

/* full-width strip: conviction meter (left) + trade-level tiles (right) */
.card-strip{display:flex;align-items:flex-end;gap:32px;margin-top:22px;padding-top:20px;
  border-top:1px solid var(--line);}
.meter-col{flex:1;min-width:120px;}
.meter-col .vk{margin-bottom:10px;}
.meter-wrap{width:100%;}
.levels{flex:0 0 auto;width:360px;max-width:100%;}
.levels-head{font-size:9px;font-weight:600;letter-spacing:1.1px;text-transform:uppercase;
  color:var(--muted);margin-bottom:9px;}
.levels-tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;}
.lvl{border:1px solid var(--line);border-radius:9px;padding:8px 10px;background:var(--warm);}
.lvl-l{font-size:8.5px;font-weight:600;letter-spacing:.6px;text-transform:uppercase;color:var(--muted);}
.lvl-v{font-family:var(--mono);font-size:13px;font-weight:500;color:var(--ink);margin-top:4px;
  font-variant-numeric:tabular-nums;letter-spacing:-.3px;}
.lvl-stop .lvl-v{color:var(--bear);}
.lvl-tgt .lvl-v{color:var(--bull);}
.levels-note{font-size:9px;color:var(--muted);margin-top:9px;letter-spacing:.2px;}

/* factor groups: a responsive grid inside the card (4 / 2 / 1 by card width) */
.groups{margin-top:28px;padding-top:26px;border-top:1px solid var(--line);
  display:grid;grid-template-columns:1fr;gap:26px 34px;}
.grp{min-width:0;}
.grp-head{display:flex;align-items:baseline;justify-content:space-between;gap:8px;margin-bottom:11px;
  padding-bottom:9px;border-bottom:1px solid var(--line);}
.grp-name{font-size:10px;font-weight:600;letter-spacing:1.3px;text-transform:uppercase;color:var(--ink);}
.grp-z{font-family:var(--mono);font-size:11.5px;font-weight:500;font-variant-numeric:tabular-nums;}
.units{display:grid;grid-template-columns:1fr;gap:0;}
.unit{padding:7px 0;border-top:1px dotted var(--line);}
.unit:first-child{border-top:none;padding-top:1px;}
.unit-row{display:flex;align-items:baseline;justify-content:space-between;gap:12px;}
.unit-l{font-size:9px;font-weight:600;letter-spacing:.5px;text-transform:uppercase;color:var(--muted);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.unit-v{font-family:var(--mono);font-size:13px;font-weight:500;color:var(--ink);
  font-variant-numeric:tabular-nums;letter-spacing:-.2px;white-space:nowrap;}
.unit-z{display:flex;align-items:center;gap:8px;margin-top:7px;}
.znum{font-family:var(--mono);font-size:10px;font-weight:500;font-variant-numeric:tabular-nums;
  min-width:32px;text-align:right;}

/* diverging bars */
.zbar,.meter{position:relative;display:inline-block;background:var(--track);border-radius:100px;
  vertical-align:middle;overflow:hidden;}
.zbar{width:100%;height:6px;flex:1;}
.meter{width:100%;height:10px;}
.zbar::before,.meter::before{content:"";position:absolute;left:calc(50% - .5px);top:0;bottom:0;
  width:1px;background:rgba(22,24,29,.20);z-index:1;}
.zbar-fill,.meter-fill{position:absolute;top:0;bottom:0;z-index:0;}
.zbar-pos,.meter-pos{left:50%;background:linear-gradient(90deg,#cdead9,var(--bull));}
.zbar-neg,.meter-neg{right:50%;background:linear-gradient(270deg,#f4cec6,var(--bear));}

/* labeled divider between the long block and the avoid block */
.cards-divider{display:flex;align-items:center;gap:18px;margin:44px auto 30px;max-width:1080px;}
.cards-divider::before,.cards-divider::after{content:"";height:1px;background:var(--line);flex:1;}
.cards-divider span{font-size:10px;font-weight:600;letter-spacing:1.3px;text-transform:uppercase;
  color:var(--muted);white-space:nowrap;}

.empty{color:var(--muted);font-size:13px;padding:14px 0;}

/* Phase 5D: executive summary strip */
.exec-summary{display:flex;flex-wrap:wrap;align-items:center;gap:8px 10px;margin:30px 0 4px;
  padding:14px 20px;border:1px solid var(--line);border-radius:11px;background:var(--warm);
  font-family:var(--mono);font-size:12.5px;color:var(--ink2);font-variant-numeric:tabular-nums;}
.es-item{font-weight:500;color:var(--ink);}
.es-sep{color:var(--muted);margin:0 2px;}

/* Phase 5D: exec / risk panel */
.exec-panel{margin-top:4px;}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;}
.stat{border:1px solid var(--line);border-radius:12px;padding:15px 16px;background:var(--canvas);
  box-shadow:0 1px 2px rgba(22,24,29,.04);}
.stat-l{font-size:9px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:var(--muted);}
.stat-v{font-family:var(--mono);font-size:23px;font-weight:500;color:var(--ink);margin-top:9px;
  line-height:1;font-variant-numeric:tabular-nums;letter-spacing:-.5px;}
.stat-sub{font-size:10.5px;color:var(--muted);margin-top:7px;}
.stat--bull .stat-v{color:var(--bull);}
.stat--bear .stat-v{color:var(--bear);}
.stat--warn .stat-v{color:#96601a;}
.exec-sub{margin-top:18px;display:flex;flex-wrap:wrap;align-items:center;gap:8px;}
.exec-flags{margin-top:12px;display:flex;flex-wrap:wrap;align-items:center;gap:8px;}
.exec-sub-l{font-size:9px;font-weight:600;letter-spacing:1px;text-transform:uppercase;
  color:var(--muted);margin-right:4px;}
.sec-chip{font-family:var(--mono);font-size:11px;color:var(--ink2);background:var(--warm);
  border:1px solid var(--line);border-radius:100px;padding:4px 11px;font-variant-numeric:tabular-nums;}
.sec-chip b{color:var(--ink);font-weight:600;}
.flag-chip{font-size:10px;font-weight:600;letter-spacing:.3px;padding:4px 10px;border-radius:100px;
  border:1px solid var(--bear-bar);background:var(--bear-fill);color:var(--bear);}
.flag-chip.lever{border-color:var(--line);background:var(--warm);color:var(--ink2);}
.flag-none{font-size:11px;color:var(--muted);}

/* Phase 5D: suggested actions */
.act-note{font-size:11.5px;font-style:italic;color:var(--muted);margin:0 0 20px;letter-spacing:.2px;}
.act-note-warn{color:var(--bear);font-style:normal;font-weight:500;}
.act-grp{border:1px solid var(--line);border-left:3px solid var(--muted);border-radius:11px;
  padding:15px 18px 7px;margin-bottom:16px;background:var(--canvas);
  box-shadow:0 1px 2px rgba(22,24,29,.04);}
.act-grp-head{display:flex;align-items:baseline;gap:12px;margin-bottom:6px;}
.act-tag{font-size:11px;font-weight:700;letter-spacing:1.2px;}
.act-hint{font-size:11px;color:var(--muted);}
.act-count{margin-left:auto;font-family:var(--mono);font-size:10.5px;color:var(--muted);
  font-variant-numeric:tabular-nums;}
.act-grp--buy{border-left-color:var(--bull);}
.act-grp--buy .act-tag{color:var(--bull);}
.act-grp--add{border-left-color:var(--bull-bar);}
.act-grp--add .act-tag{color:var(--bull);}
.act-grp--trim{border-left-color:#e6c88f;}
.act-grp--trim .act-tag{color:#96601a;}
.act-grp--sell{border-left-color:var(--bear);}
.act-grp--sell .act-tag{color:var(--bear);}
.act-grp--hold{border-left-color:var(--line);}
.act-grp--hold .act-tag{color:var(--ink2);}
.act-rows{display:flex;flex-direction:column;}
.act-row{display:grid;grid-template-columns:minmax(150px,1.5fr) 56px minmax(150px,1.4fr) auto;
  align-items:center;gap:8px 16px;padding:10px 0;border-top:1px dotted var(--line);}
.act-row:first-child{border-top:none;}
.act-id{min-width:0;}
.act-tkr{font-family:var(--serif);font-weight:600;font-size:16px;color:var(--ink);letter-spacing:-.2px;}
.act-name{font-size:11px;color:var(--muted);line-height:1.3;}
.act-conv{font-family:var(--mono);font-size:12.5px;font-variant-numeric:tabular-nums;text-align:right;}
.act-move{font-family:var(--mono);font-size:12.5px;color:var(--ink2);font-variant-numeric:tabular-nums;}
.act-move .to{color:var(--ink);font-weight:600;}
.act-move .act-dpp{color:var(--muted);}
.act-usd{color:var(--muted);margin-left:8px;}
.act-pnl{font-family:var(--mono);font-size:11px;font-weight:600;margin-left:10px;white-space:nowrap;}
.act-levels{display:flex;flex-wrap:wrap;gap:6px;justify-content:flex-end;}
.act-lvl{font-family:var(--mono);font-size:10.5px;color:var(--ink2);background:var(--warm);
  border:1px solid var(--line);border-radius:7px;padding:3px 8px;font-variant-numeric:tabular-nums;}
.act-lvl.sl{color:var(--bear);}
.act-lvl.tp{color:var(--bull);}
/* action cards (deep-dive) */
.act-summary{font-family:var(--mono);font-size:12.5px;font-weight:600;color:var(--ink2);
  margin-bottom:16px;letter-spacing:.3px;font-variant-numeric:tabular-nums;}
.act-tag--buy{color:var(--bull);}
.act-tag--add{color:var(--bull);}
.act-tag--trim{color:#96601a;}
.act-tag--sell{color:var(--bear);}
.act-tag--hold{color:var(--ink2);}
.act-cards{display:flex;flex-direction:column;gap:16px;padding-top:10px;}
.ac-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px 30px;
  flex-wrap:wrap;}
.ac-id{min-width:0;flex:1;}
.ac-tkr{font-family:var(--serif);font-optical-sizing:auto;font-weight:700;font-size:26px;
  line-height:1;letter-spacing:-.4px;color:var(--ink);}
.ac-name{font-size:13px;color:var(--ink2);margin-top:6px;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;}
.ac-verdict{display:flex;flex-direction:column;align-items:flex-end;gap:4px;flex-shrink:0;}
.ac-conv{font-family:var(--mono);font-weight:500;font-size:22px;line-height:.9;
  letter-spacing:-.8px;font-variant-numeric:tabular-nums;}
.ac-move{font-family:var(--mono);font-size:12px;color:var(--ink2);
  font-variant-numeric:tabular-nums;text-align:right;}
.ac-strip{margin-top:14px;padding-top:12px;border-top:1px solid var(--line);}

/* cluster summary strip: always visible in the collapsed card header */
.cluster-strip{display:flex;flex-wrap:wrap;gap:7px;margin-top:14px;}
.cz-chip{display:inline-flex;align-items:center;gap:5px;background:var(--warm);
  border:1px solid var(--line);border-radius:7px;padding:4px 8px;white-space:nowrap;}
.cz-chip.cz-nan{opacity:.55;}
.cz-label{font-size:8.5px;font-weight:600;letter-spacing:.6px;text-transform:uppercase;
  color:var(--muted);}
.cz-bar{width:32px;height:4px;background:var(--track);border-radius:100px;
  overflow:hidden;flex-shrink:0;}
.cz-fill{height:100%;border-radius:100px;}
.cz-val{font-family:var(--mono);font-size:9.5px;font-weight:500;
  font-variant-numeric:tabular-nums;min-width:26px;text-align:right;}
.cz-nan .cz-val{color:var(--muted);}

/* collapsible factor-groups disclosure */
details.card-more{margin-top:14px;padding-top:12px;border-top:1px solid var(--line);}
summary.card-more-h{display:flex;align-items:center;justify-content:space-between;
  padding:4px 0;cursor:pointer;list-style:none;-webkit-list-style:none;
  font-size:9px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;
  color:var(--muted);user-select:none;}
summary.card-more-h::-webkit-details-marker{display:none;}
summary.card-more-h::marker{display:none;}
.chev{display:inline-block;transition:transform .2s ease;font-size:11px;line-height:1;}
details[open].card-more .chev{transform:rotate(180deg);}
details[open].card-more>summary{margin-bottom:12px;}

.acct-grid{margin-bottom:14px;}
.card--action{padding:20px 24px 22px;}
.card--action .groups{margin-top:16px;padding-top:14px;gap:14px 20px;}
@media (max-width:900px){.stat-grid{grid-template-columns:repeat(2,1fr);}}
@media (max-width:560px){
  .act-row{grid-template-columns:1fr auto;}
  .act-move,.act-levels{grid-column:1 / -1;justify-content:flex-start;}
}
@media (max-width:420px){.stat-grid{grid-template-columns:1fr 1fr;}}

/* footer */
.footer{margin-top:72px;padding-top:26px;border-top:2px solid var(--ink);}
.footer .eyebrow{margin-bottom:12px;}
.method{font-size:11.5px;color:var(--muted);line-height:1.75;}
.method b{color:var(--ink2);font-weight:600;}
.foot-stamp{margin-top:18px;font-family:var(--mono);font-size:11px;color:var(--muted);
  font-variant-numeric:tabular-nums;}

@keyframes rise{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:none;}}

/* card internals reflow to the CARD's own width (not the viewport) */
@container (max-width:640px){
  .card-top{flex-direction:column;}
  .verdict-top{width:100%;justify-content:space-between;}
  .card-strip{flex-direction:column;align-items:stretch;gap:18px;}
  .levels{width:100%;}
}
@container (min-width:640px){
  .groups{grid-template-columns:1fr 1fr;}
}
@container (min-width:1000px){
  .groups{grid-template-columns:repeat(4,1fr);}
}
@container (min-width:820px){
  .card--action .groups{grid-template-columns:repeat(4,1fr);}
}
@media (max-width:820px){
  .hm-head,.hm-row{grid-template-columns:12px 26px 62px minmax(60px,1fr) 46px 200px;}
  .hm-barcell{display:none;}
  .hm-name{display:none;}
}
@media (max-width:639px){
  /* Phone: ticker always visible; hide sector column to reclaim width. */
  .hm-head,.hm-row{grid-template-columns:12px 24px 60px 44px 1fr;}
  .hm-sector{display:none;}
}
@media (max-width:680px){
  .wrap{padding:0 18px 60px;}
  .mast-title{font-size:42px;}
  .mast-stats{margin-left:0;width:100%;}
  .card{padding:22px 20px 26px;}
}
/* portfolio allocation bars */
.alloc-section{margin-top:28px;padding-top:22px;border-top:1px solid var(--line);}
.alloc-heading{font-size:10px;font-weight:600;letter-spacing:1.3px;text-transform:uppercase;
  color:var(--muted);margin-bottom:16px;}
.alloc-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;}
.alloc-grp-title{font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase;
  color:var(--ink);margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid var(--line);}
.alloc-row{display:grid;grid-template-columns:110px 1fr 34px;align-items:center;gap:8px;
  margin-bottom:7px;}
.alloc-lbl{font-size:11px;color:var(--ink2);white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;}
.alloc-track{height:26px;background:var(--track);border-radius:6px;overflow:hidden;}
.alloc-fill{height:100%;background:var(--accent);border-radius:6px;opacity:.65;}
.alloc-pct{font-family:var(--mono);font-size:11px;color:var(--muted);text-align:right;
  font-variant-numeric:tabular-nums;}
@media (max-width:720px){.alloc-grid{grid-template-columns:1fr;}}
@media (prefers-reduced-motion:reduce){
  .card{animation:none;}
  .card:hover{transform:none;}
}
""".strip()


# --- top-level --------------------------------------------------------------
def render_report(
    scores: pd.DataFrame,
    meta: dict,
    *,
    max_long_cards: int = _MAX_LONG_CARDS,
    max_avoid_cards: int = _MAX_AVOID_CARDS,
    portfolio: dict | None = None,
    actions: list | None = None,
    conditioning: dict | None = None,
) -> str:
    """Render the full standalone HTML factor report.

    Args:
        scores: Scored frame (from ``compute_scores``) indexed by ticker,
            with cluster z's, ``conviction``, ``rank`` and ``is_portfolio``.
        meta: Header metadata (date, regime, counts, system_read, ...). When
            ``actions`` is supplied, ``meta["current_weights_approx"]`` (bool)
            drives a note that current weights are an equal-split approximation.
        max_long_cards: Max highest-conviction candidate cards to detail.
        max_avoid_cards: Max lowest-conviction candidate cards to detail.
        portfolio: Optional ``build_portfolio`` result dict. When given, an
            executive / risk panel (deployment, vol vs ceiling, CVaR, net beta,
            effective bets, caps, sector + USD-bloc exposure, binding/gate flags)
            renders ABOVE the factor cards.
        actions: Optional ``build_actions`` list. When given, a decision-support
            Suggested Actions section (grouped BUY / ADD / TRIM / SELL / HOLD)
            renders ABOVE the factor cards.
        conditioning: Optional ``resolve_deployment`` dial-diagnostics dict, used
            by the exec panel + summary line for regime / deployment context.

    ``portfolio`` / ``actions`` / ``conditioning`` are fully optional: when all
    three are ``None`` the output is byte-for-byte identical to the legacy report.

    Returns:
        A complete HTML document string.
    """
    meta = meta or {}

    # Eligibility gate: only equities with core data are ranked / shown. Frames
    # without an ``eligible`` column (older callers) are treated as all-eligible.
    if "eligible" in scores.columns:
        elig = scores["eligible"].fillna(False).astype(bool)
    else:
        elig = pd.Series(True, index=scores.index)
    scored = scores[elig]
    n_excluded = int((~elig).sum())

    n_port = meta.get("n_portfolio", int(scored.get("is_portfolio", pd.Series(dtype=bool)).sum()))
    n_cand = meta.get("n_candidates", 0)
    date = _esc(meta.get("date", ""))
    regime = _esc(meta.get("regime", "NEUTRAL"))
    regime_detail = _esc(meta.get("regime_detail", ""))
    system_read = _esc(meta.get("system_read", ""))
    priced = _esc(meta.get("priced", ""))
    enriched = _esc(meta.get("enriched", ""))
    generated = _esc(meta.get("generated_utc", ""))
    _cov = meta.get("coverage") or {}
    cov_str = ""
    if _cov.get("n_scored"):
        cov_str = f" · Coverage {_cov.get('pct', 0) * 100:.0f}% ({_cov.get('n_eligible', 0)}/{_cov['n_scored']} eligible)"
        if _cov.get("adv_dropped"):
            cov_str += f", ADV-dropped {_cov['adv_dropped']}"

    regime_cls = {"RISK_ON": "regime-on", "RISK_OFF": "regime-off"}.get(
        str(meta.get("regime")), "regime-neutral"
    )
    conv_scale = _conv_scale(scored)
    n_universe = len(scored)

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
        f"Priced {priced} / Enriched {enriched}{cov_str}</span>"
        f"</div></header>"
    )

    # Phase 5D: exec / risk panel + suggested actions render ABOVE the factor
    # cards. Section numerals are assigned left-to-right so that when no new
    # section leads, Overview / Portfolio / Candidates keep I / II / III.
    sec_idx = 0
    new_top = ""
    if portfolio is not None:
        new_top += (
            _section_head(
                _ROMAN[sec_idx],
                "Positioning and Risk",
                "Deployment, portfolio risk and exposure limits after the risk gate",
            )
            + _exec_panel(portfolio, conditioning, meta)
            + _allocation_bars(meta.get("allocations"))
        )
        sec_idx += 1
    if actions is not None:
        new_top += _section_head(
            _ROMAN[sec_idx],
            "Suggested Actions",
            "Risk-gated target book vs current holdings · grouped by action",
        ) + _actions_block_cards(
            actions, scored, conv_scale, bool(meta.get("current_weights_approx"))
        )
        sec_idx += 1

    overview = _section_head(
        _ROMAN[sec_idx],
        "Overview",
        f"Conviction heatmap · all {n_universe} scored names ranked · "
        "cluster factor tilt by cell color",
    ) + _overview(scored, conv_scale)
    if n_excluded:
        plural = "" if n_excluded == 1 else "s"
        overview += (
            f'<div class="excluded-note">{n_excluded} name{plural} excluded '
            f"(non-equity or insufficient data).</div>"
        )
    sec_idx += 1
    summary_line = (
        _exec_summary_line(meta, portfolio, actions, conditioning)
        if (portfolio is not None or actions is not None)
        else ""
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
        f"present. Conviction is re-z-scored cross-sectionally; rank 1 = best. Bars and heat-cells "
        f"are centered at 0: green reads attractive, red unattractive, intensity scaling with the "
        f"z. The <b>Overview</b> ranks the full eligible universe; detailed cards are curated to "
        f"the portfolio plus the strongest and weakest candidates. Non-equity instruments (ETFs, "
        f"funds) and names missing core data are held out of the ranking so equity-factor "
        f"convictions are not applied to instruments they do not describe. <b>Trade Levels</b> "
        f"are vol-scaled: the monthly move sigma is the Close-based realized daily vol times the "
        f"square root of 21; the stop sits near 2 sigma below entry, the target near 3 sigma "
        f"above (about 1.5:1 reward-to-risk), and are indicative, not orders.<br><br>"
        f"This is a <b>shadow / decision-support</b> snapshot, not yet capital-authorized. The "
        f"naive price-spine (12-1 momentum plus low-vol) showed no standalone out-of-sample edge "
        f"in validation, so treat single-factor tilts with caution and weigh the full-cluster "
        f"conviction over any one number."
        f"</div>"
        f'<div class="foot-stamp">Generated {generated} · decision-support, not investment advice.</div>'
        f"</footer>"
    )

    body = f'<div class="wrap">{masthead}{summary_line}{new_top}{overview}{footer}</div>'

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Trading Model v3 · Factor Report</title>"
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        "family=Hanken+Grotesk:wght@400;500;600;700;800&"
        'family=IBM+Plex+Mono:wght@400;500;600&display=swap">'
        f"<style>{_stylesheet()}</style>"
        "</head>"
        f"<body>{body}</body></html>"
    )
