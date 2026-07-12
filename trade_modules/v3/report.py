"""v3 factor report renderer — house Swiss style, Outlook-safe HTML.

``render_report(scores, meta) -> str`` returns a full standalone HTML
document.  All multi-column layout uses ``<table>`` (no flex/gap, no
border-radius) so it renders in Outlook as well as browsers.

``compute_regime(index_close) -> (label, detail)`` is a pure helper the
driver script feeds with a fetched index series; kept network-free here.
"""

from __future__ import annotations

import html as _html

import numpy as np
import pandas as pd

# --- Swiss palette / type tokens -------------------------------------------
BG = "#ffffff"
BG_ALT = "#fafafa"
INK = "#1a1a1a"
TXT = "#333333"
MUTE = "#888888"
FAINT = "#999999"
LINE = "#e5e5e5"
BULL = "#2d6a4f"
BULL_BG = "#ecfdf5"
BEAR = "#c0392b"
BEAR_BG = "#fdf2f2"
FONT = "'Helvetica Neue',Helvetica,Arial,sans-serif"

_EYEBROW = (
    f"font-size:10px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:{MUTE};"
)

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

CLUSTER_LABELS = [
    ("value_z", "Val"),
    ("quality_z", "Qual"),
    ("momentum_z", "Mom"),
    ("lowvol_z", "LoVol"),
    ("strength_z", "Str"),
]


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
        return "—"
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
        return FAINT
    return BULL if z > 0 else (BEAR if z < 0 else FAINT)


def _z_cell(z) -> str:
    if _isnan(z):
        return f'<span style="color:{FAINT};">·</span>'
    return f'<span style="color:{_z_color(z)};font-weight:600;">{z:+.2f}</span>'


def _tilt(conv: float, is_portfolio: bool) -> tuple[str, str, str]:
    """Return (label, text_color, bg_color) for the tilt badge."""
    if _isnan(conv):
        return ("n/a", FAINT, BG_ALT)
    if is_portfolio:
        if conv > 0.5:
            return ("ADD", BULL, BULL_BG)
        if conv < -0.5:
            return ("TRIM", BEAR, BEAR_BG)
        return ("HOLD", TXT, BG_ALT)
    if conv > 0.5:
        return ("BUY-WATCH", BULL, BULL_BG)
    if conv < -0.5:
        return ("PASS", BEAR, BEAR_BG)
    return ("WATCH", TXT, BG_ALT)


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
def _group_block(row: pd.Series, cols, group_label: str, metrics) -> str:
    rows_html = []
    for key, label in metrics:
        raw = row.get(key, np.nan) if key in cols else np.nan
        zkey = f"{key}_z"
        has_z = zkey in cols
        z_html = (
            _z_cell(row.get(zkey, np.nan)) if has_z else f'<span style="color:{FAINT};">·</span>'
        )
        rows_html.append(
            f"<tr>"
            f'<td style="padding:2px 8px 2px 0;color:{TXT};font-size:11px;white-space:nowrap;">{label}</td>'
            f'<td style="padding:2px 8px;color:{INK};font-size:11px;font-weight:600;text-align:right;white-space:nowrap;">{_fmt(key, raw)}</td>'
            f'<td style="padding:2px 0 2px 8px;font-size:11px;text-align:right;white-space:nowrap;">{z_html}</td>'
            f"</tr>"
        )
    return (
        f'<div style="{_EYEBROW}margin:0 0 4px 0;color:{FAINT};">{group_label}</div>'
        f'<table cellpadding="0" cellspacing="0" style="width:100%;border-collapse:collapse;margin-bottom:14px;">'
        f"{''.join(rows_html)}</table>"
    )


def _card(tkr: str, row: pd.Series, cols) -> str:
    conv = row.get("conviction", np.nan)
    rank = row.get("rank", np.nan)
    is_port = bool(row.get("is_portfolio", False))
    name = _esc(row.get("name", ""))
    sector = _esc(row.get("sector", "")) or "—"
    price = _fmt("price", row.get("price", np.nan))
    conv_color = _z_color(conv)
    conv_txt = "—" if _isnan(conv) else f"{conv:+.2f}"
    rank_txt = "—" if _isnan(rank) else f"#{int(rank)}"
    tilt_label, tilt_c, tilt_bg = _tilt(conv, is_port)

    # cluster-z strip
    chips = []
    for ckey, clabel in CLUSTER_LABELS:
        cz = row.get(ckey, np.nan)
        chips.append(
            f'<td style="padding:0 10px 0 0;white-space:nowrap;">'
            f'<span style="{_EYEBROW}color:{FAINT};">{clabel}</span> '
            f'<span style="font-size:12px;font-weight:600;color:{_z_color(cz)};">'
            f"{'·' if _isnan(cz) else f'{cz:+.2f}'}</span></td>"
        )
    cluster_strip = (
        f'<table cellpadding="0" cellspacing="0" style="border-collapse:collapse;margin-top:10px;"><tr>'
        f"{''.join(chips)}</tr></table>"
    )

    # header (two columns via table): identity left, verdict right
    header = (
        f'<table cellpadding="0" cellspacing="0" style="width:100%;border-collapse:collapse;">'
        f"<tr>"
        f'<td style="vertical-align:top;">'
        f'<span style="font-size:20px;font-weight:600;color:{INK};letter-spacing:-0.3px;">{_esc(tkr)}</span>'
        f'<span style="font-size:12px;color:{FAINT};">&nbsp;&nbsp;{price}</span>'
        f'<div style="font-size:12px;color:{TXT};margin-top:2px;">{name}</div>'
        f'<div style="{_EYEBROW}margin-top:4px;">{sector}</div>'
        f"</td>"
        f'<td style="vertical-align:top;text-align:right;white-space:nowrap;">'
        f'<div style="font-size:28px;font-weight:300;color:{conv_color};line-height:1;">{conv_txt}</div>'
        f'<div style="{_EYEBROW}margin-top:2px;">Conviction {rank_txt}</div>'
        f'<div style="display:inline-block;margin-top:6px;padding:3px 10px;background:{tilt_bg};'
        f'color:{tilt_c};font-size:11px;font-weight:600;letter-spacing:1px;">{tilt_label}</div>'
        f"</td>"
        f"</tr></table>"
        f"{cluster_strip}"
    )

    # factor grid: 8 display groups laid two-per-row via a table
    blocks = [_group_block(row, cols, gl, ms) for gl, ms in DISPLAY_GROUPS]
    grid_rows = []
    for i in range(0, len(blocks), 2):
        left = blocks[i]
        right = blocks[i + 1] if i + 1 < len(blocks) else ""
        grid_rows.append(
            f"<tr>"
            f'<td style="width:50%;vertical-align:top;padding:0 20px 0 0;">{left}</td>'
            f'<td style="width:50%;vertical-align:top;padding:0 0 0 20px;">{right}</td>'
            f"</tr>"
        )
    grid = (
        f'<table cellpadding="0" cellspacing="0" style="width:100%;border-collapse:collapse;margin-top:18px;'
        f'border-top:1px solid {LINE};padding-top:12px;">{"".join(grid_rows)}</table>'
    )

    return (
        f'<table cellpadding="0" cellspacing="0" style="width:100%;border-collapse:collapse;'
        f'border:1px solid {LINE};background:{BG};margin:0 0 16px 0;">'
        f'<tr><td style="padding:20px 22px;">{header}{grid}</td></tr></table>'
    )


def _section_header(label: str, sub: str) -> str:
    return (
        f'<div style="padding:28px 0 12px 0;border-bottom:1px solid {LINE};margin-bottom:18px;">'
        f'<div style="{_EYEBROW}">{label}</div>'
        f'<div style="color:{FAINT};font-size:11px;margin-top:2px;">{sub}</div></div>'
    )


def _signed(v) -> str:
    """'·' when NaN, else a signed 2dp string (for compact table cells)."""
    return "·" if _isnan(v) else f"{v:+.2f}"


def _summary_table(scores: pd.DataFrame) -> str:
    ordered = scores.sort_values("conviction", ascending=False)
    top = ordered.head(5)
    bottom = ordered.tail(5)

    def _rows(frame):
        out = []
        for tkr, r in frame.iterrows():
            conv = r.get("conviction", np.nan)
            rank = r.get("rank", np.nan)
            conv_s = "—" if _isnan(conv) else f"{conv:+.2f}"
            rank_s = "—" if _isnan(rank) else f"#{int(rank)}"
            sector_s = _esc(r.get("sector", "")) or "—"
            cluster_tds = "".join(
                f'<td style="padding:5px 8px;text-align:right;font-size:11px;'
                f'color:{_z_color(r.get(ck, np.nan))};">{_signed(r.get(ck, np.nan))}</td>'
                for ck, _ in CLUSTER_LABELS
            )
            out.append(
                f"<tr>"
                f'<td style="padding:5px 8px;font-size:12px;font-weight:600;color:{INK};">{_esc(tkr)}</td>'
                f'<td style="padding:5px 8px;font-size:11px;color:{FAINT};">{sector_s}</td>'
                f'<td style="padding:5px 8px;text-align:right;font-size:12px;font-weight:600;'
                f'color:{_z_color(conv)};">{conv_s}</td>'
                f'<td style="padding:5px 8px;text-align:right;font-size:11px;color:{TXT};">{rank_s}</td>'
                f"{cluster_tds}</tr>"
            )
        return "".join(out)

    head = (
        f'<tr style="border-bottom:1px solid {LINE};">'
        f'<td style="{_EYEBROW}padding:0 8px 6px 8px;">Ticker</td>'
        f'<td style="{_EYEBROW}padding:0 8px 6px 8px;">Sector</td>'
        f'<td style="{_EYEBROW}padding:0 8px 6px 8px;text-align:right;">Conv</td>'
        f'<td style="{_EYEBROW}padding:0 8px 6px 8px;text-align:right;">Rank</td>'
        + "".join(
            f'<td style="{_EYEBROW}padding:0 8px 6px 8px;text-align:right;">{lbl}</td>'
            for _, lbl in CLUSTER_LABELS
        )
        + "</tr>"
    )
    divider = (
        f'<tr><td colspan="9" style="{_EYEBROW}padding:12px 8px 4px 8px;color:{FAINT};">'
        f"Bottom 5</td></tr>"
    )
    return (
        f'<table cellpadding="0" cellspacing="0" style="width:100%;border-collapse:collapse;margin-bottom:8px;">'
        f"{head}{_rows(top)}{divider}{_rows(bottom)}</table>"
    )


def _cards_section(scores: pd.DataFrame, is_port: bool, label: str, sub: str) -> str:
    cols = scores.columns
    subset = scores[scores.get("is_portfolio", pd.Series(False, index=scores.index)) == is_port]
    subset = subset.sort_values("conviction", ascending=False)
    if subset.empty:
        body = f'<div style="color:{FAINT};font-size:12px;padding:8px 0;">No names in this group.</div>'
    else:
        body = "".join(_card(t, r, cols) for t, r in subset.iterrows())
    return _section_header(label, sub) + body


# --- top-level --------------------------------------------------------------
def render_report(scores: pd.DataFrame, meta: dict) -> str:
    meta = meta or {}
    n_port = meta.get("n_portfolio", int(scores.get("is_portfolio", pd.Series(dtype=bool)).sum()))
    n_cand = meta.get("n_candidates", 0)
    date = _esc(meta.get("date", ""))
    regime = _esc(meta.get("regime", "NEUTRAL"))
    regime_detail = _esc(meta.get("regime_detail", ""))
    system_read = _esc(meta.get("system_read", ""))
    priced = meta.get("priced", "")
    enriched = meta.get("enriched", "")
    generated = _esc(meta.get("generated_utc", ""))

    regime_color = {"RISK_ON": BULL, "RISK_OFF": BEAR}.get(str(meta.get("regime")), MUTE)

    header = (
        f'<div style="padding:48px 0 28px 0;border-bottom:2px solid {INK};">'
        f'<div style="{_EYEBROW}">Trading Model v3 · Factor Report</div>'
        f'<h1 style="margin:8px 0 0 0;font-size:32px;font-weight:300;color:{INK};letter-spacing:-0.5px;">'
        f"Factor Snapshot</h1>"
        f'<div style="margin-top:8px;font-size:13px;color:{MUTE};">{date} &nbsp;&middot;&nbsp; '
        f"Portfolio {n_port} &nbsp;&middot;&nbsp; Candidates {n_cand} &nbsp;&middot;&nbsp; "
        f"Priced {priced} / Enriched {enriched}</div>"
        f'<div style="margin-top:14px;font-size:13px;color:{TXT};">'
        f'<span style="{_EYEBROW}">Regime</span> &nbsp;'
        f'<span style="font-weight:600;color:{regime_color};">{regime}</span> '
        f'<span style="color:{FAINT};">&nbsp;{regime_detail}</span></div>'
        f'<div style="margin-top:8px;font-size:13px;color:{TXT};font-style:italic;">{system_read}</div>'
        f"</div>"
    )

    summary = _section_header(
        "Top & Bottom", "Ranked by conviction across the full universe"
    ) + _summary_table(scores)
    port_sec = _cards_section(
        scores, True, "Portfolio", "Current holdings · tilt = ADD / HOLD / TRIM"
    )
    cand_sec = _cards_section(
        scores, False, "Candidates", "BUY-signal watchlist · tilt = BUY-WATCH / WATCH / PASS"
    )

    footer = (
        f'<div style="padding:28px 0 48px 0;border-top:2px solid {INK};margin-top:24px;">'
        f'<div style="{_EYEBROW}margin-bottom:8px;">Methodology</div>'
        f'<div style="font-size:11px;color:{FAINT};line-height:1.6;">'
        f"Each metric is winsorized (1/99) and cross-sectionally z-scored; low-is-good "
        f"metrics (valuation, leverage, beta, realized-vol, short interest, target dispersion) "
        f"are negated so a high z is always attractive. Metric z-scores are sector-neutral "
        f"(demeaned within GICS sector). Five cluster z-scores (Value, Quality, Momentum, "
        f"Low-vol, Strength) combine into conviction with a near-equal weighting — Value + "
        f"Quality jointly capped at ~55% — renormalized per-row over the clusters actually "
        f"present. Conviction is re-z-scored cross-sectionally; rank 1 = best.<br><br>"
        f"This is a SHADOW / decision-support snapshot, not yet capital-authorized. The naive "
        f"price-spine (12-1 momentum + low-vol) showed no standalone out-of-sample edge in "
        f"validation, so treat single-factor tilts with caution and weigh the full-cluster "
        f"conviction over any one number.<br><br>"
        f"Generated {generated}. Not investment advice."
        f"</div></div>"
    )

    body = (
        f'<div class="container" style="max-width:880px;margin:0 auto;padding:0 48px;">'
        f"{header}{summary}{port_sec}{cand_sec}{footer}</div>"
    )

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Trading Model v3 · Factor Report</title>"
        "<style>@media screen and (max-width:768px){.container{padding:0 16px !important;}}</style>"
        "</head>"
        f'<body style="margin:0;padding:0;background:{BG};font-family:{FONT};color:{TXT};'
        f'line-height:1.55;-webkit-font-smoothing:antialiased;">'
        f"{body}</body></html>"
    )
