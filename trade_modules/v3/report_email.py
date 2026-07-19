"""Outlook-safe email edition of the v3 factor report.

``render_email_report`` consumes the SAME inputs as ``report.render_report``
(``scores, meta, portfolio, actions, conditioning``) but emits table-based,
inline-styled HTML that survives the Outlook / OWA sanitizer and renders
faithfully *inside the email body*:

* no ``<style>``-block reliance (every style is inline; the head block is a
  progressive-enhancement bonus only),
* no CSS custom properties (``var(--x)`` -> literal hex),
* no ``<details>``/``<summary>`` (all content is always visible),
* no flex / grid (layout is ``<table role="presentation">``),
* no SVG / web fonts / animations (bars are nested table cells; font is a
  web-safe stack).

Actionable names (BUY / ADD / TRIM / SELL) render as rich per-stock cards with a
six-cluster factor grid (Value, Quality, Growth, Momentum, Analyst, Risk), each
showing the cluster z-score plus its key underlying metrics; HOLD names render as
compact maintain-rows. The rich browser report (``report.render_report``) is
unchanged.
"""

from __future__ import annotations

import html as _htmllib
import math

# --- Palette (literal; Outlook has no CSS variables) ------------------------
INK = "#16181d"
INK2 = "#4a4e57"
MUTED = "#8a9099"
LINE = "#e4e3df"
WARM = "#faf9f7"
CANVAS = "#ffffff"
ACCENT = "#123b3a"
TRACK = "#ecebe4"
BULL, BULL_FILL, BULL_BAR = "#2d6a4f", "#e9f5ee", "#bfe3cd"
BEAR, BEAR_FILL, BEAR_BAR = "#b3402f", "#fbeeec", "#f2c4bb"
WARN, WARN_FILL, WARN_BAR = "#916516", "#f8f1df", "#e6d6a8"

FONT = "Aptos,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
MONO = "'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace"

ACTION_META = {
    "BUY": (BULL, BULL_FILL, BULL_BAR, "New position"),
    "ADD": (BULL, BULL_FILL, BULL_BAR, "Increase"),
    "TRIM": (WARN, WARN_FILL, WARN_BAR, "Reduce"),
    "SELL": (BEAR, BEAR_FILL, BEAR_BAR, "Exit"),
    "HOLD": (INK2, WARM, TRACK, "Maintain"),
}
ACTION_ORDER = ["BUY", "ADD", "TRIM", "SELL", "HOLD"]
CONTENT_W = 680

# Per-stock factor grid: (cluster label, z-column or None, [(metric_key, short label), ...]).
CARD_FACTORS = [
    ("Value", "value_z", [("pe_forward", "P/E"), ("pb", "P/B"), ("ps_sector", "P/S")]),
    ("Quality", "quality_z", [("roe", "ROE"), ("fcf", "FCF"), ("gp_assets", "GP/A")]),
    ("PEAD", "pead_z", [("sue", "SUE")]),
    ("Growth", "growth_z", [("earn_growth", "EPS"), ("rev_growth", "rev")]),
    (
        "Momentum",
        "momentum_z",
        [("mom_12_1", "12-1"), ("pct_52w_high", "52w"), ("price_perf", "12m")],
    ),
    (
        "Strength",
        "strength_z",
        [("analyst_mom", "revis"), ("upside", "upside"), ("buy_pct", "buy")],
    ),
    ("Risk", "lowvol_z", [("beta", "β"), ("realized_vol", "vol"), ("de", "D/E")]),
]

# metric formatting: key -> (kind, decimals, signed). kind: fracpct (x100), pct
# (already a percent), ratio (plain number). Scales verified against live scores.
_METRIC_FMT = {
    "mom_12_1": ("fracpct", 0, True),
    "gp_assets": ("fracpct", 0, False),  # gross profit / assets
    "sue": ("ratio", 2, True),  # standardized unexpected earnings (z-like)
    "pct_52w_high": ("pct", 0, False),
    "price_perf": ("pct", 0, True),
    "pe_forward": ("ratio", 1, False),
    "pb": ("ratio", 2, False),
    "ps_sector": ("ratio", 2, False),
    "roe": ("pct", 1, False),
    "op_margin": ("fracpct", 0, False),
    "fcf": ("pct", 1, False),
    "earn_growth": ("pct", 0, True),
    "rev_growth": ("fracpct", 0, True),
    "upside": ("pct", 1, True),
    "buy_pct": ("pct", 0, False),
    "analyst_mom": ("ratio", 0, True),
    "beta": ("ratio", 2, False),
    "realized_vol": ("fracpct", 0, False),
    "de": ("ratio", 1, False),
    "target_dispersion": ("fracpct", 0, False),
}


# --- Formatters -------------------------------------------------------------
def _esc(v) -> str:
    return _htmllib.escape(str(v), quote=True)


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _money(v) -> str:
    return "n/a" if _nan(v) else f"${v:,.0f}"


def _smoney(v) -> str:
    if _nan(v):
        return "n/a"
    return f"+${v:,.0f}" if v >= 0 else f"-${abs(v):,.0f}"


def _pct1(v) -> str:  # fraction -> "8.8%"
    return "n/a" if _nan(v) else f"{v * 100:.1f}%"


def _spct(v) -> str:  # already-percent -> "+7.85%"
    return "n/a" if _nan(v) else f"{v:+.2f}%"


def _pp(v) -> str:  # fraction delta -> "-2.5pp"
    return "n/a" if _nan(v) else f"{v * 100:+.1f}pp"


def _num(v, d=2) -> str:
    return "n/a" if _nan(v) else f"{v:.{d}f}"


def _mfmt(key: str, v) -> str:
    """Format a raw factor metric for the card grid (compact, readable units)."""
    if _nan(v):
        return "n/a"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return _esc(v)
    kind, dp, signed = _METRIC_FMT.get(key, ("ratio", 2, False))
    x = v * 100 if kind == "fracpct" else v
    body = f"{x:+.{dp}f}" if signed else f"{x:.{dp}f}"
    return body + ("%" if kind in ("fracpct", "pct") else "")


def _tone(v) -> str:
    if _nan(v):
        return INK2
    return BULL if v > 0 else (BEAR if v < 0 else INK2)


def _get(row, key):
    try:
        val = row[key]
    except Exception:
        return None
    return None if _nan(val) else val


def _full_name(desc, fallback: str) -> str:
    """Derive the full company name from the yfinance description prefix.

    The eToro NAME field is truncated to ~14 chars ("MICRON TECHNOL"), but the
    description reliably starts with the legal name followed by a lowercase
    verb/article ("Micron Technology, Inc. designs...", "Tencent Holdings
    Limited, an investment..."). Keep leading name-like tokens, stop at the first
    lowercase word, and fall back to the truncated eToro name when there is no
    description.
    """
    desc = (desc or "").strip() if isinstance(desc, str) else ""
    if not desc:
        return fallback
    out: list[str] = []
    for i, tok in enumerate(desc.split()):
        if i > 0 and tok[:1].islower():
            break
        out.append(tok)
        if len(out) >= 8:
            break
    name = " ".join(out).rstrip(" ,.;:")
    return name if len(name) >= 2 else fallback


def _desc_block(desc) -> str:
    """Company business description, shown as the last element of a card."""
    desc = desc.strip() if isinstance(desc, str) else ""
    if not desc:
        return ""
    if len(desc) > 260:
        desc = desc[:260].rsplit(" ", 1)[0].rstrip(" ,.;:") + "…"
    return (
        f'<div style="border-top:1px solid {LINE};margin-top:10px;padding-top:8px;'
        f'font-family:{FONT};font-size:11px;color:{INK2};line-height:1.55;">{_esc(desc)}</div>'
    )


# --- Primitive components ---------------------------------------------------
def _bar(frac: float, fill: str, track: str = TRACK, h: int = 7) -> str:
    """A horizontal bar as a two-cell table (Outlook renders cell bgcolor)."""
    pct = max(0.0, min(1.0, float(frac or 0.0))) * 100.0
    left = f'<td width="{pct:.1f}%" bgcolor="{fill}" style="background:{fill};height:{h}px;line-height:{h}px;font-size:0;">&nbsp;</td>'
    right = (
        f'<td bgcolor="{track}" style="background:{track};height:{h}px;line-height:{h}px;font-size:0;">&nbsp;</td>'
        if pct < 100
        else ""
    )
    return (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="border-collapse:collapse;table-layout:fixed;"><tr>{left}{right}</tr></table>'
    )


def _tiles(items: list[tuple], per_row: int = 4) -> str:
    """Grid of stat tiles as fixed-column tables. items: (label, value, sub, tone)."""
    out = []
    for i in range(0, len(items), per_row):
        chunk = items[i : i + per_row]
        w = 100.0 / per_row
        cells = []
        for label, value, sub, tone in chunk:
            cells.append(
                f'<td width="{w:.4f}%" valign="top" style="padding:0 6px 12px 0;">'
                f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
                f'style="border-collapse:separate;"><tr><td bgcolor="{CANVAS}" '
                f'style="background:{CANVAS};border:1px solid {LINE};border-radius:8px;padding:11px 12px;">'
                f'<div style="font-family:{FONT};font-size:9.5px;font-weight:700;letter-spacing:.7px;'
                f'text-transform:uppercase;color:{MUTED};">{_esc(label)}</div>'
                f'<div style="font-family:{FONT};font-size:19px;font-weight:700;color:{tone};'
                f'line-height:1.15;margin-top:5px;">{value}</div>'
                + (
                    f'<div style="font-family:{FONT};font-size:10.5px;color:{MUTED};margin-top:3px;">{sub}</div>'
                    if sub
                    else ""
                )
                + "</td></tr></table></td>"
            )
        for _ in range(per_row - len(chunk)):
            cells.append(f'<td width="{w:.4f}%" style="padding:0 6px 12px 0;">&nbsp;</td>')
        out.append(
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
            f'style="border-collapse:collapse;"><tr>{"".join(cells)}</tr></table>'
        )
    return "".join(out)


def _section_head(numeral: str, title: str, sub: str) -> str:
    return (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        'style="border-collapse:collapse;margin:0;"><tr>'
        f'<td width="34" valign="top" style="font-family:{MONO};font-size:12px;font-weight:700;'
        f'color:{ACCENT};padding:0 10px 0 0;">{numeral}</td>'
        f'<td valign="top"><div style="font-family:{FONT};font-size:16px;font-weight:700;color:{INK};'
        f'line-height:1.2;">{_esc(title)}</div>'
        f'<div style="font-family:{FONT};font-size:11.5px;color:{MUTED};margin-top:2px;">{_esc(sub)}</div></td>'
        "</tr></table>"
    )


def _rule(space_top: int = 22, space_bot: int = 14) -> str:
    return (
        f'<div style="border-top:1px solid {LINE};font-size:0;line-height:0;'
        f'margin:{space_top}px 0 {space_bot}px 0;">&nbsp;</div>'
    )


# --- Allocation bars --------------------------------------------------------
def _alloc_group(title: str, data: dict | None, color: str) -> str:
    if not data:
        return ""
    rows = []
    for name, frac in sorted(data.items(), key=lambda kv: -kv[1]):
        rows.append(
            "<tr>"
            f'<td width="150" valign="middle" style="font-family:{FONT};font-size:11.5px;color:{INK2};'
            f'padding:5px 10px 5px 0;white-space:nowrap;">{_esc(name)}</td>'
            f'<td valign="middle" style="padding:5px 10px 5px 0;">{_bar(frac, color)}</td>'
            f'<td width="46" align="right" valign="middle" style="font-family:{MONO};font-size:11px;'
            f'color:{INK};font-weight:700;padding:5px 0;">{_pct1(frac)}</td>'
            "</tr>"
        )
    return (
        f'<div style="font-family:{FONT};font-size:10px;font-weight:700;letter-spacing:.6px;'
        f'text-transform:uppercase;color:{MUTED};margin:0 0 5px 0;">{_esc(title)}</div>'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="border-collapse:collapse;margin-bottom:16px;">{"".join(rows)}</table>'
    )


# --- Per-stock factor grid --------------------------------------------------
def _factor_cell(label: str, z_col: str | None, metrics: list, row) -> str:
    z = _get(row, z_col) if z_col else None
    zbadge = (
        f'&nbsp;<span style="font-family:{MONO};font-size:10px;font-weight:700;color:{_tone(z)};">{z:+.2f}</span>'
        if z is not None
        else ""
    )
    sep = f' <span style="color:{MUTED};">·</span> '
    chips = sep.join(
        f'<span style="color:{MUTED};">{_esc(lbl)}</span> '
        f'<span style="color:{INK};font-weight:700;">{_mfmt(k, _get(row, k))}</span>'
        for k, lbl in metrics
    )
    return (
        f'<td width="50%" valign="top" style="padding:6px 12px 6px 0;">'
        f'<div style="font-family:{FONT};font-size:9.5px;font-weight:700;letter-spacing:.5px;'
        f'text-transform:uppercase;color:{ACCENT};">{_esc(label)}{zbadge}</div>'
        f'<div style="font-family:{MONO};font-size:10.5px;line-height:1.5;margin-top:2px;">{chips}</div>'
        "</td>"
    )


def _factor_grid(row) -> str:
    if row is None:
        return ""
    cells = [_factor_cell(lbl, zc, ms, row) for lbl, zc, ms in CARD_FACTORS]
    rows = []
    for i in range(0, len(cells), 2):
        pair = cells[i : i + 2]
        if len(pair) == 1:
            pair.append('<td width="50%">&nbsp;</td>')
        rows.append("<tr>" + "".join(pair) + "</tr>")
    return (
        f'<div style="border-top:1px solid {LINE};margin:10px 0 0 0;padding-top:8px;">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="border-collapse:collapse;">{"".join(rows)}</table></div>'
    )


# --- Action header cells (shared by card + compact row) ---------------------
def _conv_cell(conv, cmax: float) -> str:
    cbar_frac = min(1.0, abs(conv or 0) / cmax) if cmax else 0.0
    return (
        f'<div style="font-family:{MONO};font-size:14px;font-weight:700;color:{_tone(conv)};">'
        f"{(conv if conv is not None else 0):+.2f}</div>"
        f'<div style="margin-top:4px;">{_bar(cbar_frac, _tone(conv), TRACK, 5)}</div>'
    )


def _move_cell(a: dict, color: str) -> str:
    act = a.get("action")
    cur, tgt, dlt, price = (
        a.get("current_pct"),
        a.get("target_pct"),
        a.get("delta_pct"),
        a.get("price"),
    )
    if act == "BUY":
        move = f'<span style="color:{BULL};font-weight:700;">new {_pct1(tgt)}</span>'
    elif act == "SELL":
        move = f'<span style="color:{BEAR};font-weight:700;">exit</span> <span style="color:{MUTED};">from {_pct1(cur)}</span>'
    elif act == "HOLD":
        move = f'<span style="color:{INK2};font-weight:700;">hold {_pct1(cur)}</span>'
    else:
        move = (
            f'<span style="color:{INK2};">{_pct1(cur)}</span> <span style="color:{MUTED};">&rarr;</span> '
            f'<span style="color:{INK};font-weight:700;">{_pct1(tgt)}</span> '
            f'<span style="color:{color};font-weight:700;">{_pp(dlt)}</span>'
        )
    return (
        f'<div style="font-family:{FONT};font-size:12px;line-height:1.4;">{move}</div>'
        f'<div style="font-family:{MONO};font-size:10px;color:{MUTED};margin-top:2px;">@ {_money(price)}</div>'
    )


def _pl_cell(a: dict) -> str:
    pnl, pnl_pct = a.get("pnl"), a.get("pnl_pct")
    if pnl is not None:
        return (
            f'<div style="font-family:{MONO};font-size:12px;font-weight:700;color:{_tone(pnl)};">{_smoney(pnl)}</div>'
            f'<div style="font-family:{MONO};font-size:10.5px;color:{_tone(pnl)};margin-top:2px;">{_spct(pnl_pct)}</div>'
        )
    du = a.get("delta_usd")
    return (
        f'<div style="font-family:{MONO};font-size:11px;color:{MUTED};">trade</div>'
        f'<div style="font-family:{MONO};font-size:12px;font-weight:700;color:{INK2};margin-top:2px;">'
        f"{_money(abs(du) if du is not None else None)}</div>"
    )


def _name_cell(a: dict) -> str:
    nm = a.get("_full_name") or a.get("name", "")
    return (
        f'<div style="font-family:{FONT};font-size:13px;font-weight:700;color:{INK};">{_esc(a.get("ticker", ""))}</div>'
        f'<div style="font-family:{FONT};font-size:11px;color:{INK2};margin-top:1px;line-height:1.3;">{_esc(nm)}</div>'
        f'<div style="font-family:{FONT};font-size:10px;color:{MUTED};margin-top:2px;">{_esc(a.get("sector", ""))}</div>'
    )


def _header_row(a: dict, color: str, cmax: float) -> str:
    return (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        'style="border-collapse:collapse;"><tr>'
        f'<td width="228" valign="top" style="padding:0 10px 0 0;">{_name_cell(a)}</td>'
        f'<td width="88" valign="top" style="padding:0 10px 0 0;">{_conv_cell(a.get("conviction"), cmax)}</td>'
        f'<td valign="top" style="padding:0 10px 0 0;">{_move_cell(a, color)}</td>'
        f'<td width="104" valign="top" align="right">{_pl_cell(a)}</td>'
        "</tr></table>"
    )


def _action_card(a: dict, row, color: str, cmax: float) -> str:
    """Rich per-stock card: header + six-cluster factor grid."""
    return (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        'style="border-collapse:separate;margin-bottom:10px;"><tr>'
        f'<td bgcolor="{CANVAS}" style="background:{CANVAS};border:1px solid {LINE};'
        f'border-left:3px solid {color};border-radius:0 9px 9px 0;padding:12px 14px;">'
        f"{_header_row(a, color, cmax)}{_factor_grid(row)}{_desc_block(a.get('_desc'))}"
        "</td></tr></table>"
    )


def _hold_row(a: dict, cmax: float) -> str:
    """Compact maintain-row (no factor grid)."""
    return (
        f'<tr><td style="border-top:1px solid {LINE};padding:9px 0;">'
        f"{_header_row(a, INK2, cmax)}</td></tr>"
    )


def _group_header(act: str, color: str, fill: str, verb: str, n: int) -> str:
    return (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        'style="border-collapse:collapse;margin-bottom:6px;"><tr>'
        f'<td bgcolor="{fill}" style="background:{fill};border-left:3px solid {color};'
        f'padding:7px 12px;border-radius:0 6px 6px 0;">'
        f'<span style="font-family:{FONT};font-size:12px;font-weight:800;letter-spacing:.5px;'
        f'text-transform:uppercase;color:{color};">{act}</span> '
        f'<span style="font-family:{FONT};font-size:11px;color:{INK2};">&nbsp;{verb} &middot; {n} name'
        f"{'s' if n != 1 else ''}</span></td></tr></table>"
    )


def _action_group(act: str, rows: list, cmax: float, scores) -> str:
    if not rows:
        return ""
    color, fill, _bar_c, verb = ACTION_META[act]
    header = _group_header(act, color, fill, verb, len(rows))
    if act == "HOLD":
        body = "".join(_hold_row(a, cmax) for a in rows)
        body = (
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
            f'style="border-collapse:collapse;margin-bottom:18px;">{body}</table>'
        )
    else:

        def _row_of(a):
            tkr = a.get("ticker")
            try:
                return scores.loc[tkr] if (scores is not None and tkr in scores.index) else None
            except Exception:
                return None

        body = "".join(_action_card(a, _row_of(a), color, cmax) for a in rows)
        body += '<div style="height:8px;font-size:0;">&nbsp;</div>'
    return header + body


# --- Main -------------------------------------------------------------------
def render_email_report(
    scores,
    meta: dict,
    *,
    portfolio: dict | None = None,
    actions: list | None = None,
    conditioning: dict | None = None,
) -> str:
    meta = meta or {}
    actions = actions or []
    portfolio = portfolio or {}

    # Enrich each action with the full company name + business description,
    # resolved from the scores row (the eToro NAME field is truncated).
    for a in actions:
        t = a.get("ticker")
        row = None
        try:
            if scores is not None and t in scores.index:
                row = scores.loc[t]
        except Exception:
            row = None
        desc = _get(row, "description") if row is not None else None
        a["_desc"] = desc
        a["_full_name"] = _full_name(desc, a.get("name", ""))
    diag = (portfolio.get("diagnostics") or {}).get("gate") or {}
    acct = meta.get("account") or {}
    soc = meta.get("social") or {}
    cov = meta.get("coverage") or {}
    caps = meta.get("caps") or {}
    alloc = meta.get("allocations") or {}

    by_action: dict[str, list] = {k: [] for k in ACTION_ORDER}
    for a in actions:
        by_action.setdefault(a.get("action"), []).append(a)
    cmax = max((abs(a.get("conviction") or 0.0) for a in actions), default=3.0) or 3.0
    counts = {k: len(v) for k, v in by_action.items()}

    regime = str(meta.get("regime", "")).upper()
    date = _esc(meta.get("date", ""))
    gen = _esc(meta.get("generated_utc", ""))

    # ---- Masthead ----
    masthead = (
        f'<div style="font-family:{FONT};font-size:10.5px;font-weight:700;letter-spacing:1.4px;'
        f'text-transform:uppercase;color:{ACCENT};">Trading Model v3</div>'
        f'<div style="font-family:{FONT};font-size:29px;font-weight:800;color:{INK};'
        f'line-height:1.08;margin-top:4px;">Factor Snapshot</div>'
        f'<div style="border-top:2px solid {INK};width:44px;font-size:0;line-height:0;margin:14px 0;">&nbsp;</div>'
        f'<div style="font-family:{FONT};font-size:12px;color:{INK2};line-height:1.6;">'
        f"A daily overlay on your live eToro book: keep what clears the bar, sell the genuinely weak, "
        f"buy the strongest non-held names, then risk-gate the whole book. Each traded name carries its "
        f"six-cluster factor breakdown.</div>"
        f'<div style="margin-top:12px;">'
        f'<span style="font-family:{MONO};font-size:11px;color:{INK};background:{WARM};'
        f'border:1px solid {LINE};border-radius:20px;padding:4px 11px;">Regime <b>{_esc(regime)}</b></span>'
        f'&nbsp;<span style="font-family:{MONO};font-size:11px;color:{INK2};background:{WARM};'
        f'border:1px solid {LINE};border-radius:20px;padding:4px 11px;">'
        f"{counts.get('BUY', 0)} buy &middot; {counts.get('ADD', 0) + counts.get('TRIM', 0) + counts.get('HOLD', 0)} keep &middot; {counts.get('SELL', 0)} sell</span>"
        f'&nbsp;<span style="font-family:{MONO};font-size:11px;color:{MUTED};">{gen}</span></div>'
    )

    # ---- Section I: account ----
    pl = acct.get("unrealized_pnl")
    acct_tiles = _tiles(
        [
            ("Total equity", _money(acct.get("total_equity")), "live eToro", INK),
            ("Unrealized P/L", _smoney(pl), _spct(acct.get("profit_pct")), _tone(pl)),
            ("Invested cost", _money(acct.get("invested_cost")), "", INK),
            ("Available", _money(acct.get("available")), "cash", INK),
        ],
        per_row=4,
    )
    soc_tiles = ""
    if soc:
        soc_tiles = _tiles(
            [
                (
                    "Copiers",
                    f"{soc.get('copiers', 'n/a')}",
                    _spct(soc.get("copiers_gain_pct")),
                    _tone(soc.get("copiers_gain_pct")),
                ),
                (
                    "Win ratio",
                    f"{_num(soc.get('win_ratio'), 1)}%",
                    "YTD trades " + str(soc.get("trades_ytd", "n/a")),
                    INK,
                ),
                (
                    "Gain YTD",
                    _spct(soc.get("gain_ytd")),
                    "MTD " + _spct(soc.get("gain_mtd")),
                    _tone(soc.get("gain_ytd")),
                ),
                (
                    "Risk score",
                    f"{soc.get('risk_score', 'n/a')}",
                    f"max {soc.get('max_daily_risk', 'n/a')} &middot; {_esc(soc.get('aum_tier_desc', ''))}",
                    INK,
                ),
            ],
            per_row=4,
        )

    # ---- Positioning / risk ----
    usd_bloc = diag.get("usd_bloc")
    usd_cap = caps.get("usd_bloc")
    usd_breach = usd_bloc is not None and usd_cap is not None and usd_bloc > usd_cap + 0.005
    pos_tiles = _tiles(
        [
            (
                "Deployed",
                _pct1(diag.get("gross_after")),
                f"target {_pct1(conditioning.get('final_deployment') if conditioning else None)}",
                INK,
            ),
            (
                "Portfolio vol",
                _pct1(diag.get("vol_after")),
                f"ceiling {_pct1(diag.get('vol_ceiling'))}",
                INK,
            ),
            ("CVaR-95", _pct1(diag.get("cvar_after")), "deployed book", INK),
            ("Net beta", _num(diag.get("net_beta")), "band 0.3-1.1", INK),
            (
                "Effective bets",
                _num(diag.get("effective_bets"), 0),
                f"min {_num(diag.get('min_effective_bets'), 0)}",
                INK,
            ),
            (
                "USD-bloc",
                _pct1(usd_bloc),
                (
                    f"cap {_pct1(usd_cap)} &middot; BREACH"
                    if usd_breach
                    else f"cap {_pct1(usd_cap)}"
                ),
                BEAR if usd_breach else INK,
            ),
        ],
        per_row=3,
    )

    # ---- Section III: decisions ----
    groups = "".join(
        _action_group(act, by_action.get(act, []), cmax, scores) for act in ACTION_ORDER
    )

    # ---- Section IV: allocations ----
    alloc_html = (
        _alloc_group("Geography", alloc.get("geography"), ACCENT)
        + _alloc_group("Asset type", alloc.get("asset_type"), BULL)
        + _alloc_group("Sector", alloc.get("sector"), INK2)
    )

    # ---- Footer ----
    cov_pct = cov.get("pct")
    footer = (
        f'<div style="font-family:{FONT};font-size:10.5px;color:{MUTED};line-height:1.7;">'
        f"Universe {cov.get('n_scored', 'n/a')} scored &middot; {cov.get('n_eligible', 'n/a')} eligible"
        f"{(' (' + _pct1(cov_pct) + ' coverage)') if cov_pct is not None else ''}"
        f" &middot; caps name {_pct1(caps.get('name'))} / sector {_pct1(caps.get('sector'))} / USD-bloc {_pct1(caps.get('usd_bloc'))}.<br>"
        f"Factor grid z-scores are the rank-normalized cluster contributions (value, quality, momentum, growth, "
        f"low-vol, strength); metrics are the underlying raw values. ERC + conviction tilt, hard risk gate. "
        f"Decision-support, not advice. Generated {gen}."
        "</div>"
    )

    body = (
        masthead
        + _rule(26, 16)
        + _section_head("I", "Account", "live equity, P/L and standing")
        + '<div style="height:12px;font-size:0;">&nbsp;</div>'
        + acct_tiles
        + soc_tiles
        + _rule()
        + _section_head("II", "Positioning & risk", "the gated book vs its limits")
        + '<div style="height:12px;font-size:0;">&nbsp;</div>'
        + pos_tiles
        + _rule()
        + _section_head("III", "Decisions", f"{len(actions)} names · factor cards for every trade")
        + '<div style="height:14px;font-size:0;">&nbsp;</div>'
        + groups
        + _rule()
        + _section_head("IV", "Allocation", "current book by geography, asset, sector")
        + '<div style="height:12px;font-size:0;">&nbsp;</div>'
        + alloc_html
        + _rule()
        + footer
    )

    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Trading Model v3 · Factor Snapshot</title></head>"
        f'<body style="margin:0;padding:0;background:{WARM};">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="border-collapse:collapse;background:{WARM};"><tr><td align="center" style="padding:22px 12px;">'
        f'<table role="presentation" width="{CONTENT_W}" cellpadding="0" cellspacing="0" border="0" '
        f'style="border-collapse:collapse;width:{CONTENT_W}px;max-width:100%;">'
        f'<tr><td bgcolor="{CANVAS}" style="background:{CANVAS};border:1px solid {LINE};'
        f'border-radius:14px;padding:30px 32px;">{body}</td></tr>'
        f'<tr><td style="padding:14px 4px;font-family:{FONT};font-size:10px;color:{MUTED};text-align:center;">'
        f"Trading Model v3 &middot; generated on the VPS &middot; {date}</td></tr>"
        "</table></td></tr></table></body></html>"
    )
