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

import pandas as pd

# Single-source the data model + metric formatting from the browser report so the two
# editions never drift — only the HTML rendering differs (this file stays email-safe).
from trade_modules.v3.report import (
    CLUSTER_CELLS,
    DISPLAY_GROUPS,
    GROUP_CLUSTER,
)
from trade_modules.v3.report import (
    _fmt as _bfmt,
)

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


# --- Formatters -------------------------------------------------------------
def _esc(v) -> str:
    return _htmllib.escape(str(v), quote=True)


def _nan(v) -> bool:
    # pd.isna catches None, float NaN, and pandas NAType (rank is an Int64 that can be NA);
    # guard the array case (never expected here) so a stray Series can't raise.
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return v is None


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
        f'<span style="color:{INK};font-weight:700;">{_bfmt(k, _get(row, k))}</span>'
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
    # Full info-parity with the browser report: every DISPLAY_GROUPS group + all its
    # metrics (not the old CARD_FACTORS subset), with the cluster z from GROUP_CLUSTER.
    cells = [_factor_cell(lbl, GROUP_CLUSTER.get(lbl), ms, row) for lbl, ms in DISPLAY_GROUPS]
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


def _levels_tiles(row) -> str:
    """Vol-scaled trade levels (entry / stop / target / R:R) as email tiles."""
    if row is None:
        return ""
    entry, stop, tp, rr = (
        _get(row, "entry"),
        _get(row, "stop_loss"),
        _get(row, "take_profit"),
        _get(row, "rr"),
    )
    if all(_nan(x) for x in (entry, stop, tp, rr)):
        return ""
    return (
        '<div style="margin-top:8px;">'
        + _tiles(
            [
                ("Entry", _money(entry), "", INK),
                ("Stop", _money(stop), "vol-scaled", BEAR),
                ("Target", _money(tp), "vol-scaled", BULL),
                ("R:R", _num(rr, 1), "reward/risk", INK),
            ],
            per_row=4,
        )
        + "</div>"
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
    """Rich per-stock card: header + trade levels + full factor grid + description."""
    return (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        'style="border-collapse:separate;margin-bottom:10px;"><tr>'
        f'<td bgcolor="{CANVAS}" style="background:{CANVAS};border:1px solid {LINE};'
        f'border-left:3px solid {color};border-radius:0 9px 9px 0;padding:12px 14px;">'
        f"{_header_row(a, color, cmax)}{_levels_tiles(row)}{_factor_grid(row)}"
        f"{_desc_block(a.get('_desc'))}"
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


# --- Conviction heatmap (email-safe: solid bgcolor cells, no rgba) ----------
def _heat_hex(z) -> str:
    """Solid hex tint for a heat cell — the browser's green/red rgba blended over
    white (Outlook's ``bgcolor`` needs a solid hex, not rgba)."""
    if _nan(z):
        return WARM
    a = min(abs(float(z)) / 2.0, 1.0)
    alpha = 0.10 + a * 0.50
    r, g, b = (45, 106, 79) if z >= 0 else (179, 64, 47)

    def _mix(c: int) -> int:
        return round(255 * (1 - alpha) + c * alpha)

    return f"#{_mix(r):02x}{_mix(g):02x}{_mix(b):02x}"


def _heat_z(z) -> str:
    return "·" if _nan(z) else f"{z:+.1f}"


def _heatmap_section(scores, max_rows: int = 120) -> str:
    """Ranked names × cluster z heat-strip — the browser overview, as an email table.
    Shows all holdings plus the top ``max_rows`` by conviction (email-deliverable size)."""
    if scores is None or "conviction" not in getattr(scores, "columns", []):
        return ""
    ranked = scores.sort_values("conviction", ascending=False, na_position="last")
    total = int(ranked["conviction"].notna().sum())
    if total == 0:
        return ""
    held = ranked.index[ranked.get("is_portfolio", pd.Series(False, index=ranked.index)) == True]  # noqa: E712
    keep = set(held) | set(ranked.head(max_rows).index)
    shown = ranked.loc[[i for i in ranked.index if i in keep]]

    def _th(txt: str, align: str = "center") -> str:
        return (
            f'<td align="{align}" style="font-family:{FONT};font-size:9px;font-weight:700;'
            f'letter-spacing:.3px;text-transform:uppercase;color:{MUTED};padding:0 3px 6px;">{txt}</td>'
        )

    header = (
        '<tr><td style="padding:0 8px 6px 0;">&nbsp;</td>'
        + "".join(_th(lbl) for _, lbl in CLUSTER_CELLS)
        + _th("Conv", "right")
        + _th("Rank", "right")
        + "</tr>"
    )
    body = []
    for tkr, r in shown.iterrows():
        conv, rank = r.get("conviction"), r.get("rank")
        pf = (
            f' <span style="color:{ACCENT};font-size:8px;font-weight:700;">PF</span>'
            if bool(r.get("is_portfolio"))
            else ""
        )
        cells = "".join(
            f'<td bgcolor="{_heat_hex(r.get(zc))}" align="center" style="font-family:{MONO};'
            f'font-size:9.5px;color:{INK};padding:4px 3px;border:2px solid {CANVAS};">{_heat_z(r.get(zc))}</td>'
            for zc, _ in CLUSTER_CELLS
        )
        body.append(
            f'<tr><td style="font-family:{MONO};font-size:10.5px;font-weight:700;color:{INK};'
            f'padding:4px 8px 4px 0;white-space:nowrap;">{_esc(str(tkr))}{pf}</td>{cells}'
            f'<td align="right" style="font-family:{MONO};font-size:10px;font-weight:700;'
            f'color:{_tone(conv)};padding:4px 0 4px 8px;">{"·" if _nan(conv) else f"{conv:+.2f}"}</td>'
            f'<td align="right" style="font-family:{MONO};font-size:9.5px;color:{MUTED};'
            f'padding:4px 0 4px 6px;">{"" if _nan(rank) else "#" + str(int(rank))}</td></tr>'
        )
    cap = f" &middot; showing {len(shown)} of {total}" if len(shown) < total else ""
    return (
        '<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="border-collapse:collapse;">{header}{"".join(body)}</table></div>'
        f'<div style="font-family:{FONT};font-size:10px;color:{MUTED};margin-top:6px;">'
        f"Cluster z per name &middot; green above-peer / red below{cap}.</div>"
    )


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
        f"buy the strongest non-held names, then risk-gate the whole book. Every scored name is ranked "
        f"in the conviction heatmap; each traded name carries its full factor breakdown + trade levels.</div>"
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
        f"Factor grid z-scores are the rank-normalized cluster contributions (value, quality, momentum, "
        f"PEAD, trajectory, low-vol, strength, growth); metrics are the underlying raw values. Each metric "
        f"is winsorized (1/99) and cross-sectionally z-scored; conviction re-z-scored, rank 1 = best. "
        f"ERC + conviction tilt, hard risk gate. Decision-support, not advice. Generated {gen}."
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
        + _section_head("III", "Conviction heatmap", "every scored name, ranked · cluster z-scores")
        + '<div style="height:12px;font-size:0;">&nbsp;</div>'
        + _heatmap_section(scores)
        + _rule()
        + _section_head(
            "IV", "Decisions", f"{len(actions)} names · full factor cards for every trade"
        )
        + '<div style="height:14px;font-size:0;">&nbsp;</div>'
        + groups
        + _rule()
        + _section_head("V", "Allocation", "current book by geography, asset, sector")
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


def render_summary(
    meta: dict,
    actions: list,
    *,
    portfolio: dict | None = None,
) -> str:
    """Compact Outlook-safe summary body for the scheduled 4h mail.

    Masthead + regime / buy-keep-sell counts / deployment / vol, then a
    one-line-per-name action table (BUY/ADD/TRIM/SELL). The FULL browser Factor
    Snapshot (heatmap, per-stock factor cards, trade levels) rides along as the
    attachment. Call after :func:`render_email_report` so each action already carries
    ``_full_name``; falls back to the eToro name / ticker otherwise.
    """
    meta = meta or {}
    actions = actions or []
    diag = ((portfolio or {}).get("diagnostics") or {}).get("gate") or {}
    by_action: dict[str, list] = {k: [] for k in ACTION_ORDER}
    for a in actions:
        by_action.setdefault(a.get("action"), []).append(a)
    c = {k: len(v) for k, v in by_action.items()}
    keep = c.get("ADD", 0) + c.get("TRIM", 0) + c.get("HOLD", 0)
    regime = _esc(str(meta.get("regime", "")).upper())
    gen = _esc(meta.get("generated_utc", ""))
    date = _esc(meta.get("date", ""))
    dep = _pct1(diag.get("gross_after"))
    vol = _pct1(diag.get("vol_after"))

    head = (
        f'<div style="font-family:{FONT};font-size:10.5px;font-weight:700;letter-spacing:1.4px;'
        f'text-transform:uppercase;color:{ACCENT};">Trading Model v3</div>'
        f'<div style="font-family:{FONT};font-size:25px;font-weight:800;color:{INK};'
        f'line-height:1.08;margin-top:3px;">Factor Snapshot</div>'
        f'<div style="border-top:2px solid {INK};width:44px;font-size:0;line-height:0;'
        f'margin:12px 0;">&nbsp;</div>'
        f'<div style="font-family:{FONT};font-size:12.5px;color:{INK2};line-height:1.7;">'
        f'Regime <b style="color:{INK};">{regime}</b> &middot; '
        f'<b style="color:{BULL};">{c.get("BUY", 0)}</b> buy &middot; {keep} keep &middot; '
        f'<b style="color:{BEAR};">{c.get("SELL", 0)}</b> sell &middot; '
        f'deployment <b style="color:{INK};">{dep}</b> &middot; vol {vol}</div>'
        f'<div style="font-family:{FONT};font-size:11px;color:{MUTED};margin-top:3px;">{gen}</div>'
    )

    def _th(label: str, align: str = "left", pr: str = "8px") -> str:
        return (
            f'<td style="padding:0 {pr} 7px 0;font-family:{FONT};font-size:10px;font-weight:700;'
            f"text-transform:uppercase;letter-spacing:.5px;color:{MUTED};"
            f'text-align:{align};">{label}</td>'
        )

    def _conv(a: dict) -> float:
        """Conviction as a float; NaN when absent/unparseable (renders as a dot)."""
        try:
            return float(a.get("conviction"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float("nan")

    def _sort_key(a: dict) -> float:
        v = _conv(a)
        return -abs(v) if v == v else 0.0  # NaN sorts last

    trs = []
    for act in ("BUY", "ADD", "TRIM", "SELL"):
        color = ACTION_META[act][0]
        for a in sorted(by_action.get(act, []), key=_sort_key):
            tkr = _esc(a.get("ticker", ""))
            nm = _esc(a.get("_full_name") or a.get("name") or a.get("ticker", ""))
            cur, tgt = _pct1(a.get("current_pct")), _pct1(a.get("target_pct"))
            cvf = _conv(a)
            conv_s = f"{cvf:+.2f}" if cvf == cvf else "&middot;"  # NaN -> dot
            bt = f"border-top:1px solid {LINE};"
            trs.append(
                "<tr>"
                f'<td style="padding:7px 8px 7px 0;{bt}"><span style="font-family:{FONT};'
                f'font-size:11px;font-weight:800;color:{color};">{act}</span></td>'
                f'<td style="padding:7px 8px;{bt}font-family:{MONO};font-size:12px;'
                f'font-weight:700;color:{INK};">{tkr}</td>'
                f'<td style="padding:7px 8px;{bt}font-family:{FONT};font-size:12px;'
                f'color:{INK2};">{nm}</td>'
                f'<td style="padding:7px 8px;{bt}font-family:{MONO};font-size:12px;'
                f'color:{INK2};white-space:nowrap;">{cur} &#8594; {tgt}</td>'
                f'<td style="padding:7px 0 7px 8px;{bt}font-family:{MONO};font-size:12px;'
                f'color:{INK2};text-align:right;">{conv_s}</td>'
                "</tr>"
            )
    table = (
        (
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
            'style="border-collapse:collapse;margin-top:18px;"><tr>'
            + _th("")
            + _th("Ticker")
            + _th("Name")
            + _th("Current &#8594; Target")
            + _th("Conv", "right", "0")
            + "</tr>"
            + "".join(trs)
            + "</table>"
        )
        if trs
        else ""
    )

    note = (
        f'<div style="font-family:{FONT};font-size:11.5px;color:{INK2};margin-top:18px;'
        f'padding-top:12px;border-top:1px solid {LINE};line-height:1.6;">The full '
        f"<b>Factor Snapshot</b> &mdash; conviction heatmap, per-stock factor cards and trade "
        f"levels &mdash; is <b>attached</b> as HTML; open it in a browser for the complete view. "
        f"{c.get('HOLD', 0)} held name{'s' if c.get('HOLD', 0) != 1 else ''} maintained.</div>"
    )

    body = head + table + note
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Trading Model v3 · Factor Snapshot</title></head>"
        f'<body style="margin:0;padding:0;background:{WARM};">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="border-collapse:collapse;background:{WARM};"><tr>'
        f'<td align="center" style="padding:22px 12px;">'
        f'<table role="presentation" width="{CONTENT_W}" cellpadding="0" cellspacing="0" border="0" '
        f'style="border-collapse:collapse;width:{CONTENT_W}px;max-width:100%;">'
        f'<tr><td bgcolor="{CANVAS}" style="background:{CANVAS};border:1px solid {LINE};'
        f'border-radius:14px;padding:30px 32px;">{body}</td></tr>'
        f'<tr><td style="padding:14px 4px;font-family:{FONT};font-size:10px;color:{MUTED};'
        f'text-align:center;">Trading Model v3 &middot; generated on the VPS &middot; {date}</td></tr>'
        "</table></td></tr></table></body></html>"
    )
