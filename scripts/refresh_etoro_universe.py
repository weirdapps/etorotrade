"""Refresh yahoofinance/input/etoro.csv from the eToro market-data API.

Fetches stocks + ETFs from https://public-api.etoro.com/api/v1/market-data/instruments
and atomically overwrites the input file with one row per (symbol, company, exchange).
Symbols come back from the API mostly in Yahoo-Finance format but need
light normalization: strip .US suffix, drop .RTH duplicates, trim HK
5-digit to 4-digit (00001.HK → 0001.HK).
"""

import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

INSTRUMENTS_URL = "https://public-api.etoro.com/api/v1/market-data/instruments"
STOCK_TYPE_ID = 5
ETF_TYPE_ID = 6
MIN_INSTRUMENTS_THRESHOLD = 1000

EXCHANGE_NAMES: dict[int, str] = {
    2: "NYSE",
    4: "Nasdaq",
    5: "NYSE",
    6: "FRA",
    7: "LSE",
    8: "NYSE",
    9: "Euronext Paris",
    10: "Bolsa De Madrid",
    11: "Borsa Italiana",
    12: "SIX",
    14: "Oslo Stock Exchange",
    15: "Stockholm Stock Exchange",
    16: "Copenhagen Stock Exchange",
    17: "Helsinki Stock Exchange",
    19: "OTC Markets",
    20: "CBOE",
    21: "HKEX",
    22: "Euronext Lisbon",
    23: "Euronext Brussels",
    24: "Tadawul",
    30: "Euronext Amsterdam",
    31: "ASX",
    32: "Vienna",
    33: "Xetra",
    34: "Dublin",
    35: "Prague SE",
    36: "Warsaw",
    37: "Budapest",
    38: "Xetra ETFs",
    39: "DFM",
    41: "Abu Dhabi",
    42: "LSE AIM",
    43: "LSE AIM",
    44: "LSE",
    56: "Tokyo Stock Exchange",
}

_HTTP_TIMEOUT_SEC = 30
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)
_INTER_PAGE_DELAY_SEC = 0.5

_DEFAULT_OUTPUT_CSV = str(Path(__file__).parent.parent / "yahoofinance" / "input" / "etoro.csv")
_DEFAULT_DELTA_LOG = str(
    Path(__file__).parent.parent / "yahoofinance" / "input" / ".universe-refresh-log.json"
)


def is_etorian_alias(item: dict) -> bool:
    """True if the symbol or displayName starts with 'ETORIAN' (deprecated placeholder)."""
    sym = (item.get("symbol") or "").upper()
    name = (item.get("displayName") or "").upper()
    return sym.startswith("ETORIAN") or name.startswith("ETORIAN")


_HK_PAD_WIDTH = 4

# eToro's spelling of a VENUE -> Yahoo's spelling of the same venue.
# .IM and .LN are Bloomberg-style codes that trade_modules/config_manager.py has always
# remapped at FETCH time while this generator did not, so the universe carried the eToro
# spelling and only the resolver knew better. Measured 2026-08-11 (1-month bars):
#   28IA.IM 0 / 28IA.MI 22 · ITBL.IM 0 / ITBL.MI 2 · SHLD.LN 0 / SHLD.L 22 · CSX5.LN 0 / CSX5.L 22
_SUFFIX_REMAP = {
    ".ASX": ".AX",
    ".ZU": ".SW",
    ".NV": ".AS",
    ".LSB": ".LS",
    ".IM": ".MI",
    ".LN": ".L",
}

# Markers for a WRAPPER over a US listing rather than a venue: the bare ticker IS the listing.
# .CH is eToro's Chinese-ADR marker on NYSE/Nasdaq (TCOM Trip.com, ATHM Autohome, QIHU, JMEI,
# CYOU, QUNR, GSOL) — NOT Switzerland, which eToro spells .ZU and Yahoo spells .SW, both handled
# above. Without a rule here these seven reached etoro.csv still suffixed, and ConfigManager's
# table then turned them into Swiss symbols that do not exist.
_STRIP_SUFFIXES = (".US", ".CH")

# Currency / quote-unit lines eToro appends. Not a venue and not a separate security. Mirrors
# _CURRENCY_SUFFIXES in trade_modules/config_manager.py, which strips the same four at fetch time.
#
# WHETHER TO STRIP OR DROP DEPENDS ON THE SHAPE, and the live file shows both, cleanly split:
#   KSP.L.GBX  -> the stem KSP.L is still venue-qualified and NO bare KSP.L row exists, so
#                 stripping RECOVERS a London listing. All 3 .GBX rows are this shape.
#   AAPL.EUR   -> the stem AAPL carries no venue, the row's exchange is FRA, and a Nasdaq AAPL
#                 row already exists. All 10 .EUR rows are this shape: stripping would collide
#                 with the real row and dedupe could keep the one labelled FRA.
# Hence: strip when the stem is still venue-qualified, drop when it is not. That rule is derived
# from the shape rather than from today's suffix census, so a future .USD/.GBP row is handled
# correctly without another audit.
_CURRENCY_SUFFIXES = (".EUR", ".USD", ".GBP", ".GBX")

_DROP_SUFFIXES = frozenset(
    {
        ".RTH",
        ".DELISTED",
        ".TEST",
        ".OLD",
        ".EXT",
        ".24-7",
        ".CALL1",
        ".CALL2",
        ".PUT1",
        ".PUT2",
        ".TENDER",
        ".CASHRESERVED",
        ".MOEX",
        ".RIGHT",
        ".WS",
        ".PFD",
        # eToro duplicate-listing artifacts for Xetra/Euronext funds: CEBP.DE11, IQQ6.D11,
        # ICGB.DE22. No Yahoo counterpart exists under either the digit form or the plain
        # venue form, and all 15 rows have never returned a single price bar.
        ".DE11",
        ".D11",
        ".DE22",
        # corporate-action lifecycle markers
        ".ACQUIRED",
        ".DELETE",
        # pre-IPO product (SpaceX.IPO) — an eToro instrument, not a listed security
        ".IPO",
    }
)

#: Venue strings that mean "a US listing", so a bare symbol on one of them is correct under the
#: owner's rule. Matched case-insensitively as substrings because two producers spell them
#: differently: this script's EXCHANGE_NAMES yields "CBOE" while eToro's own catalogue says
#: "Chicago Board Options Exchange".
_US_VENUE_TOKENS = ("NYSE", "NASDAQ", "OTC", "CBOE", "CHICAGO BOARD OPTIONS")

# Suffixes that are ALREADY valid and must pass through untouched. Membership here means
# "a human has ruled on this suffix" — that is the whole point of the list. Anything absent
# from every ruling set raises a warning instead of sailing through, which is precisely how
# `.CH` sat mis-mapped for months: nothing in the pipeline could tell "known-good" from
# "never looked at".
#
# Two entries are deliberate exceptions to "valid Yahoo symbol", recorded so nobody reopens them:
#   .DH  Abu Dhabi. Yahoo indexes the Gulf under numeric codes, so all 33 rows are unfetchable
#        under this spelling. Kept ON PURPOSE — the fix is per-name substitutions, not a suffix
#        rule, and dropping the venue would lose the identities as well as the prices.
#   .A/.B/.C  US class shares. eToro and SHARADAR both key on the DOT form (BF.A, BRK.B) while
#        Yahoo wants the dash form; the fetch resolvers already convert, so the universe keeps
#        the upstream identity and must not be "corrected" here.
_ALLOWED_PASSTHROUGH = frozenset(
    {
        ".L",
        ".DE",
        ".PA",
        ".AX",
        ".ST",
        ".OL",
        ".HK",
        ".T",
        ".MI",
        ".HE",
        ".AS",
        ".CO",
        ".BR",
        ".SW",
        ".MC",
        ".VI",
        ".LS",
        ".WA",
        ".BD",
        ".IR",
        ".PR",
        ".AE",
        ".DH",
        ".A",
        ".B",
        ".C",
    }
)

# Stems that mark a placeholder no matter what follows them.
_JUNK_STEMS = ("CVR", "WRTS", "DORMANT", "DRM")


def _is_junk_suffix(suffix: str) -> bool:
    """True if the SUFFIX alone marks a junk/placeholder instrument."""
    s = suffix.lstrip(".").upper()
    if f".{s}" in _DROP_SUFFIXES:
        return True
    if s.startswith(("CVR", "DUP", "ETF", "STOCK")):
        return True
    # ANY purely numeric suffix. No exchange on earth spells itself with digits, so the old
    # `len(s) > 3` guard bought nothing and let WRTS.APRN.15 / .18 / .20 through.
    if s.isdigit():
        return True
    return False


def is_junk_symbol(symbol: str, company: str = "") -> bool:
    """True if the WHOLE symbol is a placeholder, independent of its suffix.

    ``_is_junk_suffix`` reads only the text after the LAST dot, which is why it caught
    ``US.CVR1`` and missed ``CVR.THS`` — same marker, other side of the dot. It also misses
    every space-delimited placeholder (``LSE CVR``, ``MTN DUMMY CVR``, ``CEPU ESCROW ASSET``)
    because those contain no dot at all.
    """
    s = (symbol or "").upper().strip()
    if not s:
        return True
    stem = s.split(".")[0]
    if stem in _JUNK_STEMS or stem.startswith(("DORMANT", "DRM")):
        return True
    # CVR / escrow / merger / option placeholders. eToro writes these with spaces, and a space
    # can never appear in a real ticker, but see bloomberg_shape() before widening this to
    # "any space" — four rows with a space are real securities in Bloomberg notation.
    if " " in s and re.search(r"\b(CVR\d*|ESCROW|DUMMY|LOCKED)\b", s):
        return True
    if " " in s and re.search(r"\b[CP]\d+$", s):  # option line, e.g. "ETOR 4 C40"
        return True
    # Lifecycle markers, in the four spellings eToro actually uses: SNDK_OLD, BMPS-OLD,
    # MRK.DE_OLD and "STJ.L OLD" (a space). \bOLD$ cannot match GOLD — no word boundary there.
    if re.search(r"(_OLD|\bOLD)$", s):
        return True
    # eToro corporate-action / pre-listing placeholders, where the row's only NAME is its own
    # internal code: CA141.L named "CA141", IPO56.L named "IPO56", CA.OPS31.L named "CA.Ops31".
    #
    # THE company==code CONJUNCTION IS LOAD-BEARING, twice over. A bare ^CA\d+ symbol rule also
    # deletes CA21 (Royal Dutch Shell), CA8 (Meredith Holdings), CA12908 (Everfuel A/S),
    # CA13026 (Believe SA) and four more real companies — measured, 63 junk rows with the
    # conjunction against 71 without. And a bare ^IPO\d* rule would delete IPO.L, which is
    # IP Group PLC. Only the row whose name IS its code is a placeholder.
    code = company.strip().upper().replace(" ", "")
    if code and re.match(r"^(CA\d+|CA\.OPS\d+|IPO\d+)$", code):
        # the symbol must be that same code, optionally venue-qualified (IPO56 / IPO56.L)
        without_venue = s.rsplit(".", 1)[0] if "." in s else s
        if code in (s, without_venue):
            return True
    return False


#: An unknown suffix on this many symbols FAILS the refresh instead of warning. A genuine new
#: venue arrives in bulk — .ASX was 244 rows, .DE11 13, .LSB 8, .CH 7 — while stray one-off junk
#: is 1-2 rows. Three is the smallest number that separates them, and failing is deliberate:
#: main() returns 1 before writing, the workflow's commit step is skipped, and the universe
#: holds at its last good state until a human rules on the suffix.
_UNKNOWN_SUFFIX_FAIL_THRESHOLD = 3


def audit_symbols(rows: list[dict]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Find eToro symbols that NO rule has ruled on. The guard this module was missing.

    Every suffix is meant to be in exactly one of four sets — ``_SUFFIX_REMAP`` (venue spelling),
    ``_STRIP_SUFFIXES`` / ``_CURRENCY_SUFFIXES`` (a wrapper over a listing), ``_DROP_SUFFIXES``
    plus the junk predicates (not a security), or ``_ALLOWED_PASSTHROUGH`` (already valid).
    A suffix in none of them used to be returned unchanged and written to the universe in
    silence. That is how ``.CH`` — eToro's China-ADR marker — spent months being rewritten into
    Swiss symbols that do not exist: nothing in the pipeline could distinguish "we decided this
    is fine" from "we have never looked at this".

    Returns ``(unknown, review)``:
      * ``unknown`` maps an unruled suffix to example symbols. Escalates to a hard failure at
        ``_UNKNOWN_SUFFIX_FAIL_THRESHOLD``.
      * ``review`` maps a KNOWN-but-unresolved shape to example symbols. Always warns, never
        fails — these are pre-existing backlog, and a guard that is red on day one is a guard
        nobody reads.
    """
    unknown: dict[str, list[str]] = {}
    review: dict[str, list[str]] = {}

    def _add(bucket: dict[str, list[str]], key: str, sym: str) -> None:
        bucket.setdefault(key, []).append(sym)

    for row in rows:
        sym = (row.get("symbol") or "").upper().strip()
        company = row.get("company") or ""
        exchange = (row.get("exchange") or "").strip()
        if not sym or is_junk_symbol(sym, company):
            continue
        if bloomberg_shape(sym):
            _add(review, "bloomberg TICKER-CC form (needs a per-venue remap)", sym)
            continue
        if " " in sym:
            _add(review, "space in symbol (never fetchable as spelled)", sym)
            continue
        if "." not in sym:
            # A bare symbol is a US listing under the owner's rule. When eToro says otherwise,
            # the row carries a non-US venue with no suffix to key on, and passthrough quietly
            # aims a US ticker at a foreign company. Reported, not guessed at: the fix is a
            # per-row remap keyed on the exchange COLUMN and it must be hand-reviewed.
            if exchange and not any(v in exchange.upper() for v in _US_VENUE_TOKENS):
                _add(review, f"bare symbol on a non-US venue ({exchange})", sym)
            continue
        suffix = f".{sym.rsplit('.', 1)[1]}"
        if (
            suffix in _SUFFIX_REMAP
            or suffix in _STRIP_SUFFIXES
            or suffix in _CURRENCY_SUFFIXES
            or suffix in _ALLOWED_PASSTHROUGH
            or _is_junk_suffix(suffix)
        ):
            continue
        _add(unknown, suffix, sym)

    return unknown, review


def report_audit(unknown: dict[str, list[str]], review: dict[str, list[str]]) -> bool:
    """Log the audit. Returns True when the refresh must ABORT rather than write."""
    for shape, syms in sorted(review.items(), key=lambda kv: -len(kv[1])):
        logger.warning(
            "REVIEW — %d symbol(s) %s: %s%s",
            len(syms),
            shape,
            ", ".join(sorted(syms)[:5]),
            " ..." if len(syms) > 5 else "",
        )
    if not unknown:
        logger.info("Suffix audit clean: every suffix is ruled on.")
        return False
    worst = 0
    for suffix, syms in sorted(unknown.items(), key=lambda kv: -len(kv[1])):
        worst = max(worst, len(syms))
        logger.error(
            "UNRULED SUFFIX %s on %d symbol(s): %s%s",
            suffix,
            len(syms),
            ", ".join(sorted(syms)[:5]),
            " ..." if len(syms) > 5 else "",
        )
    if worst < _UNKNOWN_SUFFIX_FAIL_THRESHOLD:
        logger.warning(
            "Unruled suffixes are below the failure threshold (%d < %d); continuing.",
            worst,
            _UNKNOWN_SUFFIX_FAIL_THRESHOLD,
        )
        return False
    logger.error(
        "ABORTING: an unruled suffix appears on >= %d symbols, which is what a NEW EXCHANGE "
        "looks like. Rule on it in scripts/refresh_etoro_universe.py — add it to _SUFFIX_REMAP "
        "(eToro's spelling of a venue Yahoo spells differently), _STRIP_SUFFIXES (a wrapper "
        "over a listing), _DROP_SUFFIXES (not a security), or _ALLOWED_PASSTHROUGH (already a "
        "valid symbol) — then re-run. The universe is left at its last good state.",
        _UNKNOWN_SUFFIX_FAIL_THRESHOLD,
    )
    return True


def bloomberg_shape(symbol: str) -> bool:
    """True for ``SRT3 GY`` / ``HAWK US`` — a real security in Bloomberg's ``TICKER CC`` form.

    Deliberately NOT junk and deliberately NOT resolved: the correct answer is a per-venue
    remap (`` GY`` -> ``.DE``), which is a reviewed per-row decision, not a suffix rule. They
    are reported by :func:`audit_symbols` so they stay visible instead of being silently
    dropped alongside the CVR placeholders they superficially resemble.
    """
    parts = (symbol or "").upper().strip().split()
    return len(parts) == 2 and len(parts[1]) == 2 and parts[1].isalpha()


def normalize_symbol(symbol: str, company: str = "") -> str | None:
    """Normalize eToro symbol to Yahoo Finance format.

    Returns None for symbols that should be skipped (RTH variants, junk instruments).
    """
    upper = symbol.upper().strip()

    if is_junk_symbol(upper, company):
        return None

    # A doubled dot is eToro writing an LSE TIDM that ends in a period: YU..L is Yu Group,
    # RE..L is R.E.A. Holdings — both live, both 22 bars under the collapsed spelling, and
    # neither collides with an existing row. Collapse it; NEVER strip it to the bare stem,
    # because FA..L -> FA and IX..L -> IX would merge into unrelated US tickers.
    while ".." in upper:
        upper = upper.replace("..", ".")

    if "." in upper:
        base, dot_suffix = upper.rsplit(".", 1)
        full_suffix = f".{dot_suffix}"
        if _is_junk_suffix(full_suffix):
            return None
        if full_suffix in _SUFFIX_REMAP:
            upper = base + _SUFFIX_REMAP[full_suffix]

    # A currency line on a BARE stem is a duplicate of a row that already exists (AAPL.EUR on
    # FRA against the Nasdaq AAPL). Drop it rather than strip it into a collision.
    for cur_suffix in _CURRENCY_SUFFIXES:
        if upper.endswith(cur_suffix) and "." not in upper[: -len(cur_suffix)]:
            return None

    # Strip wrapper and currency markers, repeatedly: KSP.L.GBX -> KSP.L, BRK.B.US -> BRK.B.
    # Bounded so a pathological symbol cannot spin.
    for _ in range(4):
        for strip_suffix in _STRIP_SUFFIXES + _CURRENCY_SUFFIXES:
            if upper.endswith(strip_suffix) and len(upper) > len(strip_suffix):
                upper = upper[: -len(strip_suffix)]
                break
        else:
            break

    if upper.endswith(".HK"):
        base = upper[:-3]
        # Pad BOTH ways. The old guard was `len(base) > _HK_PAD_WIDTH`, so it only ever
        # shortened 5-digit codes and left 1-3 digit ones alone — 5.HK and 66.HK stayed
        # unpadded and 404. zfill never truncates, so genuine long codes are untouched.
        if base.isdigit():
            upper = base.lstrip("0").zfill(_HK_PAD_WIDTH) + ".HK"
    return upper


# Venues where Yahoo HYPHENATES a share-class letter. Stockholm and Copenhagen do; Helsinki and
# Oslo DO NOT, and having them here was a live bug — measured against Yahoo on 2026-08-11,
# 1-month bar counts:
#     KESKOA.HE 21 / KESKO-A.HE 0        METSA.HE 21 / METS-A.HE 0
#     ODFB.OL   21 / ODF-B.OL   0        ASSA-B.ST 21 / ASSAB.ST  0   <- control, hyphen correct
# The generator was writing KESKO-A.HE, KESKO-B.HE, METS-A.HE and METS-B.HE into etoro.csv, so
# Kesko A/B and Metsa Board A/B were in the universe as ACTIVE and could never return a price.
# Oslo never fired in practice (no symmetric XA/XB pair present) but was armed: VENDB.OL is
# already in the file, and the day eToro adds VENDA.OL both would be hyphenated into nothing.
_SCANDI_SUFFIXES = frozenset({".ST", ".CO"})
_SHARE_CLASS_KEYWORDS = ("ser.", "series", "class", "klass", "aktie")


def fix_share_classes(rows: list[dict]) -> list[dict]:
    """Insert hyphen before Scandinavian share-class letters (A/B).

    Detects share classes via two strategies:
    1. A/B pair: both BASEA.ST and BASEB.ST exist → insert hyphen in both
    2. Company name: contains 'ser.', 'Series', 'Class', etc. → insert hyphen

    Already-hyphenated symbols (ASSA-B.ST) are left alone.
    """
    all_symbols = {r["symbol"] for r in rows}

    def _needs_hyphen(sym: str, company: str) -> bool:
        for suf in _SCANDI_SUFFIXES:
            if not sym.endswith(suf):
                continue
            base = sym[: -len(suf)]
            if len(base) < 2 or base[-1] not in ("A", "B") or "-" in base:
                return False
            other_class = "A" if base[-1] == "B" else "B"
            other_sym = base[:-1] + other_class + suf
            other_hyph = base[:-1] + "-" + other_class + suf
            if other_sym in all_symbols or other_hyph in all_symbols:
                return True
            if any(kw in company.lower() for kw in _SHARE_CLASS_KEYWORDS):
                return True
            return False
        return False

    def _insert_hyphen(sym: str) -> str:
        for suf in _SCANDI_SUFFIXES:
            if sym.endswith(suf):
                base = sym[: -len(suf)]
                return base[:-1] + "-" + base[-1] + suf
        return sym

    out: list[dict] = []
    fixed = 0
    for row in rows:
        sym = row["symbol"]
        company = row.get("company", "")
        if _needs_hyphen(sym, company):
            row = {**row, "symbol": _insert_hyphen(sym)}
            fixed += 1
        out.append(row)
    if fixed:
        logger.info("Fixed %d Scandinavian share-class symbols (inserted hyphen)", fixed)
    return out


def dedupe_by_symbol(rows: list[dict]) -> list[dict]:
    """Remove duplicate rows by 'symbol' key, preserving first occurrence and overall order."""
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        sym = row["symbol"]
        if sym in seen:
            continue
        seen.add(sym)
        out.append(row)
    return out


def _get_credential(env_var: str, keychain_service: str) -> str | None:
    """Return credential from environment variable; fall back to macOS keychain on local runs."""
    value = os.environ.get(env_var)
    if value:
        return value
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", "etoro-api", "-s", keychain_service, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_credentials() -> tuple[str, str]:
    """Resolve (api_key, user_key) from env vars or macOS keychain. Raises if either missing."""
    api_key = _get_credential("ETORO_API_KEY", "etoro-public-key")
    user_key = _get_credential("ETORO_USER_KEY", "etoro-user-key")
    if not api_key or not user_key:
        missing = []
        if not api_key:
            missing.append("ETORO_API_KEY")
        if not user_key:
            missing.append("ETORO_USER_KEY")
        raise RuntimeError(
            f"Missing credentials: {', '.join(missing)}. Set env vars or store in macOS keychain (service: etoro-public-key / etoro-user-key)."
        )
    return api_key, user_key


def fetch_all_instruments(
    api_key: str,
    user_key: str,
    max_retries: int = 3,
) -> list[dict]:
    """Fetch all instruments from the market-data API (single call, no pagination).

    Filters to Stocks (type 5) and ETFs (type 6), then maps fields to the
    format expected by the rest of the pipeline.
    """
    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        headers = {
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
            "x-api-key": api_key,
            "x-user-key": user_key,
            "x-request-id": str(uuid.uuid4()),
        }
        try:
            response = requests.get(INSTRUMENTS_URL, headers=headers, timeout=_HTTP_TIMEOUT_SEC)
            if response.status_code == 200:
                raw = response.json().get("instrumentDisplayDatas", [])
                filtered = [
                    i for i in raw if i.get("instrumentTypeID") in (STOCK_TYPE_ID, ETF_TYPE_ID)
                ]
                items = []
                for i in filtered:
                    items.append(
                        {
                            "instrumentId": i.get("instrumentID"),
                            "symbol": i.get("symbolFull", ""),
                            "displayName": i.get("instrumentDisplayName", ""),
                            "exchangeName": EXCHANGE_NAMES.get(i.get("exchangeID", 0), ""),
                        }
                    )
                logger.info(
                    "  Fetched %d total, %d Stocks+ETFs after type filter",
                    len(raw),
                    len(items),
                )
                return items
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < max_retries:
            time.sleep(2**attempt)

    raise RuntimeError(
        f"fetch_all_instruments: all {max_retries} attempts failed. Last error: {last_error}"
    )


_CSV_COLUMNS = ["symbol", "company", "exchange"]


def write_universe_csv(rows: list[dict], path: str) -> None:
    """Atomically write rows to a CSV at `path`.

    Writes to `path + ".tmp"` first, then os.replace() to final location.
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


_SAMPLE_LIMIT = 50


def write_delta_log(
    path: str,
    new_symbols: list[str],
    removed_symbols: list[str],
    total_count: int,
) -> None:
    """Write a JSON snapshot of this run's delta."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_count": total_count,
        "new_count": len(new_symbols),
        "removed_count": len(removed_symbols),
        "sample_new": sorted(new_symbols)[:_SAMPLE_LIMIT],
        "sample_removed": sorted(removed_symbols)[:_SAMPLE_LIMIT],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _read_existing_symbols(path: str) -> set[str]:
    """Return the set of symbols currently in input/etoro.csv (uppercase). Empty if missing."""
    if not os.path.exists(path):
        return set()
    out: set[str] = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            sym = (row.get("symbol") or "").strip().upper()
            if sym:
                out.add(sym)
    return out


def main(
    output_csv_path: str = _DEFAULT_OUTPUT_CSV,
    delta_log_path: str = _DEFAULT_DELTA_LOG,
) -> int:
    """Run the refresh pipeline. Returns exit code (0 success, 1 error)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        api_key, user_key = get_credentials()
    except RuntimeError as e:
        logger.error(str(e))
        return 1

    logger.info("Fetching eToro market-data instruments (Stocks + ETFs)...")
    try:
        items = fetch_all_instruments(api_key, user_key)
    except RuntimeError as e:
        logger.error(str(e))
        return 1

    logger.info("Fetched %d Stocks+ETFs from market-data API", len(items))

    if len(items) < MIN_INSTRUMENTS_THRESHOLD:
        logger.error(
            "Refusing to proceed: only %d items returned (threshold %d). API may be broken.",
            len(items),
            MIN_INSTRUMENTS_THRESHOLD,
        )
        return 1

    # AUDIT BEFORE NORMALISING. Runs on the RAW eToro symbols, because the question is "did
    # eToro send us something no rule has ruled on" — after normalisation an unruled suffix is
    # indistinguishable from a deliberate passthrough, which is exactly the blindness that let
    # .CH survive. Aborts before anything is written, so a new venue leaves the universe at its
    # last good state rather than half-ruled.
    unknown, review = audit_symbols(
        [
            {
                "symbol": (i.get("symbol") or "").strip(),
                "company": i.get("displayName", ""),
                "exchange": i.get("exchangeName", ""),
            }
            for i in items
            if not is_etorian_alias(i)
        ]
    )
    if report_audit(unknown, review):
        return 1

    # Filter ETORIAN aliases + empty symbols + normalize (.US, .RTH, HK 5→4 digit)
    rows: list[dict] = []
    skipped_no_symbol = 0
    skipped_alias = 0
    skipped_rth = 0
    for item in items:
        if is_etorian_alias(item):
            skipped_alias += 1
            continue
        raw_symbol = (item.get("symbol") or "").strip()
        if not raw_symbol:
            skipped_no_symbol += 1
            continue
        normalized = normalize_symbol(raw_symbol, item.get("displayName", ""))
        if normalized is None:
            skipped_rth += 1
            continue
        rows.append(
            {
                "symbol": normalized,
                "company": item.get("displayName", ""),
                "exchange": item.get("exchangeName", ""),
            }
        )

    logger.info(
        "After filters: %d candidates (skipped %d aliases, %d empty, %d RTH)",
        len(rows),
        skipped_alias,
        skipped_no_symbol,
        skipped_rth,
    )

    rows = fix_share_classes(rows)
    deduped = dedupe_by_symbol(rows)
    logger.info("After dedupe: %d unique symbols", len(deduped))

    existing_symbols = _read_existing_symbols(output_csv_path)
    new_symbols_set = {r["symbol"] for r in deduped}
    new_symbols = sorted(new_symbols_set - existing_symbols)
    removed_symbols = sorted(existing_symbols - new_symbols_set)
    logger.info(
        "Delta: +%d new, -%d removed (vs %d existing)",
        len(new_symbols),
        len(removed_symbols),
        len(existing_symbols),
    )

    write_universe_csv(deduped, output_csv_path)
    write_delta_log(
        delta_log_path,
        new_symbols=new_symbols,
        removed_symbols=removed_symbols,
        total_count=len(deduped),
    )

    logger.info("Wrote %s", output_csv_path)
    logger.info("Delta log: %s", delta_log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
