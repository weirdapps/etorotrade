"""Consistency of the v3 pipeline map (a sign-off surface — it must match the engine).

A factor must never be shown as BOTH scored and non-scored: e.g. pe_forward was
listed DISCARDED while the live value recipes score it, so the report contradicted
itself. Gates may legitimately overlap the scored set (a factor can be scored AND an
eligibility gate, e.g. earn_trajectory), so only DISCARDED/MONITORED are checked.
"""

from scripts.v3_pipeline_map import DISCARDED, MONITORED, _scored_members


def test_scored_factors_not_also_discarded_or_monitored():
    scored = _scored_members()
    assert not (scored & set(DISCARDED)), (
        f"shown as both scored and discarded: {scored & set(DISCARDED)}"
    )
    assert not (scored & set(MONITORED)), (
        f"shown as both scored and monitored: {scored & set(MONITORED)}"
    )
