"""The scan summary counts a ticker that came back without a price as unpriced, not succeeded."""

import math

import pytest

from yahoofinance.utils.async_utils import batch
from yahoofinance.utils.async_utils.batch import display_processing_stats, process_batch_async

ROWS = {
    "PRICED": {"ticker": "PRICED", "price": 10.0},
    "NONE": {"ticker": "NONE", "price": None, "current_price": None},
    "NAN": {"ticker": "NAN", "price": math.nan},
}


async def _processor(ticker):
    if ticker == "BROKEN":
        raise RuntimeError("fetch failed")
    return ROWS[ticker]


@pytest.mark.asyncio
async def test_unpriced_rows_are_not_counted_as_succeeded(capsys):
    results = await process_batch_async(
        ["PRICED", "NONE", "NAN", "BROKEN"], _processor, batch_size=4, show_progress=True
    )

    # The rows themselves are unchanged: an unpriced row is still returned.
    assert results == {k: ROWS[k] for k in ("PRICED", "NONE", "NAN")}
    stats = batch.get_processing_stats()
    assert (stats["success_count"], stats["unpriced_count"], stats["error_count"]) == (1, 2, 1)

    display_processing_stats()
    out = capsys.readouterr().out
    assert "1/4 succeeded, 2 unpriced, 1 failed" in out


@pytest.mark.asyncio
async def test_unpriced_counted_on_the_timeout_path():
    await process_batch_async(
        ["PRICED", "NONE"], _processor, batch_size=2, show_progress=False, timeout_per_batch=5
    )
    stats = batch.get_processing_stats()
    assert (stats["success_count"], stats["unpriced_count"]) == (1, 1)


@pytest.mark.asyncio
async def test_results_without_a_price_field_are_not_judged():
    async def square(n):
        return n * n

    await process_batch_async([1, 2, 3], square, batch_size=3, show_progress=False)
    stats = batch.get_processing_stats()
    assert (stats["success_count"], stats["unpriced_count"]) == (3, 0)


def test_a_row_with_either_price_field_set_is_priced():
    assert not batch._is_unpriced({"price": None, "current_price": 5.0})
    assert batch._is_unpriced({"price": ""})
    assert not batch._is_unpriced({"price": 0.0})
