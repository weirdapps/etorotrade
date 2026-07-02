"""
Performance Matrix Builder

Pivots a flat list of (period, config, value) records into a T×N numpy
matrix suitable for PBO/CSCV analysis.  Duplicate (period, config) pairs
are averaged.  Missing cells are NaN.

Pure / no I/O.
"""

import numpy as np


def build_perf_matrix(
    rows: list[dict],
    period_key: str = "signal_date",
    config_key: str = "ticker",
    value_key: str = "alpha",
) -> tuple[list, list, np.ndarray]:
    """
    Pivot rows into a T×N performance matrix.

    Args:
        rows:       List of dicts, each containing period_key, config_key,
                    and value_key fields.
        period_key: Column name for the period dimension (rows in matrix).
        config_key: Column name for the config dimension (columns in matrix).
        value_key:  Column name for the numeric value to aggregate.

    Returns:
        (row_labels, col_labels, matrix) where:
          - row_labels: sorted unique period values (list)
          - col_labels: sorted unique config values (list)
          - matrix:     np.ndarray of shape (T, N), dtype=np.float64;
                        cell = mean of value_key for matching rows, NaN if absent.
    """
    if not rows:
        return [], [], np.empty((0, 0), dtype=np.float64)

    row_labels = sorted({r[period_key] for r in rows})
    col_labels = sorted({r[config_key] for r in rows})
    row_idx = {label: i for i, label in enumerate(row_labels)}
    col_idx = {label: j for j, label in enumerate(col_labels)}

    T = len(row_labels)
    N = len(col_labels)

    # Accumulate sums and counts for averaging
    totals = np.zeros((T, N), dtype=np.float64)
    counts = np.zeros((T, N), dtype=np.int64)

    for row in rows:
        i = row_idx[row[period_key]]
        j = col_idx[row[config_key]]
        totals[i, j] += float(row[value_key])
        counts[i, j] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.where(counts > 0, totals / counts, np.nan).astype(np.float64)

    return row_labels, col_labels, matrix
