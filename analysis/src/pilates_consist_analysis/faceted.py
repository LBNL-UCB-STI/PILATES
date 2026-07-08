from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import pandas as pd


def _is_fake_table(table: Any) -> bool:
    return bool(getattr(table, "_pilates_fake_table", False))


def _fake_table_like(table: Any, frame: pd.DataFrame) -> Any:
    return type(table)(frame)


def _apply_at_filter(
    frame: pd.DataFrame,
    at: Optional[Mapping[str, Any]],
) -> pd.DataFrame:
    output = frame
    for column, value in (at or {}).items():
        if isinstance(value, (list, tuple, set, frozenset)):
            output = output.loc[output[column].isin(value)]
        else:
            output = output.loc[output[column] == value]
    return output.copy()


def _fake_delta(
    table: Any,
    *,
    value: str,
    over: str,
    by: Sequence[str],
    output: Optional[str],
) -> Any:
    frame = table.to_pandas().sort_values([*by, over]).reset_index(drop=True)
    delta_column = output or f"{value}_delta"
    previous = frame.groupby(list(by), dropna=False)[value].shift(1)
    frame[delta_column] = frame[value] - previous
    return _fake_table_like(table, frame)


def _fake_difference(
    table: Any,
    *,
    value: str,
    compare: str,
    baseline: Any,
    at: Optional[Mapping[str, Any]],
    by: Optional[Sequence[str]],
    output: Optional[str],
) -> Any:
    frame = _apply_at_filter(table.to_pandas(), at)
    diff_column = output or f"{value}_difference"
    baseline_column = f"{value}_baseline"
    keys = list(dict.fromkeys([*(by or ()), *((at or {}).keys())]))
    baseline_frame = frame.loc[frame[compare] == baseline, [*keys, value]].rename(
        columns={value: baseline_column}
    )
    merged = frame.merge(baseline_frame, on=keys, how="left")
    merged[diff_column] = merged[value] - merged[baseline_column]
    return _fake_table_like(table, merged)


def _fake_rank(
    table: Any,
    *,
    value: str,
    by: Sequence[str],
    descending: bool,
    output: Optional[str],
) -> Any:
    frame = table.to_pandas().copy()
    rank_column = output or f"{value}_rank"
    frame[rank_column] = (
        frame.groupby(list(by), dropna=False)[value]
        .rank(method="dense", ascending=not descending)
        .astype(int)
    )
    return _fake_table_like(table, frame)


def delta(
    table: Any,
    *,
    value: str,
    over: str,
    by: Sequence[str],
    output: Optional[str] = None,
) -> Any:
    """Return a table with first differences for ``value`` over an ordered facet."""
    if _is_fake_table(table):
        return _fake_delta(table, value=value, over=over, by=by, output=output)

    import ibis

    delta_column = output or f"{value}_delta"
    window = ibis.window(
        group_by=[table[column] for column in by],
        order_by=[table[over]],
    )
    previous = table[value].lag().over(window)
    return table.mutate(**{delta_column: table[value] - previous})


def difference(
    table: Any,
    *,
    value: str,
    compare: str,
    baseline: Any,
    at: Optional[Mapping[str, Any]] = None,
    by: Optional[Sequence[str]] = None,
    output: Optional[str] = None,
) -> Any:
    """Compare values across one facet against a baseline facet value."""
    if _is_fake_table(table):
        return _fake_difference(
            table,
            value=value,
            compare=compare,
            baseline=baseline,
            at=at,
            by=by,
            output=output,
        )

    filtered = table
    for column, selected in (at or {}).items():
        if isinstance(selected, (list, tuple, set, frozenset)):
            filtered = filtered.filter(filtered[column].isin(list(selected)))
        else:
            filtered = filtered.filter(filtered[column] == selected)

    keys = list(dict.fromkeys([*(by or ()), *((at or {}).keys())]))
    baseline_column = f"{value}_baseline"
    diff_column = output or f"{value}_difference"
    baseline_table = filtered.filter(filtered[compare] == baseline).select(
        *[filtered[column] for column in keys],
        filtered[value].name(baseline_column),
    )
    predicates = [filtered[column] == baseline_table[column] for column in keys]
    joined = filtered.left_join(baseline_table, predicates=predicates)
    return joined.mutate(**{diff_column: joined[value] - joined[baseline_column]})


def delta_change(
    table: Any,
    *,
    value: str,
    over: str,
    by: Sequence[str],
    delta_output: Optional[str] = None,
    output: Optional[str] = None,
) -> Any:
    """Return the change in first differences for ``value`` over an ordered facet."""
    delta_column = delta_output or f"{value}_delta"
    change_column = output or f"{delta_column}_change"
    first = delta(table, value=value, over=over, by=by, output=delta_column)
    if _is_fake_table(first):
        frame = first.to_pandas().sort_values([*by, over]).reset_index(drop=True)
        previous = frame.groupby(list(by), dropna=False)[delta_column].shift(1)
        frame[change_column] = frame[delta_column] - previous
        return _fake_table_like(first, frame)

    import ibis

    window = ibis.window(
        group_by=[first[column] for column in by],
        order_by=[first[over]],
    )
    previous = first[delta_column].lag().over(window)
    return first.mutate(**{change_column: first[delta_column] - previous})


def rank(
    table: Any,
    *,
    value: str,
    by: Sequence[str],
    descending: bool = True,
    output: Optional[str] = None,
) -> Any:
    """Rank rows within facet groups by a measured value."""
    if _is_fake_table(table):
        return _fake_rank(
            table,
            value=value,
            by=by,
            descending=descending,
            output=output,
        )

    import ibis

    rank_column = output or f"{value}_rank"
    rank_value = -table[value] if descending else table[value]
    window = ibis.window(
        group_by=[table[column] for column in by],
        order_by=[rank_value],
    )
    return table.mutate(**{rank_column: rank_value.dense_rank().over(window) + 1})


__all__ = ["delta", "difference", "delta_change", "rank"]
