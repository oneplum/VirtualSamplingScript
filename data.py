#!/usr/bin/env python3

"""
Dash application for visualizing BLTI results.

Input columns:
    dataset_name,
    level,
    method,
    lighting_enabled,
    true_samples,
    virtual_sampling_method,
    virtual_samples,
    trans_type,
    trans_axis,
    frame_idx,
    blti_0,
    blti_1,
    blti_2,
    blti_3,
    blti_4,
    blti_5
"""

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import polars as pl

from blti import BLTI_K
from common import DATASET_ORDER, METHOD_ORDER, VIRTUAL_METHOD_ORDER, FrameBasic

SEQ_COLS = [field.name for field in fields(FrameBasic)]
METRIC_COLS = [f"blti_{i}" for i in range(BLTI_K)]

X_COL = "band_idx"
Y_COL = "blti"
DELTA_COL = "diff"

AGG_EXPRS = {
    "mean": lambda expr: expr.mean(),
    "median": lambda expr: expr.median(),
    "p10": lambda expr: expr.quantile(0.10),
    "p90": lambda expr: expr.quantile(0.90),
}

COL_ORDERS = {
    "dataset_name": DATASET_ORDER,
    "method": METHOD_ORDER,
    "virtual_sampling_method": VIRTUAL_METHOD_ORDER,
}


def load_data(path: Path) -> pl.LazyFrame | None:
    parquet_files = sorted(path.glob("*.parquet"))

    if parquet_files:
        print(f"Found {len(parquet_files)} parquet files.")
        return pl.scan_parquet(parquet_files)

    csv_files = sorted(path.glob("*.csv"))

    if csv_files:
        print(f"Found {len(csv_files)} csv files.")
        return pl.scan_csv(csv_files)

    return None


def filter_data(lf: pl.LazyFrame, filters: dict[str, Any]) -> pl.LazyFrame:
    filter_exprs: list[pl.Expr] = []

    for col_name, col_val in filters.items():
        if col_val is None:
            continue

        if isinstance(col_val, (list, set, tuple)):
            if not col_val:
                continue

            filter_exprs.append(pl.col(col_name).is_in(list(col_val)))
        else:
            filter_exprs.append(pl.col(col_name) == col_val)

    if not filter_exprs:
        return lf

    return lf.filter(pl.all_horizontal(filter_exprs))


def sort_data(lf: pl.LazyFrame, sort_cols: list[str]) -> pl.LazyFrame:
    sort_exprs = []

    for sort_col in sort_cols:
        if sort_col in COL_ORDERS:
            order = {val: i for i, val in enumerate(COL_ORDERS[sort_col])}
            expr = pl.col(sort_col).replace_strict(
                order,
                default=len(order),
            )
        else:
            expr = pl.col(sort_col)

        sort_exprs.append(expr)

    return lf.sort(sort_exprs)


def _agg_expr(agg_type: str, cols: dict[str, str | None]) -> list[pl.Expr]:
    if agg_type not in AGG_EXPRS:
        raise ValueError(
            f"Unsupported aggregation: {agg_type}Expected one of {list(AGG_EXPRS)}"
        )

    agg_expr = AGG_EXPRS[agg_type]

    return [
        agg_expr(pl.col(col)).alias(alias) if alias else agg_expr(pl.col(col))
        for col, alias in cols.items()
    ]


def agg_data(
    lf: pl.LazyFrame,
    group_by: list[str],
    aggs: dict[str, dict[str, str | None]] | None = None,
) -> pl.LazyFrame:
    agg_exprs = []

    if aggs:
        for agg_type, cols in aggs.items():
            agg_exprs.extend(_agg_expr(agg_type, cols))

    if not agg_exprs:
        agg_exprs.append(pl.all())

    glf = lf.group_by(group_by).agg(agg_exprs)

    glf = sort_data(glf, group_by)

    return glf


def to_long(lf: pl.LazyFrame, idx_cols: list = SEQ_COLS, var_col: str = X_COL, val_col: str = Y_COL) -> pl.LazyFrame:
    llf = lf.unpivot(
        on=METRIC_COLS, index=idx_cols, variable_name=var_col, value_name=val_col
    )

    if var_col == X_COL:
        llf = llf.with_columns(pl.col(X_COL).str.strip_prefix("blti_").cast(pl.UInt8))

    return llf


def process_data(
    lf: pl.LazyFrame,
    filters: dict[str, Any],
    group_by: list[str],
    baselines: dict[str, Any] | None = None,
    one_frame: bool = False,
    use_frame: bool = False,
    debug: bool = False,
    is_long: bool = True
) -> pl.LazyFrame:
    flf = filter_data(lf, filters)

    if debug:
        data_rows, data_seq = (
            flf.select([pl.len(), pl.struct(SEQ_COLS).n_unique()]).collect().row(0)
        )
        print(f"Filtered data: {data_rows} rows, {data_seq} sequences.")

    if one_frame:
        return flf.sort(pl.col("frame_idx"))

    if use_frame:
        flf = to_long(flf)
        aggs = {
            "mean": {Y_COL: "mean"},
            "median": {Y_COL: "median"},
            "p10": {Y_COL: "p10"},
            "p90": {Y_COL: "p90"},
        }
        flf = agg_data(flf, group_by + [X_COL], aggs)
        return flf
    
    aggs = {"mean": {metric_col: None for metric_col in METRIC_COLS}}
    flf = agg_data(flf, SEQ_COLS, aggs)
    if debug:
        data_rows, data_seq = (
            flf.select([pl.len(), pl.struct(SEQ_COLS).n_unique()]).collect().row(0)
        )
        print(f"Sequence data: {data_rows} rows, {data_seq} sequences.")

    flf = agg_data(flf, group_by, aggs)

    var_col = X_COL
    val_col = Y_COL
    if baselines:
        on_keys = [
            c for c in flf.collect_schema().names()
            if c not in baselines and c not in METRIC_COLS
        ]

        if on_keys:
            select_cols = on_keys + [pl.col(metric_col).alias(f"{metric_col}_baseline") for metric_col in METRIC_COLS]
        else:
            select_cols = [pl.col(metric_col).alias(f"{metric_col}_baseline") for metric_col in METRIC_COLS]
        
        blf = filter_data(flf, baselines).select(select_cols)

        if on_keys:
            flf = flf.join(blf, on=on_keys, how="inner", validate="m:1")
        else:
            flf = flf.join(blf, how="cross")

        flf = (
            flf
            .with_columns([(pl.col(metric_col) - pl.col(f"{metric_col}_baseline")).alias(metric_col) for metric_col in METRIC_COLS ])
            .drop([f"{metric_col}_baseline" for metric_col in METRIC_COLS])
        )

        val_col = DELTA_COL

    sort_cols = group_by
    if is_long:
        flf = to_long(flf, group_by, var_col, val_col)
        sort_cols = group_by + [var_col]
    else:
        sort_cols = group_by

    flf = sort_data(flf, sort_cols)

    return flf


def load_col_data(lf: pl.LazyFrame, cols: list) -> dict[str, list]:
    slf = sort_data(lf, SEQ_COLS)

    return {
        col: (
            slf.select(pl.col(col).unique(maintain_order=True))
            .collect()
            .get_column(col)
            .to_list()
        )
        for col in cols
    }


def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Compute BLTI for image sequences.")

    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        required=True,
        help="Path to parquet data",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash debug mode.",
    )

    return parser


def show_plot(lf: pl.LazyFrame, x_col, y_col, group_col: str | None = None):
    select_cols = [x_col, y_col]
    if group_col:
        select_cols.append(group_col)
    df = lf.select(select_cols).collect()

    fig, ax = plt.subplots()

    groups = (
        df.partition_by(group_col, maintain_order=True, as_dict=True)
        if group_col
        else {(): df}
    )

    for group_val, group in groups.items():
        if not group_col:
            group_val = ""
        ax.plot(
            group[x_col].to_numpy(),
            group[y_col].to_numpy(),
            label=group_val
        )
    ax.grid(True, axis="y", alpha=0.25)
    if group_col:
        ax.legend()
    plt.show()


def show_stem(lf: pl.LazyFrame, x_col, y_col, group_col: str | None = None):
    select_cols = [x_col, y_col]
    if group_col:
        select_cols.append(group_col)
    df = lf.select(select_cols).collect()

    fig, ax = plt.subplots()

    groups = (
        df.partition_by(group_col, maintain_order=True, as_dict=True)
        if group_col
        else {(): df}
    )

    n_groups = len(groups)
    offset = 0.6 / max(1, n_groups)

    x_data = list(dict.fromkeys(df[x_col]))
    x_map = {v: k for k, v in enumerate(x_data)}

    for group_idx, (group_val, data) in enumerate(groups.items()):
        if data.is_empty():
            continue

        if not group_col:
            group_val = ""
        else:
            group_val = "-".join(group_val)

        x = np.array([x_map[v] for v in data[x_col]], dtype=float) + (group_idx - (n_groups - 1) / 2) * offset

        markerline, stemlines, baseline = ax.stem(
            x,
            data[y_col].to_numpy(),
            label=group_val
        )
        colors = plt.cm.viridis(np.linspace(0, 1, n_groups))[group_idx]
        stemlines.set_color(colors)
        markerline.set_color(colors)
        if group_col == "virtual_sampling_method" and group_val != "none":
            stemlines.set_linestyle("--")

    ax.set_xticks(list(x_map.values()))
    ax.set_xticklabels(list(x_map.keys()), rotation=45, ha="right")

    ax.grid(True, axis="y", alpha=0.25)
    if group_col:
        ax.legend()
    plt.show()


def main() -> int:
    print("Starting...")

    args = build_argument_parser().parse_args()

    lf = load_data(args.data_dir)

    if lf is None:
        print("No valid data files found.")
        return 1

    if args.debug:
        data_rows = lf.select(pl.len()).collect().item()
        print(f"Found {data_rows} rows data.")

    """
    'dataset_name': ['Aneurism1', 'Aneurism2', 'Head1', 'Head2', 'ML1', 'ML2', 'Sphere', 'Tree1', 'Tree2'], 
    'level': [0, 1], 
    'method': ['cr', 'cubicB', 'cubicBf', 'lin', 'quadB', 'quadBf'], 
    'lighting_enabled': [False, True], 
    'true_samples': [1, 5, 10, 15], 
    'virtual_sampling_method': ['crvs', 'hermvs', 'linvs', 'monhermvs', 'none'], 
    'virtual_samples': [0, 1, 6, 12, 18], 
    'trans_type': ['rot', 'trans'], 
    'trans_axis': ['diag', 'x', 'y', 'z']
    """
    filter_by = {
        "dataset_name": "Tree2",
        "level": 0,
        # "method": "lin",
        "lighting_enabled": False,
        # "true_samples": 15,
        # "virtual_sampling_method": "none",
        # "virtual_samples": 0,
        # "trans_type": "rot",
        "trans_axis": ['diag', 'x', 'y'],
    }

    group_by = [
        # "",
        "method",
        "virtual_sampling_method"
    ]
    baselines = {
        "method": "lin",
        "virtual_sampling_method": "none",
    }
    plf = process_data(
        lf,
        filter_by,
        group_by,
        debug=args.debug,
        one_frame=False,
        baselines=None,
        use_frame=False,
        is_long=False
    )
    # show_plot(plf, "frame_idx", "blti_0")
    show_stem(plf, "method", "blti_0", "virtual_sampling_method")
    # print(plf.collect().to_dicts())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
