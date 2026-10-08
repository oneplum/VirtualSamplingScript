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


def to_long(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.unpivot(
        on=METRIC_COLS, index=SEQ_COLS, variable_name=X_COL, value_name=Y_COL
    ).with_columns(pl.col(X_COL).str.strip_prefix("blti_").cast(pl.UInt8))


def process_data(
    lf: pl.LazyFrame,
    filters: dict[str, Any],
    facet_cols: list,
    group_cols: list,
    baselines: dict[str, Any] | None = None,
    use_frame: bool = False,
    debug: bool = False,
) -> pl.LazyFrame:
    flf = filter_data(lf, filters)

    if debug:
        data_rows, data_seq = (
            flf.select([pl.len(), pl.struct(SEQ_COLS).n_unique()]).collect().row(0)
        )
        print(f"Filtered data: {data_rows} rows, {data_seq} sequences.")

    if not use_frame:
        flf = agg_data(
            flf, SEQ_COLS, {"mean": {metric_col: None for metric_col in METRIC_COLS}}
        )

        if debug:
            data_rows, data_seq = (
                flf.select([pl.len(), pl.struct(SEQ_COLS).n_unique()]).collect().row(0)
            )
            print(f"Sequence data: {data_rows} rows, {data_seq} sequences.")

    flf = to_long(flf)

    group_by = list(dict.fromkeys([*facet_cols, *group_cols, X_COL]))

    if use_frame:
        aggs = {
            "mean": {Y_COL: "mean"},
            "median": {Y_COL: "median"},
            "p10": {Y_COL: "p10"},
            "p90": {Y_COL: "p90"},
        }
    else:
        aggs = {"mean": {Y_COL: None}}

    flf = agg_data(flf, group_by, aggs)

    if baselines and not use_frame:
        blf = filter_data(flf, baselines).select(
            [*facet_cols, X_COL, pl.col(Y_COL).alias("baseline")]
        )

        flf = (
            flf.join(blf, on=[*facet_cols, X_COL], how="inner", validate="m:1")
            .with_columns((pl.col(Y_COL) - pl.col("baseline")).alias(DELTA_COL))
            .drop([Y_COL, "baseline"])
        )

        flf = sort_data(flf, group_by)

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
        "dataset_name": "Head1",
        "level": 0,
        "method": ["lin", "quadB"],
        "lighting_enabled": True,
        "true_samples": 15,
        "virtual_sampling_method": "none",
        "virtual_samples": 0,
        "trans_type": "rot",
        "trans_axis": "y",
    }

    facet_cols = ["dataset_name", "level"]
    group_cols = [
        "method",
        "true_samples",
        "virtual_sampling_method",
        "virtual_samples",
    ]
    baselines = {
        "method": "lin",
        "virtual_sampling_method": "none",
        "true_samples": 15,
        "virtual_samples": 0,
    }
    plf = process_data(
        lf,
        filter_by,
        facet_cols,
        group_cols,
        baselines=None,
        debug=args.debug,
        use_frame=False,
    )
    print(plf.collect().to_dicts())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
