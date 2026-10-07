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
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from blti import BLTI_K, GAUSSIAN_SCALES
from common import FrameBasic

SEQ_COLS = [field.name for field in fields(FrameBasic)]
METRIC_COLS = [f"blti_{i}" for i in range(BLTI_K)]

X_COL = "band_idx"
Y_COL = "blti"
GROUP_COL = "sample_method"

X_LABELS = [
    f"{np.sqrt(GAUSSIAN_SCALES[k] * GAUSSIAN_SCALES[k + 1]):.2f}" for k in range(BLTI_K)
]
AGG_EXPRS = {
    "mean": lambda expr: expr.mean(),
    "median": lambda expr: expr.median(),
    "p10": lambda expr: expr.quantile(0.10),
    "p90": lambda expr: expr.quantile(0.90),
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

    if agg_exprs:
        return lf.group_by(group_by).agg(agg_exprs)

    return lf.group_by(group_by).agg(pl.all())


def to_long(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.unpivot(
        on=METRIC_COLS, index=SEQ_COLS, variable_name=X_COL, value_name=Y_COL
    ).with_columns(pl.col(X_COL).str.strip_prefix("blti_").cast(pl.UInt8))


def diff_data(
    lf: pl.LazyFrame, facet_cols: list, group_cols: list, baselines: dict[str, Any]
) -> pl.DataFrame:
    dlf = agg_data(lf, SEQ_COLS, {"mean": {metric_col: None for metric_col in METRIC_COLS}})
    
    dlf = to_long(dlf)
    
    group_by = list(dict.fromkeys([*facet_cols, *group_cols, X_COL]))

    dlf = agg_data(dlf, group_by, {"mean": {Y_COL: None}}).sort(group_by)

    blf = filter_data(dlf, baselines).select(
        [*facet_cols, X_COL, pl.col(Y_COL).alias("baseline")]
    )

    dlf = (
        dlf.join(blf, on=[*facet_cols, X_COL], how="inner")
        .with_columns((pl.col(Y_COL) - pl.col("baseline")).alias("diff"))
        .drop([Y_COL, "baseline"])
    )

    return dlf.collect()


def diff_plot(df: pl.DataFrame, facet_cols: list, group_cols: list) -> None:
    if df.is_empty():
        print("No valid data found.")
        return

    facets = df.select(facet_cols).unique().sort(facet_cols)

    ncols = 2
    nrows = math.ceil(len(facets) / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8 * ncols, 6 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    colors = plt.cm.tab10(
        np.linspace(0, 1, max(1, df.select(group_cols).unique().height))
    )

    for ax, facet_idx in zip(axes.flat, facets.iter_rows()):
        rows = df.filter(
            pl.all_horizontal(
                [pl.col(col) == val for col, val in zip(facet_cols, facet_idx)]
            )
        ).sort([*group_cols, X_COL])

        groups = rows.select(group_cols).unique().sort(group_cols)

        n_groups = len(groups)
        width = 0.6 / max(1, n_groups)

        for i, (group_idx, color) in enumerate(zip(groups.iter_rows(), colors)):
            data = rows.filter(
                pl.all_horizontal(
                    [pl.col(col) == val for col, val in zip(group_cols, group_idx)]
                )
            ).sort(X_COL)

            if data.is_empty():
                continue

            x = data[X_COL].to_numpy()

            x = x + (i - (n_groups - 1) / 2) * width

            diff = data["diff"].to_numpy()

            # lollipop stem
            ax.vlines(
                x,
                0,
                diff,
                color=color,
                alpha=0.6,
                linewidth=1.5,
            )

            # lollipop head
            ax.scatter(
                x,
                diff,
                color=color,
                s=60,
                label="-".join(map(str, group_idx)),
                zorder=3,
            )

        ax.axhline(
            0,
            color="black",
            linewidth=1,
        )

        ax.set_xticks(range(BLTI_K))
        ax.set_xticklabels(X_LABELS)

        ax.set_title("-".join(map(str, facet_idx)))

        ax.grid(
            axis="y",
            alpha=0.2,
        )

    for ax in axes.flat[len(facets) :]:
        ax.remove()

    for ax in axes[-1, :]:
        ax.set_xlabel("Band Center")

    for ax in axes[:, 0]:
        ax.set_ylabel("ΔBLTI to Linear")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Method",
            loc="lower right",
            bbox_to_anchor=(0.98, 0.02),
        )

    plt.tight_layout()
    plt.show()


def seq_plot(df: pl.DataFrame):
    groups = df[GROUP_COL].unique().sort()

    ncols = 2
    nrows = math.ceil(len(groups) / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for ax, group_idx in zip(axes.flat, groups):
        group = df.filter(pl.col(GROUP_COL) == group_idx).sort(X_COL)

        x = group[X_COL].to_list()
        mean = group["mean"].to_numpy()
        median = group["median"].to_numpy()
        p10 = group["p10"].to_numpy()
        p90 = group["p90"].to_numpy()

        ax.plot(
            x,
            mean,
            marker="o",
            label="Mean",
        )

        ax.plot(
            x,
            median,
            # marker="o",
            label="Median",
        )

        ax.fill_between(
            range(len(x)),
            p10,
            p90,
            alpha=0.2,
            label="P10–P90",
        )

        ax.set_title(group_idx)
        ax.grid(alpha=0.2)

    for ax in axes.flat[len(groups) :]:
        ax.remove()

    fig.supxlabel("Band Center")
    fig.supylabel("BLTI")

    axes.flat[0].legend()

    plt.tight_layout()
    plt.show()


def load_col_data(lf: pl.LazyFrame, cols: list) -> dict[str, list]:
    return {
        col: lf.select(pl.col(col).unique().sort()).collect()[col].to_list()
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
    return parser


def main() -> int:
    print("Starting...")

    args = build_argument_parser().parse_args()

    lf = load_data(args.data_dir)

    if lf is None:
        print("No valid data files found.")
        return 1

    data_rows = lf.select(pl.len()).collect().item()
    print(f"Found {data_rows} rows data.")

    # filter
    # filter_values = load_col_data(lf, SEQ_COLS)
    # print(filter_values)
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
        # "dataset_name": ['Aneurism1', 'Aneurism2', 'Head1', 'Head2', 'ML1', 'ML2'],
        "level": 0,
        # "method": ["lin", "quadB"],
        "lighting_enabled": True,
        "true_samples": 15,
        "virtual_sampling_method": "none",
        "virtual_samples": 0,
        "trans_type": "rot",
        "trans_axis": "y",
    }

    # filter_by = get_user_filters(filter_data)

    lf = filter_data(lf, filter_by)
    data_rows, data_seq = (
        lf.select([pl.len(), pl.struct(SEQ_COLS).n_unique()]).collect().row(0)
    )
    print(f"Filtered data: {data_rows} rows, {data_seq} sequences.")

    if data_rows == 0:
        print("No data after filtering.")
        return 1

    plot_type = "diff"

    if plot_type == "diff":
        facet_cols = ["dataset_name", "level"]
        group_cols = [
            "method",
            "virtual_sampling_method",
            # "true_samples",
            # "virtual_samples",
        ]
        baselines = {
            "method": "lin",
            "virtual_sampling_method": "none",
            # "true_samples": 15,
            # "virtual_samples": 0,
        }
        df = diff_data(lf, facet_cols, group_cols, baselines)

    # df.write_csv("output.csv")

    # sequence line plot

    # lf = lf.group_by(SEQ_COLS).agg([pl.col(metric_col).mean() for metric_col in METRIC_COLS])

    # print(lf.select(pl.len()).collect().item())

    # draw_dash(filter_data, SEQ_COLS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
