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

import dash
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import ALL, Input, Output, State, ctx, dcc, html

from blti import BLTI_K, GAUSSIAN_SCALES
from common import COL_META, ColMeta, FrameBasic

COL_META["band_idx"] = ColMeta(
    label="Band Center Scale",
    values={
        f"blti_{k}": (f"{np.sqrt(GAUSSIAN_SCALES[k] * GAUSSIAN_SCALES[k + 1]):.2f}")
        for k in range(BLTI_K)
    },
)

COL_META["blti"] = ColMeta(
    label="BLTI",
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--data-dir",
    type=Path,
    required=True,
    help="Path to parquet data",
)
args = parser.parse_args()

print("Starting...")

parquet_files = sorted(args.data_dir.glob("*.parquet"))

print(f"Found {len(parquet_files)} parquet files:")

lf = pl.scan_parquet(parquet_files).with_columns(
    pl.concat_str(["method", "virtual_sampling_method"], separator="-").alias(
        "sampling_method"
    )
)

FILTER_COLS = [col.name for col in fields(FrameBasic)]
FILTER_VALUES = {}
for col in FILTER_COLS:
    values = lf.select(pl.col(col).unique()).collect().to_series().to_list()

    FILTER_VALUES[col] = COL_META[col].order_vals(values)

METRIC_COLS = [f"blti_{i}" for i in range(BLTI_K)]

Y_COL = "blti"
X_COL = "band_idx"
GROUP_COL = "sampling_method"
FACET_COLS = [
    "method",
    "dataset_name",
    "lighting_enabled",
    "level",
    "trans_type",
    "trans_axis",
]

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            (
                "Evaluating Temporal Coherance of reconstruction methods in Volume Rendering"
            ),
            style={
                "textAlign": "center",
                "fontFamily": "sans-serif",
                "marginBottom": "10px",
            },
        ),
        html.Div(
            [
                html.H4("Filter By:"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    f"{COL_META[filter_col].label}: ",
                                    style={"fontWeight": "bold"},
                                ),
                                dcc.Dropdown(
                                    id={
                                        "type": "filter-dropdown",
                                        "column": filter_col,
                                    },
                                    options=[
                                        {
                                            "label": COL_META[filter_col].get_val_label(
                                                val
                                            ),
                                            "value": val,
                                        }
                                        for val in FILTER_VALUES[filter_col]
                                    ],
                                    value=(
                                        ["none"]
                                        if filter_col == "virtual_sampling_method"
                                        else None
                                    ),
                                    clearable=True,
                                    debounce=True,
                                    closeOnSelect=False,
                                    multi=True,
                                ),
                            ],
                            style={
                                "flex": "1",
                                "minWidth": "0",
                                "margin": "auto 2px",
                            },
                        )
                        for filter_col in FILTER_COLS
                    ],
                    style={
                        "display": "flex",
                        "width": "100%",
                        "marginBottom": "15px",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "Facet By: ",
                                    style={"fontWeight": "bold"},
                                ),
                                dcc.Dropdown(
                                    id="facet-dropdown",
                                    options=[
                                        {
                                            "label": COL_META[facet_col].label,
                                            "value": facet_col,
                                        }
                                        for facet_col in FACET_COLS
                                    ],
                                    value="method",
                                    clearable=False,
                                ),
                            ],
                            style={
                                "minWidth": "0",
                                "flex": "1",
                                "margin": "auto 2px",
                            },
                        ),
                        html.Div(
                            [
                                html.Button("Download CSV", id="download-csv-btn"),
                                dcc.Store(id="csv-data"),
                                dcc.Download(id="download-csv"),
                            ],
                            style={
                                "minWidth": "0",
                                "flex": "1",
                                "margin": "auto 2px",
                            },
                        ),
                    ],
                    style={
                        "width": "100%",
                        "display": "flex",
                    },
                ),
            ],
            style={
                "width": "95%",
                "margin": "10px auto",
            },
        ),
        html.Div(
            [dcc.Graph(id="main-chart")],
            style={
                "width": "95%",
                "margin": "20px auto",
            },
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="baseline-chart")],
                    style={
                        "flex": "1",
                        "minWidth": "0",
                    },
                ),
                html.Div(
                    [dcc.Graph(id="cmp-chart")],
                    style={
                        "flex": "1",
                        "minWidth": "0",
                    },
                ),
            ],
            style={"width": "95%", "display": "flex", "margin": "20px auto"},
        ),
    ]
)


def filter_exprs(filter_list) -> tuple[list, list]:
    method_filters = []
    other_filters = []
    filter_dict = {
        filter_item["id"]["column"]: filter_item["value"] for filter_item in filter_list
    }
    filter_virtual_method = filter_dict.get("virtual_sampling_method", [])
    filter_true_samples = filter_dict.get("true_samples", [])
    filter_virtual_samples = filter_dict.get("virtual_samples", [])
    if (
        filter_true_samples
        and 1 not in filter_true_samples
        and (
            not filter_virtual_method or any(x != "none" for x in filter_virtual_method)
        )
    ):
        filter_dict["true_samples"].append(1)
    if (
        filter_virtual_samples
        and 0 not in filter_virtual_samples
        and (not filter_virtual_method or "none" in filter_virtual_method)
    ):
        filter_dict["virtual_samples"].append(0)

    for col_name, filter_val in filter_dict.items():
        if not filter_val:
            continue

        if len(filter_val) == len(FILTER_VALUES[col_name]):
            continue

        if col_name in ["method", "virtual_sampling_method", "virtual_samples"]:
            method_filters.append(pl.col(col_name).is_in(filter_val))
        else:
            other_filters.append(pl.col(col_name).is_in(filter_val))

    return method_filters, other_filters


def process_data(
    lf: pl.LazyFrame, filters: list, groupby: list[str], aggs: list[str]
) -> pl.LazyFrame:
    idx_cols = groupby + [GROUP_COL]
    select_cols = idx_cols + METRIC_COLS
    agg_map = {
        "mean": pl.col(Y_COL).mean(),
        "median": pl.col(Y_COL).median(),
        "p10": pl.col(Y_COL).quantile(0.10),
        "p90": pl.col(Y_COL).quantile(0.90),
    }
    if len(aggs) == 1:
        agg_exprs = agg_map[aggs[0]]
    else:
        agg_exprs = [agg_map[agg_expr].alias(agg_expr) for agg_expr in aggs]

    return (
        lf.filter(pl.all_horizontal(filters))
        .select(select_cols)
        .unpivot(on=METRIC_COLS, index=idx_cols, variable_name=X_COL, value_name=Y_COL)
        .group_by(idx_cols + [X_COL])
        .agg(agg_exprs)
        .sort(idx_cols + [X_COL])
    )


def line_chart(df: pl.DataFrame, sel_facet: str, sel_method: str) -> go.Figure:
    categroy_orders = {
        col: COL_META[col].order_vals(df.get_column(col).unique().to_list())
        if col in COL_META
        else df.get_column(col).unique().to_list()
        for col in [sel_facet, X_COL]
    }

    fig = px.line(
        df,
        x=X_COL,
        y="mean",
        markers=True,
        facet_col=sel_facet,
        facet_col_wrap=1,
        facet_row_spacing=0.05,
        category_orders=categroy_orders,
        labels={
            "mean": COL_META[Y_COL].label,
            X_COL: COL_META[X_COL].label,
        },
    )
    fig.data[0].update(name="Mean", showlegend=True)

    for i, val in enumerate(categroy_orders[sel_facet]):
        sdf = df.filter(pl.col(sel_facet) == val)

        x = sdf[X_COL].to_list()
        p10 = sdf["p10"].to_list()
        p90 = sdf["p90"].to_list()

        row_idx = i + 1
        col_idx = 1

        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=p90 + p10[::-1],
                fill="toself",
                fillcolor="rgba(100, 150, 255, 0.2)",
                line={"color": "rgba(0,0,0,0)"},
                name="10-90% envelope",
                hoverinfo="skip",
                showlegend=(i == 0),
            ),
            row=row_idx,
            col=col_idx,
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=sdf["median"],
                mode="lines+markers",
                name="Median",
                line=dict(color="orange"),
                showlegend=(i == 0),
            ),
            row=row_idx,
            col=col_idx,
        )

    dynamic_height = (len(categroy_orders[sel_facet]) * 250) + 150

    fig.update_layout(
        title={
            "text": (COL_META[GROUP_COL].get_val_label(sel_method)),
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        height=dynamic_height,
        xaxis_title=COL_META[X_COL].label,
        yaxis_title=COL_META[Y_COL].label,
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(BLTI_K)),
        ticktext=list(COL_META[X_COL].values.values()),
    )

    fig.for_each_annotation(
        lambda annotation: annotation.update(
            text=COL_META[sel_facet].get_val_label(
                next(
                    v
                    for v in categroy_orders[sel_facet]
                    if str(v) == annotation.text.split("=")[-1]
                )
            )
            if sel_facet in COL_META
            else annotation.text
        )
    )

    fig.data = (
        fig.data[-len(categroy_orders[sel_facet]) :]
        + fig.data[: -len(categroy_orders[sel_facet])]
    )

    return fig


def lollipop_chart(df: pl.DataFrame, sel_facet: str) -> go.Figure:
    category_orders = {
        col: COL_META[col].order_vals(df.get_column(col).unique().to_list())
        if col in COL_META
        else df.get_column(col).unique().to_list()
        for col in [GROUP_COL, sel_facet, X_COL]
    }

    pdf = df.to_pandas()

    x_map = {value: i for i, value in enumerate(category_orders[X_COL])}
    pdf["_x"] = pdf[X_COL].map(x_map)

    group_order = category_orders[GROUP_COL]
    n_groups = len(group_order)
    group_width = 0.7
    group_map = {group: i for i, group in enumerate(group_order)}

    pdf["_x"] += pdf[GROUP_COL].map(
        lambda group: (group_map[group] - (n_groups - 1) / 2) * group_width / n_groups
    )

    fig = px.scatter(
        pdf,
        x="_x",
        y=f"{Y_COL}_diff",
        color=GROUP_COL,
        facet_col=sel_facet,
        facet_col_wrap=2,
        facet_col_spacing=0.12,
        facet_row_spacing=0.12,
        category_orders=category_orders,
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        custom_data=[GROUP_COL, sel_facet, X_COL],
        labels={
            f"{Y_COL}_diff": "Δ BLTI to linear",
            "_x": COL_META[X_COL].label,
        },
    )

    fig.update_traces(
        mode="markers",
        marker=dict(size=10),
    )

    for trace in list(fig.data):
        if trace.mode != "markers":
            continue

        if trace.customdata is None:
            continue

        stem_x = []
        stem_y = []

        for point in trace.customdata:
            group = point[0]
            facet = point[1]
            x_category = point[2]

            rows = pdf[
                (pdf[GROUP_COL] == group)
                & (pdf[sel_facet] == facet)
                & (pdf[X_COL] == x_category)
            ]

            if rows.empty:
                continue

            x = rows["_x"].iloc[0]
            y = rows[f"{Y_COL}_diff"].iloc[0]

            stem_x.extend([x, x, None])
            stem_y.extend([0, y, None])

        fig.add_trace(
            go.Scatter(
                x=stem_x,
                y=stem_y,
                mode="lines",
                line=dict(
                    color=trace.marker.color,
                    width=2,
                ),
                showlegend=False,
                hoverinfo="skip",
                xaxis=trace.xaxis,
                yaxis=trace.yaxis,
            )
        )

    fig.add_hline(
        y=0,
        line_width=1,
        line_color="black",
    )

    fig.update_xaxes(
        matches=None,
        showticklabels=True,
        tickmode="array",
        tickvals=list(range(BLTI_K)),
        ticktext=list(COL_META[X_COL].values.values()),
        showgrid=False,
        title_text=COL_META[X_COL].label,
    )

    fig.update_yaxes(
        matches=None,
        showticklabels=True,
        showline=True,
        linecolor="black",
        linewidth=1,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1,
        title_text="Δ BLTI to linear",
    )

    dynamic_height = (len(category_orders[sel_facet]) / 2 * 250) + 200

    fig.update_layout(
        template="plotly_white",
        height=dynamic_height,
        # title="Δ BLTI Delta vs line",
        legend_title_text=COL_META[GROUP_COL].label,
    )

    fig.for_each_annotation(
        lambda annotation: annotation.update(
            text=COL_META[sel_facet].get_val_label(
                next(
                    v
                    for v in category_orders[sel_facet]
                    if str(v) == annotation.text.split("=")[-1]
                )
            )
            if sel_facet in COL_META
            else annotation.text
        )
    )

    fig.for_each_trace(
        lambda trace: trace.update(
            name=COL_META[GROUP_COL].get_val_label(trace.name)
            if trace.name in group_order
            else trace.name
        )
    )

    return fig


@app.callback(
    Output("main-chart", "figure"),
    Output("baseline-chart", "figure"),
    Output("csv-data", "data"),
    Input(
        {"type": "filter-dropdown", "column": ALL},
        "value",
    ),
    Input("facet-dropdown", "value"),
)
def update_main_chart(filter_vals, sel_facet):
    method_filters, other_filters = filter_exprs(ctx.inputs_list[0])
    groupby = [sel_facet]
    filters = method_filters + other_filters
    flf = process_data(lf, filters, groupby, ["mean"])

    dflf = flf

    baseline_filters = [(pl.col("sampling_method") == "lin-none")] + other_filters
    baggs = ["mean", "median", "p10", "p90"]
    blf = process_data(lf, baseline_filters, groupby, baggs)

    filters = (
        method_filters + other_filters
    )
    merge_cols = [X_COL]
    if sel_facet != "method":
        merge_cols.append(sel_facet)
    dflf = (
        dflf.join(blf.select(merge_cols + ["mean"]), on=merge_cols, how="left")
        .with_columns((pl.col(Y_COL) - pl.col("mean")).alias(f"{Y_COL}_diff"))
        .drop(["mean", Y_COL])
    )

    fdf = flf.collect().sort(groupby + [GROUP_COL, X_COL])
    dfdf = dflf.collect().sort([sel_facet, GROUP_COL, X_COL])
    bdf = blf.collect().sort([sel_facet, X_COL])

    return (
        lollipop_chart(dfdf, sel_facet),
        line_chart(bdf, sel_facet, "lin-none"),
        fdf.write_csv(),
    )


@app.callback(
    Output("download-csv", "data"),
    Input("download-csv-btn", "n_clicks"),
    State("csv-data", "data"),
    prevent_initial_call=True,
)
def download_data(n_clicks, csv_data):
    return {
        "content": csv_data,
        "filename": "data.csv",
        "type": "text/csv",
    }


@app.callback(
    Output("cmp-chart", "figure"),
    Input("main-chart", "clickData"),
    Input(
        {"type": "filter-dropdown", "column": ALL},
        "value",
    ),
    Input("facet-dropdown", "value"),
)
def update_cmp_chart(click_data, filter_vals, sel_facet):
    if click_data is None:
        return go.Figure()

    point = click_data["points"][0]

    custom_data = point["customdata"]

    method_filters, other_filters = filter_exprs(ctx.inputs_list[1])
    groupby = [sel_facet]
    sel_method = custom_data[0]

    filters = other_filters + [(pl.col("sampling_method") == sel_method)]
    baggs = ["mean", "median", "p10", "p90"]
    blf = process_data(lf, filters, groupby, baggs)
    bdf = blf.collect()

    return line_chart(bdf, sel_facet, sel_method)


server = app.server

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
    )
