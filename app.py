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

from pathlib import Path

import dash
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import ALL, Input, Output, State, dcc, html

from blti import BLTI_K, GAUSSIAN_SCALES
from common import (
    DATASET_LABELS,
    DATASET_ORDER,
    METHOD_LABELS,
    METHOD_ORDER,
    TRANS_AXIS_LABELS,
    TRANS_TYPE_LABELS,
    VIRTUAL_METHOD_LABELS,
    VIRTUAL_METHOD_ORDER,
)

BASE_DIR = Path(__file__).resolve().parent

print("Starting...")

SAMPLING_METHOD_LABEL = {}
for m in METHOD_ORDER:
    s = METHOD_LABELS.get(m, m)
    for vm in VIRTUAL_METHOD_ORDER:
        if vm != "none":
            s = f"{s} + {VIRTUAL_METHOD_LABELS.get(vm, vm)}"
        SAMPLING_METHOD_LABEL[f"{m}-{vm}"] = s

lf = (
    pl.scan_parquet(BASE_DIR / "data" / "parquet")
    .with_columns(
        pl.concat_str(
            ["method", "virtual_sampling_method"],
            separator="-"
        ).alias("sampling_method")
    )
)

COL_META = {
    "dataset_name": {
        "label": "Dataset",
        "values": DATASET_LABELS,
        "order": DATASET_ORDER
    },
    "level": {
        "label": "Downsamples",
        "values": {i: f"Downloaded {i}" for i in range(3)},
        "order": list(range(3))
    },
    "method": {
        "label": "True Method",
        "values": METHOD_LABELS,
        "order": METHOD_ORDER
    },
    "lighting_enabled": {
        "label": "Illumination",
        "values": {True: "Lighting enabled", False: "Lighting disabled", "True": "Lighting enabled", "False": "Lighting disabled"},
        "order": [False, True]
    },
    "virtual_sampling_method": {
        "label": "Virtual Method",
        "values": VIRTUAL_METHOD_LABELS,
        "order": VIRTUAL_METHOD_ORDER
    },
    "trans_type": {
        "label": "Transform Type",
        "values": TRANS_TYPE_LABELS,
        "order": list(TRANS_TYPE_LABELS.keys())
    },
    "trans_axis": {
        "label": "Transform Axis",
        "values": TRANS_AXIS_LABELS,
        "order": list(TRANS_AXIS_LABELS.keys())
    },
    "band_idx": {
        "label": "Band Center Scale",
        "values": {k: (f"{np.sqrt(GAUSSIAN_SCALES[k] * GAUSSIAN_SCALES[k + 1]):.2f}") for k in range(BLTI_K)},
        "order": list(range(BLTI_K))
    },
    "sampling_method": {
        "label": "Method",
        "values": SAMPLING_METHOD_LABEL,
        "order": list(SAMPLING_METHOD_LABEL.keys())
    },
    "blti": {
        "label": "BLTI",
        "values": {},
        "order": []
    }
}

FILTER_COLS = [
    "dataset_name",
    "level",
    "method",
    "lighting_enabled",
    "virtual_sampling_method",
    "trans_type",
    "trans_axis",
]

FILTER_VALUES = {}
for col in FILTER_COLS:
    values = (
        lf
        .select(pl.col(col).unique())
        .collect()
        .to_series()
        .to_list()
    )

    FILTER_VALUES[col] = [v for v in COL_META[col]["order"] if v in values]

METRIC_COLS = [f"blti_{i}" for i in range(BLTI_K)]

Y_COL = "blti"
X_COL = "band_idx"
GROUP_COL = "sampling_method"
PLOT_COLS = ["method", "dataset_name", "lighting_enabled", "level", "trans_type", "trans_axis"]

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
                                    f"{COL_META[filter_col]["label"]}: ",
                                    style={"fontWeight": "bold"},
                                ),
                                dcc.Dropdown(
                                    id={
                                        "type": "filter-dropdown",
                                        "column": filter_col,
                                    },
                                    options=[
                                        {
                                            "label": COL_META[filter_col]["values"][val],
                                            "value": val,
                                        }
                                        for val in FILTER_VALUES[filter_col]
                                    ],
                                    value=(["none"] if filter_col == "virtual_sampling_method" else None),
                                    clearable=True,
                                    debounce=True,
                                    closeOnSelect=False,
                                    multi=True
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
                                    "Plot By: ",
                                    style={"fontWeight": "bold"},
                                ),
                                dcc.Dropdown(
                                    id="facet-dropdown",
                                    options=[
                                        {
                                            "label": COL_META[plot_col]["label"],
                                            "value": plot_col,
                                        }
                                        for plot_col in PLOT_COLS
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
                        )
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
                )
            ],
            style={"width": "95%", "display": "flex", "margin": "20px auto"},
        ),
    ]
)

def process_data(lf: pl.LazyFrame, filters: list, groupby: list[str], aggs: list[str]) -> pl.LazyFrame:
    idx_cols = groupby + [GROUP_COL]
    select_cols = idx_cols+ METRIC_COLS
    agg_map = {
        "mean": pl.col(Y_COL).mean(),
        "median": pl.col(Y_COL).median(),
        "p10": pl.col(Y_COL).quantile(0.10),
        "p90": pl.col(Y_COL).quantile(0.90)
    }
    if len(aggs) == 1:
        agg_exprs = agg_map[aggs[0]]
    else:
        agg_exprs = [agg_map[agg_expr].alias(agg_expr) for agg_expr in aggs]

    return (
        lf.filter(pl.all_horizontal(filters))
        .select(select_cols)
        .unpivot(
            on=METRIC_COLS,
            index=idx_cols,
            variable_name=X_COL,
            value_name=Y_COL
        )
        .group_by(idx_cols + [X_COL])
        .agg(agg_exprs)
        .sort(idx_cols + [X_COL])
    )

def filter_exprs(filter_vals) -> (list, list):
    method_filters = []
    other_filters = []
    for col_name, filter_val in zip(FILTER_COLS, filter_vals):
        if not filter_val:
            continue

        if len(filter_val) == len(FILTER_VALUES[col_name]):
            continue

        if col_name in ["method", "virtual_sampling_method"]:
            method_filters.append(pl.col(col_name).is_in(filter_val))
        else:
            other_filters.append(pl.col(col_name).is_in(filter_val))

    return method_filters, other_filters

def line_chart(df: pl.DataFrame, sel_facet: str, sel_method: str) -> go.Figure:
    facet_vals = df[sel_facet].unique().to_list()

    fig = px.line(
        df,
        x=X_COL,
        y="mean",
        markers=True,
        facet_col=sel_facet,
        facet_col_wrap=1,
        category_orders={
            sel_facet: [x for x in COL_META[sel_facet]["order"] if x in facet_vals]
        },
        labels={
            "mean": COL_META[Y_COL]["label"],
            X_COL: COL_META[X_COL]["label"],
        }
    )
    fig.data[0].update(name="Mean", showlegend=True)

    for i, val in enumerate(facet_vals):
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
            row=row_idx, col=col_idx
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
            row=row_idx, col=col_idx
        )

    dynamic_height = (len(facet_vals) * 250) + 150

    fig.update_layout(
        title={
            "text": (
                COL_META[GROUP_COL]["values"].get(sel_method, sel_method)
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        height=dynamic_height,
        xaxis_title=COL_META[X_COL]["label"],
        yaxis_title=COL_META[Y_COL]["label"],
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(BLTI_K)),
        ticktext=list(COL_META[X_COL]["values"].values())
    )

    fig.for_each_annotation(
        lambda annotation: annotation.update(
            text=COL_META[sel_facet]["values"].get(
                annotation.text.split("=")[-1],
                annotation.text,
            )
        )
    )

    fig.data = fig.data[-len(facet_vals):] + fig.data[:-len(facet_vals)]

    return fig

def bar_chart(df: pl.DataFrame, sel_facet: str) -> go.Figure:
    facet_vals = df[sel_facet].unique().to_list()
    fig = px.bar(
        df,
        x=X_COL,
        y=f"{Y_COL}_diff",
        color=GROUP_COL,
        barmode="group",
        facet_col=sel_facet,
        facet_col_wrap=2,
        category_orders={
            sel_facet: [x for x in COL_META[sel_facet]["order"] if x in facet_vals]
        },
        labels={
            f"{Y_COL}_diff": "Δ BLTI of linear",
            X_COL: COL_META[X_COL]["label"],
        },
        custom_data=[GROUP_COL, sel_facet],
        color_discrete_sequence=px.colors.qualitative.Alphabet,
    )

    fig.add_hline(
        y=0,
        line_width=1,
        line_color="black",
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(BLTI_K)),
        ticktext=list(COL_META[X_COL]["values"].values()),
    )

    dynamic_height = (len(facet_vals) / 2 * 250) + 150
    

    fig.update_layout(
        template="plotly_white",
        height=dynamic_height,
        title="Δ BLTI Delta vs line",
        legend_title_text=COL_META[GROUP_COL]["label"],
    )

    fig.for_each_annotation(
        lambda annotation: annotation.update(
            text=COL_META[sel_facet]["values"].get(
                annotation.text.split("=")[-1],
                annotation.text,
            )
        )
    )

    fig.for_each_trace(
        lambda trace: trace.update(
            name=COL_META[GROUP_COL]["values"].get(
                trace.name,
                trace.name,
            )
        )
    )
    return fig

@app.callback(
    Output("main-chart", "figure"),
    Output("baseline-chart", "figure"),
    Input(
        {"type": "filter-dropdown", "column": ALL},
        "value",
    ),
    Input("facet-dropdown", "value"),
)
def update_main_chart(
    filter_vals,
    sel_facet
):
    method_filters, other_filters = filter_exprs(filter_vals)
    groupby = [sel_facet]

    baseline_filters = other_filters + [(pl.col("sampling_method") == "lin-none")]
    baggs = ["mean", "median", "p10", "p90"]
    blf = process_data(lf, baseline_filters, groupby, baggs)

    filters = method_filters + other_filters + [(pl.col("sampling_method") != "lin-none")]
    merge_cols = [X_COL]
    if sel_facet != "method":
        merge_cols.append(sel_facet)
    flf = (
        process_data(lf, filters, groupby, ["mean"])
        .join(
            blf.select(merge_cols + ["mean"]),
            on=merge_cols,
            how="left"
        )
        .with_columns(
            (pl.col(Y_COL) - pl.col("mean")).alias(f"{Y_COL}_diff")
        )
        .drop(["mean", Y_COL])
    )
    fdf = flf.collect()
    bdf = blf.collect()

    return bar_chart(fdf, sel_facet), line_chart(bdf, sel_facet, "lin-none")

@app.callback(
    Output("cmp-chart", "figure"),
    Input("main-chart", "clickData"),
    Input(
        {"type": "filter-dropdown", "column": ALL},
        "value",
    ),
    Input("facet-dropdown", "value"),
)
def update_cmp_chart(
    click_data,
    filter_vals,
    sel_facet
):
    if click_data is None:
        return go.Figure()

    point = click_data["points"][0]

    custom_data = point["customdata"]

    method_filters, other_filters = filter_exprs(filter_vals)
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
        debug=False,
        host="0.0.0.0",
    )