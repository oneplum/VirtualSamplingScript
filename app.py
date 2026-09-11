#!/usr/bin/env python3

"""
dataset_name,level,method,lighting_enabled,true_samples,virtual_sampling_method,virtual_samples,trans_type,trans_axis,
frame_idx,blti_0,blti_1,blti_2,blti_3,blti_4,blti_5
"""
import os
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, dcc, html

from blti import BLTI_K, GAUSSIAN_SCALES
from common import (
    DATASET_LABELS,
    METHOD_LABELS,
    METHOD_ORDER,
    TRANS_AXIS_LABELS,
    TRANS_TYPE_LABELS,
    VIRTUAL_METHOD_LABELS,
    VIRTUAL_METHOD_ORDER,
)

BASE_DIR = Path(__file__).resolve().parent

print("starting ...")
# load data
df = pd.read_parquet(BASE_DIR / "data" / "parquet")
df["sampling_method"] = (
    df[["method", "virtual_sampling_method"]].astype(str).agg("-".join, axis=1)
)
# df["trans"] = df[["trans_type", "trans_axis"]].astype(str).agg("-".join, axis=1)
print(f"data loaded, total: {len(df)} rows")

filter_cols: list[str] = []
metric_cols: list[str] = []
col_vals: dict[str, list] = {}
for col in df.columns:
    if col.startswith("blti_"):
        metric_cols.append(col)
    elif col not in ["sampling_method", "frame_idx"]:
        col_vals[col] = df[col].unique().tolist()
        if col not in ["true_samples", "virtual_samples"]:
            filter_cols.append(col)
col_label = {
    "dataset_name": "Dataset",
    "level": "Downsamples",
    "method": "True Sampling Method",
    "lighting_enabled": "Illumination",
    "true_samples": "Sampling Rate",
    "virtual_sampling_method": "Virtual Sampling Method",
    "sampling_method": "Method",
    "virtual_samples": "Virtual Samples",
    "trans_type": "Transformation Type",
    "trans_axis": "Transformation Axis",
    "trans": "Transformation",
    "frame_idx": "Frame Index",
    "band_idx": "Band Center Scale",
    "blti": "BLTI",
}
col_val_label = {
    "dataset_name": DATASET_LABELS,
    "level": {l: f"Downsampled {l}" for l in range(3)},
    "method": METHOD_LABELS,
    "lighting_enabled": {True: "Lighting on", False: "Lighting off"},
    "virtual_sampling_method": VIRTUAL_METHOD_LABELS,
    "sampling_method": {
        f"{tm}-{vm}": f"{METHOD_LABELS.get(tm, tm)} + {VIRTUAL_METHOD_LABELS.get(vm, vm)}"
        for tm in METHOD_ORDER
        for vm in VIRTUAL_METHOD_ORDER
    },
    "trans_type": TRANS_TYPE_LABELS,
    "trans_axis": TRANS_AXIS_LABELS,
    "trans": {
        f"{tt}-{ta}": f"{TRANS_TYPE_LABELS.get(tt, tt)} - {TRANS_AXIS_LABELS.get(ta, ta)}"
        for tt in TRANS_TYPE_LABELS
        for ta in TRANS_AXIS_LABELS
    },
    "band_idx": {
        k: f"{np.sqrt(GAUSSIAN_SCALES[k] * GAUSSIAN_SCALES[k + 1]):.2f}"
        for k in range(BLTI_K)
    },
}

# filter_cols = ["dataset_name", "level", "lighting_enabled", "method", "virtual_sampling_method", "trans_type", "trans_axis"]
facet_cols = ["dataset_name", "level", "lighting_enabled", "method"]
line_cols = ["sampling_method", "method", "virtual_sampling_method"]
x_col = "band_idx"
y_col = "blti"

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            "Evaluating Temporal Coherance of reconstruction methods in Volume Rendering ",
            style={
                "textAlign": "center",
                "fontFamily": "sans-serif",
                "marginBottom": "10px",
            },
        ),
        # html.P("xxxxx", style={"textAlign": "center", "color": "#666", "marginBottom": "20px"}),
        html.Div(
            [
                html.H4("Filter By:"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    f"{col_label.get(filter_col, filter_col)}: ",
                                    style={"fontWeight": "bold"},
                                ),
                                dcc.Dropdown(
                                    id={
                                        "type": "filter-dropdown",
                                        "column": filter_col,
                                    },
                                    options=[
                                        {
                                            "label": col_val_label.get(
                                                filter_col, {}
                                            ).get(val, val),
                                            "value": val,
                                        }
                                        for val in col_vals.get(filter_col, [])
                                    ],
                                    value=col_vals.get(filter_col, []),
                                    # placeholder=f"Filter ",
                                    clearable=False,
                                    debounce=True,
                                    closeOnSelect=False,
                                    multi=True,
                                ),
                            ],
                            style={"flex": "1", "minWidth": "0", "margin": "auto 2px"},
                        )
                        for filter_col in filter_cols
                    ],
                    style={"display": "flex", "width": "100%", "marginBottom": "15px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Plot By: ", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id="facet-dropdown",
                                    options=[
                                        {
                                            "label": f"{col_label.get(facet_col, facet_col)}",
                                            "value": facet_col,
                                        }
                                        for facet_col in facet_cols
                                    ],
                                    value="method",
                                    clearable=False,
                                ),
                            ],
                            style={"minWidth": "0", "flex": "1", "margin": "auto 2px"},
                        ),
                        # html.Div([
                        #     html.Label("X-Axis: ", style={"fontWeight": "bold"}),
                        #     dcc.Dropdown(id="xaxis-dropdown", options={f"{col_label.get(x_col, x_col)}": x_col for x_col in x_cols}, value=["band_idx"], clearable=False, debounce=True, closeOnSelect=False, multi=True)
                        # ], style={"minWidth": "0", "flex": "1"}),
                        html.Div(
                            [
                                html.Label(
                                    "Color Line: ", style={"fontWeight": "bold"}
                                ),
                                dcc.Dropdown(
                                    id="line-dropdown",
                                    options=[
                                        {
                                            "label": f"{col_label.get(line_col, line_col)}",
                                            "value": line_col,
                                        }
                                        for line_col in line_cols
                                    ],
                                    value="virtual_sampling_method",
                                    clearable=False,
                                ),
                            ],
                            style={"minWidth": "0", "flex": "1", "margin": "auto 2px"},
                        ),
                    ],
                    style={"width": "100%", "display": "flex"},
                ),
            ],
            style={"width": "95%", "margin": "10px auto"},
        ),
        html.Div(
            [dcc.Graph(id="main-chart")], style={"width": "95%", "margin": "`0px auto"}
        ),
        dcc.Store(id="selected-data", data=[]),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            f"{col_label.get(filter_col, filter_col)}: ",
                                            style={"fontWeight": "bold"},
                                        ),
                                        dcc.Dropdown(
                                            id={
                                                "type": "filter-rate-dropdown",
                                                "column": filter_col,
                                                "idx": sub_idx,
                                            },
                                            options=[
                                                {
                                                    "label": col_val_label.get(
                                                        filter_col, {}
                                                    ).get(val, val),
                                                    "value": val,
                                                }
                                                for val in col_vals.get(filter_col, [])
                                            ],
                                            value=col_vals.get(filter_col, []),
                                            clearable=False,
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
                                for filter_col in ["true_samples", "virtual_samples"]
                            ],
                            style={
                                "display": "flex",
                                "width": "90%",
                                "margin": "auto 15px",
                            },
                        ),
                        html.Div(
                            [dcc.Graph(id=f"cmp-chart-{sub_idx}")],
                            style={"width": "100%", "marginTop": "15px"},
                        ),
                    ],
                    style={"flex": "1", "minWidth": "0"},
                )
                for sub_idx in range(2)
            ],
            style={"width": "95%", "display": "flex"},
        ),
        # html.Div([
        #     dcc.Graph(id="frame-chart")
        # ], style={"width": "95%", "margin": "auto"})
    ]
)


@app.callback(
    Output("main-chart", "figure"),
    Input({"type": "filter-dropdown", "column": ALL}, "value"),
    Input("facet-dropdown", "value"),
    Input("line-dropdown", "value"),
)
def update_main_chart(filter_vals, sel_facet, sel_line):
    conditions = [
        (df[col_name].isin(filter_val))
        for col_name, filter_val in zip(filter_cols, filter_vals)
        if filter_val and "all" not in filter_val
    ]
    groupby = [sel_facet, sel_line]
    filted_df = (
        (df[np.logical_and.reduce(conditions)] if conditions else df)
        .melt(id_vars=groupby, value_vars=metric_cols, var_name=x_col, value_name=y_col)
        .groupby(groupby + [x_col])[y_col]
        .mean()
        .reset_index()
    )

    fig = px.line(
        filted_df,
        x=x_col,
        y=y_col,
        markers=True,
        color=sel_line,
        facet_col=sel_facet,
        facet_col_wrap=2,
        # title=f"xxxxyyyyy",
        labels={y_col: col_label.get(y_col, y_col), x_col: col_label.get(x_col, x_col)},
        custom_data=[sel_facet, sel_line],
        color_discrete_sequence=px.colors.qualitative.Alphabet,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(BLTI_K)),
        ticktext=list(col_val_label.get(x_col, {}).values()),
    )
    fig.update_traces(line={"width": 2.0}, marker={"size": 8.0})
    fig.for_each_annotation(
        lambda a: a.update(
            text=col_val_label.get(sel_facet, {}).get(a.text.split("=")[-1], a.text)
        )
    )
    fig.for_each_trace(
        lambda trace: trace.update(
            name=col_val_label.get(sel_line, {}).get(trace.name, trace.name)
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=650,
        hovermode="closest",
        legend_title_text=col_label.get(sel_line, sel_line),
    )
    return fig


@app.callback(
    Output("selected-data", "data"),
    Input("main-chart", "clickData"),
    State("selected-data", "data"),
)
def update_selected_data(click_data, sel_data):
    if click_data is None:
        return sel_data

    point = click_data["points"][0]
    data = point["customdata"] + [point["x"]]

    if len(sel_data) == 0:
        return [data]

    if len(sel_data) == 1:
        return sel_data + [data]

    if len(sel_data) == 2:
        return [data]


@app.callback(
    Output("cmp-chart-0", "figure"),
    Output("cmp-chart-1", "figure"),
    Input({"type": "filter-dropdown", "column": ALL}, "value"),
    Input("facet-dropdown", "value"),
    Input("line-dropdown", "value"),
    Input("selected-data", "data"),
    Input({"type": "filter-rate-dropdown", "column": ALL, "idx": ALL}, "value"),
)
def update_cmp_chart(filter_vals, sel_facet, sel_line, sel_data, filter_rates):
    if sel_data is None or len(sel_data) == 0:
        return go.Figure(), go.Figure()
    figs = []
    groupby = [sel_facet, sel_line]
    for idx, data in enumerate(sel_data):
        conditions = [
            (df[col_name].isin(filter_val))
            for col_name, filter_val in zip(filter_cols, filter_vals)
            if filter_val
            and "all" not in filter_val
            and col_name not in [sel_facet, sel_line]
        ]
        conditions.extend([(df[sel_facet] == data[0]), (df[sel_line]) == data[1]])
        if filter_rates[idx * 2 + 0] is not None:
            conditions.append(df["true_samples"].isin(filter_rates[idx * 2 + 0]))
        if filter_rates[idx * 2 + 1] is not None:
            conditions.append(df["virtual_samples"].isin(filter_rates[idx * 2 + 1]))

        cmp_df = (
            (df[np.logical_and.reduce(conditions)] if conditions else df)
            .melt(
                id_vars=groupby,
                value_vars=metric_cols,
                var_name=x_col,
                value_name=y_col,
            )
            .groupby(groupby + [x_col])[y_col]
            .agg(
                mean="mean",
                median="median",
                p10=lambda x: x.quantile(0.10),
                p90=lambda x: x.quantile(0.90),
            )
            .reset_index()
        )

        cmp_fig = go.Figure()

        x = cmp_df[x_col].tolist()
        p10 = cmp_df["p10"].tolist()
        p90 = cmp_df["p90"].tolist()

        cmp_fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=p90 + p10[::-1],
                fill="toself",
                fillcolor="rgba(100, 150, 255, 0.2)",
                line={"color": "rgba(0,0,0,0)"},
                name="10-90% envelope",
                hoverinfo="skip",
            )
        )

        cmp_fig.add_trace(
            go.Scatter(x=x, y=cmp_df["mean"], mode="lines+markers", name="Mean")
        )
        cmp_fig.add_trace(
            go.Scatter(x=x, y=cmp_df["median"], mode="lines+markers", name="Median")
        )
        cmp_fig.update_layout(
            title={
                "text": f"{col_val_label.get(sel_facet, {}).get(data[0], data[0])}-{col_val_label.get(sel_line, {}).get(data[1], data[1])}",
                "x": 0.5,
                "xanchor": "center",
            },
            template="plotly_white",
            xaxis_title=col_label.get(x_col, x_col),
            yaxis_title=col_label.get(y_col, y_col),
        )
        cmp_fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(BLTI_K)),
            ticktext=list(col_val_label.get(x_col, {}).values()),
        )
        figs.append(cmp_fig)

    if len(figs) == 1:
        figs.append(go.Figure())

    return figs[0], figs[1]


# @app.callback(
#     Output("frame-chart", "figure"),
#     Input("main-chart", "clickData")
# )
# def update_chart(click_data):
#     if click_data is None:
#         return go.Figure()

#     point = click_data["points"][0]
#     data = point["customdata"]

#     frame_df = (
#         df[
#             (df[facet_col] == data[0]) &
#             (df[line_col] == data[1]) &
#             (df[x_col] == point["x"])
#         ]
#         .groupby(
#             [facet_col, line_col, x_col, "frame_idx"]
#         )[y_col]
#         .agg(
#             mean="mean",
#             median="median",
#             p10=lambda x: x.quantile(0.10),
#             p90=lambda x: x.quantile(0.90)
#         )
#         .reset_index()
#     )
#     frame_fig = go.Figure()

#     fx = frame_df["frame_idx"].tolist()
#     fp10 = frame_df["p10"].tolist()
#     fp90 = frame_df["p90"].tolist()

#     frame_fig.add_trace(go.Scatter(
#         x=fx + fx[::-1],
#         y=fp90 + fp10[::-1],
#         fill="toself",
#         fillcolor="rgba(100, 150, 255, 0.2)",
#         line=dict(color="rgba(0,0,0,0)"),
#         name="P10-P90",
#         hoverinfo="skip"
#     ))

#     frame_fig.add_trace(go.Scatter(
#         x=fx,
#         y=frame_df["mean"],
#         mode="lines+markers",
#         name="Mean"
#     ))
#     frame_fig.add_trace(go.Scatter(
#         x=fx,
#         y=frame_df["median"],
#         mode="lines+markers",
#         name="Median"
#     ))
#     frame_fig.update_layout(
#         title={
#             "text": f"{data[0]}-{data[1]}-Band-{point["x"]}",
#             "x": 0.5,
#             "xanchor": "center"
#         },
#         template="plotly_white",
#         xaxis_title="Timeline (Frames)",
#         yaxis_title="BLTI"
#     )

#     return frame_fig

server = app.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 12345))
    app.run(debug=False, host='0.0.0.0', port=port)
