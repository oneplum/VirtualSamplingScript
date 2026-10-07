#!/usr/bin/env python3

import argparse
import math
from dataclasses import fields
from pathlib import Path

import dash
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import ALL, Input, Output, ctx, dcc, html
from plotly.subplots import make_subplots

from blti import BLTI_K, GAUSSIAN_SCALES
from common import COL_META, ColMeta, FrameBasic
from data import (
    SEQ_COLS,
    agg_data,
    diff_data,
    filter_data,
    load_col_data,
    load_data,
    to_long,
)

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

FILTER_COLS = [col.name for col in fields(FrameBasic)]
METRIC_COLS = [f"blti_{i}" for i in range(BLTI_K)]

Y_COL = "blti"
X_COL = "band_idx"
GROUP_COL = "sampling_method"
FACET_COLS = [
    "dataset_name",
    "level",
    "lighting_enabled",
    "trans_type",
    "trans_axis",
    "method",
]

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

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

    dynamic_height = (len(categroy_orders[sel_facet]) * 250) + 200

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


# @app.callback(
#     Output("cmp-chart", "figure"),
#     Input("main-chart", "clickData"),
#     Input(
#         {"type": "filter-dropdown", "column": ALL},
#         "value",
#     ),
#     Input("facet-dropdown", "value"),
# )
# def update_cmp_chart(click_data, filter_vals, sel_facet):
#     if click_data is None:
#         return go.Figure()

#     point = click_data["points"][0]

#     custom_data = point["customdata"]

#     method_filters, other_filters = filter_exprs(ctx.inputs_list[1])
#     groupby = [sel_facet]
#     sel_method = custom_data[0]

#     filters = other_filters + [(pl.col("sampling_method") == sel_method)]
#     baggs = ["mean", "median", "p10", "p90"]
#     blf = process_data(lf, filters, groupby, baggs)
#     bdf = blf.collect()

#     return line_chart(bdf, sel_facet, sel_method)


def lollipop_plot(df: pl.DataFrame, facet_cols: list, group_cols: list) -> go.Figure:
    if df.is_empty():
        return go.Figure().update_layout(
            title="No data"
        )

    facets = df.select(facet_cols).unique().sort(facet_cols)

    ncols = 2
    nrows = math.ceil(len(facets) / ncols)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=False,
        shared_yaxes=True,
        subplot_titles=[
            " - ".join(map(str, facet))
            for facet in facets.iter_rows()
        ],
    )

    band_labels = list(COL_META[X_COL].values.values())

    for facet_idx, facet_values in enumerate(facets.iter_rows()):
        row = facet_idx // ncols + 1
        col = facet_idx % ncols + 1
        
        rows = df.filter(
            pl.all_horizontal(
                [pl.col(col) == val for col, val in zip(facet_cols, facet_values)]
            )
        ).sort([*group_cols, X_COL])

        groups = rows.select(group_cols).unique().sort(group_cols)

        n_groups = len(groups)
        width = 0.6 / max(1, n_groups)

        for group_idx, group_values in enumerate(groups.iter_rows()):
            data = rows.filter(
                pl.all_horizontal(
                    [pl.col(col) == val for col, val in zip(group_cols, group_values)]
                )
            ).sort(X_COL)

            if data.is_empty():
                continue

            diff = data["diff"].to_numpy()

            band_x = data[X_COL].to_numpy()
            band_x = band_x + (group_idx - (n_groups - 1) / 2) * width
            name = " / ".join(map(str, group_values))

            stem_x = []
            stem_y = []
            for x, y in zip(band_x, diff):
                stem_x.extend([x, x, None])
                stem_y.extend([0, y, None])

            color = COLORS[group_idx % len(COLORS)]
            
            fig.add_trace(
                go.Scatter(
                    x=stem_x,
                    y=stem_y,
                    mode="lines",
                    line=dict(width=2, color=color),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=name
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter(
                    x=band_x,
                    y=diff,
                    mode="markers",
                    name=name,
                    marker=dict(size=9, color=color),
                    legendgroup=name,
                    showlegend=(
                        facet_idx == 0
                    ),
                    customdata=[
                        [name, band_labels[_x]]
                        for _x in data[X_COL]
                    ],
                    hovertemplate=(
                        "Method: %{customdata[0]}"
                        "<br>"
                        "Band: %{customdata[1]}"
                        "<br>"
                        "ΔBLTI: %{y:.4f}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        fig.add_hline(
            y=0,
            line_width=1,
            line_color="black",
            row=row,
            col=col,
        )

        fig.update_xaxes(
            tickmode="array",
            tickvals=band_x,
            ticktext=band_labels,
            row=row,
            col=col,
        )

    fig.update_layout(
        title="BLTI Difference to Linear",
        height=450 * nrows,
        hovermode="closest",
        template="plotly_white",
        legend_title="Method",
    )

    fig.update_xaxes(
        title="Band Center",
    )

    fig.update_yaxes(
        title="ΔBLTI to Linear",
    )

    return fig


def make_layout(filter_data: dict[str, list] | None = None, facet_cols: list[str] | None = None) -> html.Div:
    return html.Div([
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
                                        for val in filter_values
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
                        for filter_col, filter_values in filter_data.items()
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
                                        for facet_col in facet_cols
                                    ],
                                    value=["method"],
                                    clearable=False,
                                    debounce=True,
                                    closeOnSelect=False,
                                    multi=True
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
                                html.Label(
                                    "Action: ",
                                    style={"fontWeight": "bold"},
                                ),
                                html.Div(
                                    [
                                        html.Button("Download CSV", id="download-csv-btn"),

                                        dcc.Download(id="download-csv"),
                                    ],
                                    style={}
                                ),
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
        # html.Div(
        #     [
        #         html.Div(
        #             [dcc.Graph(id="baseline-chart")],
        #             style={
        #                 "flex": "1",
        #                 "minWidth": "0",
        #             },
        #         ),
        #         html.Div(
        #             [dcc.Graph(id="cmp-chart")],
        #             style={
        #                 "flex": "1",
        #                 "minWidth": "0",
        #             },
        #         ),
        #     ],
        #     style={"width": "95%", "display": "flex", "margin": "20px auto"},
        # ),
    ])


def register_callback(app: dash.Dash, lf: pl.LazyFrame) -> None:
    @app.callback(
        Output("main-chart", "figure"),
        # Output("baseline-chart", "figure"),
        Input(
            {"type": "filter-dropdown", "column": ALL},
            "value",
        ),
        Input("facet-dropdown", "value"),
    )
    def update_main_chart(filter_vals, sel_facet):
        filter_by = {filter_ipt["id"]["column"]: filter_ipt["value"] for filter_ipt in ctx.inputs_list[0] if filter_ipt["value"]}

        flf = filter_data(lf, filter_by)
        data_rows, data_seq = flf.select([pl.len(),pl.struct(SEQ_COLS).n_unique()]).collect().row(0)
        print(
            f"Filtered data: "
            f"{data_rows} rows, "
            f"{data_seq} sequences."
        )

        facet_cols = sel_facet
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
        df = diff_data(flf, facet_cols, group_cols, baselines)


        return lollipop_plot(df, facet_cols, group_cols)

    # @app.callback(
    #     Output("download-csv", "data"),
    #     Input("download-csv-btn", "n_clicks"),
    #     State(
    #         {"type": "filter-dropdown", "column": ALL},
    #         "value",
    #     ),
    #     State("facet-dropdown", "value"),
    #     prevent_initial_call=True,
    # )
    # def download_data(n_clicks, filter_vals, sel_facet):
    #     if not n_clicks:
    #         raise dash.exceptions.PreventUpdate

    #     method_filters, other_filters = filter_exprs(ctx.states_list[0])
    #     groupby = [sel_facet]
    #     filters = (
    #         method_filters + other_filters
    #     )
    #     flf = process_data(lf, filters, groupby, ["mean"])
    #     return {
    #         "content": flf.collect().write_csv(),
    #         "filename": "data.csv",
    #         "type": "text/csv",
    #     }


def main() -> int:
    args = build_argument_parser().parse_args()

    print("Starting...")

    lf = load_data(args.data_dir)

    if lf is None:
        raise FileNotFoundError(f"No parquet or CSV files found in {args.data_dir}")

    filter_values = load_col_data(lf, FILTER_COLS)

    app = dash.Dash(__name__, title="BLTI Analysis")

    app.layout = make_layout(filter_values, FACET_COLS)

    register_callback(app, lf)

    app.run(
        debug=args.debug,
        host="0.0.0.0",
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
