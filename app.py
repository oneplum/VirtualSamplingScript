#!/usr/bin/env python3

import argparse
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import dash
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import ALL, Input, Output, State, ctx, dcc, html
from plotly.subplots import make_subplots

from blti import BLTI_K, GAUSSIAN_SCALES
from common import COL_META, ColMeta, FrameBasic
from data import DELTA_COL, X_COL, Y_COL, load_col_data, load_data, process_data

FILTER_COLS = [col.name for col in fields(FrameBasic)]
FILTER_DEFAULT = {
    "dataset_name": ["Head1"],
    "level": [0],
    "lighting_enabled": [True],
    "true_samples": [15],
    "virtual_sampling_method": ["none"],
    "virtual_samples": [0],
    "trans_type": ["rot"],
    "trans_axis": ["y"],
}

FACET_COLS = ["dataset_name", "level", "lighting_enabled", "trans_type", "trans_axis"]
FACET_DEFAULT = ["dataset_name", "level"]

GROUP_COLS = [
    "method",
    "true_samples",
    "virtual_sampling_method",
    "virtual_samples",
]
BASELINES = {
    "method": "lin",
    "virtual_sampling_method": "none",
    "true_samples": 15,
    "virtual_samples": 0,
}


STATISTIC_MEAN = 0
STATISTIC_DELTA = 1
STATISTIC_PERCENTILE = 3
STATISTIC_MODES = [STATISTIC_MEAN, STATISTIC_DELTA, STATISTIC_PERCENTILE]
STATISTIC_LABEL = {
    STATISTIC_MEAN: "Mean",
    STATISTIC_DELTA: "Delta vs. Baseline",
    STATISTIC_PERCENTILE: "P10-P90",
}
STATISTIC_DEFAULT = STATISTIC_MEAN

COL_META[X_COL] = ColMeta(
    label="Band Center Scale (δ in pixels)",
    values={
        k: (f"{np.sqrt(GAUSSIAN_SCALES[k] * GAUSSIAN_SCALES[k + 1]):.2f}")
        for k in range(BLTI_K)
    },
)
COL_META[Y_COL] = ColMeta(label="BLTI")
COL_META[DELTA_COL] = ColMeta(label="ΔBLTI to Linear")

TRACE_MODE_FILL = "fill"
TRACE_MODE_LINE = "line"
TRACE_MODE_LINE_MARKER = "lines+markers"
TRACE_MODE_MARKER = "markers"
TRACE_MODE_LOLLIPOP = "lollipop"

SETTING_COLORS = px.colors.qualitative.Dark24


def _find_idx(lst: list[Any], val: Any):
    return lst.index(val) if val in lst else -1


def _label(col_names: list[str], col_values: list[Any] | tuple[Any, ...]):
    labels = []

    vmethod_idx = _find_idx(col_names, "virtual_sampling_method")

    for col, val in zip(col_names, col_values):
        if col == "level" and val == 0:
            continue

        if col == "virtual_sampling_method":
            if str(val).lower() == "none":
                continue
            else:
                vs = _find_idx(col_names, "virtual_samples")
                if vs >= 0:
                    labels.append(
                        f"{col_values[vs]} Virtual {COL_META[col].get_val_label(val)} Samples"
                    )
                    continue

        if (
            col == "true_samples"
            and vmethod_idx >= 0
            and str(col_values[vmethod_idx]).lower() != "none"
        ):
            continue

        if col == "virtual_samples" and (val == 0 or vmethod_idx >= 0):
            continue

        if col in COL_META:
            labels.append(COL_META[col].get_val_label(val))
        else:
            labels.append(val)

    return ", ".join(labels)


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


def make_layout(
    filter_data: dict[str, list], facet_cols: list[str], statistic_modes: list[int]
) -> html.Div:
    return html.Div(
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
                                        f"{COL_META[filter_col].label if filter_col in COL_META else filter_col}: ",
                                        style={"fontWeight": "bold"},
                                    ),
                                    dcc.Dropdown(
                                        id={
                                            "type": "filter-dropdown",
                                            "column": filter_col,
                                        },
                                        options=[
                                            {
                                                "label": COL_META[
                                                    filter_col
                                                ].get_val_label(val)
                                                if filter_col in COL_META
                                                else val,
                                                "value": val,
                                            }
                                            for val in filter_values
                                        ],
                                        value=FILTER_DEFAULT.get(filter_col, None),
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
                                                "label": COL_META[facet_col].label
                                                if facet_col in COL_META
                                                else facet_col,
                                                "value": facet_col,
                                            }
                                            for facet_col in facet_cols
                                        ],
                                        value=FACET_DEFAULT,
                                        clearable=False,
                                        debounce=True,
                                        closeOnSelect=False,
                                        multi=True,
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
                                        "Statistic: ",
                                        style={"fontWeight": "bold"},
                                    ),
                                    dcc.Dropdown(
                                        id="stats-dropdown",
                                        options=[
                                            {
                                                "label": STATISTIC_LABEL.get(
                                                    s_mode, s_mode
                                                ),
                                                "value": s_mode,
                                            }
                                            for s_mode in statistic_modes
                                        ],
                                        value=STATISTIC_DEFAULT,
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
                                    html.Label(
                                        "Action: ",
                                        style={"fontWeight": "bold"},
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Download CSV", id="download-csv-btn"
                                            ),
                                            dcc.Download(id="download-csv"),
                                        ],
                                        style={},
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
        ]
    )


def get_data(
    lf: pl.LazyFrame,
    filters: list[Any],
    facets: Any,
    groups: Any,
    stats: Any | None = None,
    baselines: Any | None = None,
    debug: bool = False,
) -> pl.DataFrame:
    filter_by = {
        filter_ipt["id"]["column"]: filter_ipt["value"]
        for filter_ipt in filters
        if filter_ipt["value"]
    }

    if stats == STATISTIC_PERCENTILE:
        use_frame = True
    else:
        use_frame = False

    if stats == STATISTIC_DELTA:
        for col, val in baselines.items():
            if col in filter_by and val not in filter_by[col]:
                filter_by[col].append(val)
    else:
        baselines = None

    dlf = process_data(
        lf,
        filter_by,
        facets,
        groups,
        baselines,
        use_frame=use_frame,
        debug=debug,
    )
    df = dlf.collect()
    return df


def all_fig(
    df: pl.DataFrame,
    facet_cols: list[str],
    group_cols: list[str],
    traces: list[tuple[str, str, str | None, str | None]],
    x_col: str = X_COL,
) -> go.Figure:
    if df.is_empty():
        return go.Figure().update_layout(title="No data")

    facets = df.partition_by(facet_cols, as_dict=True)
    n_facets = len(facets)
    ncols = min(2, n_facets)
    nrows = math.ceil(n_facets / ncols)

    y_col = traces[0][1]

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            _label(facet_cols, facet_val) for facet_val, _ in facets.items()
        ],
    )

    for facet_idx, (facet_val, rows) in enumerate(facets.items()):
        row = facet_idx // ncols + 1
        col = facet_idx % ncols + 1

        groups = (
            rows.partition_by(group_cols, maintain_order=True, as_dict=True)
            if group_cols
            else {(): rows}
        )
        n_groups = len(groups)
        width = 0.6 / max(1, n_groups)

        is_show_hline = False

        for group_idx, (group_value, data) in enumerate(groups.items()):
            if data.is_empty():
                continue

            group_label = _label(group_cols, group_value) if group_cols else ""

            for t_idx, (t_mode, t_col, ex_col, t_name) in enumerate(traces):
                trace_name = (
                    (t_name if t_name else t_col.upper())
                    if not group_label
                    else group_label
                )
                if t_mode == TRACE_MODE_LOLLIPOP:
                    is_show_hline = True
                    diff = data[t_col].to_numpy()
                    band_x = (
                        data[x_col].to_numpy()
                        + (group_idx - (n_groups - 1) / 2) * width
                    )

                    stem_x = []
                    stem_y = []
                    for x, y in zip(band_x, diff):
                        stem_x.extend([x, x, None])
                        stem_y.extend([0, y, None])

                    color = SETTING_COLORS[group_idx % len(SETTING_COLORS)]

                    fig.add_trace(
                        go.Scatter(
                            x=stem_x,
                            y=stem_y,
                            mode="lines",
                            line=dict(width=2, color=color),
                            showlegend=False,
                            hoverinfo="skip",
                            legendgroup=trace_name,
                        ),
                        row=row,
                        col=col,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=band_x,
                            y=diff,
                            mode="markers",
                            name=trace_name,
                            marker=dict(size=9, color=color),
                            legendgroup=trace_name,
                            showlegend=(facet_idx == 0),
                            hovertemplate=(
                                "Method: %{name}"
                                "<br>"
                                "Band: %{x}"
                                "<br>"
                                "ΔBLTI: %{y:.4f}"
                                "<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=col,
                    )
                elif t_mode == TRACE_MODE_FILL:
                    x = data[x_col].to_list()
                    pl = data[t_col].to_list()
                    ph = data[ex_col].to_list()

                    fig.add_trace(
                        go.Scatter(
                            x=x + x[::-1],
                            y=ph + pl[::-1],
                            fill="toself",
                            fillcolor="rgba(100, 150, 255, 0.2)",
                            line={"color": "rgba(0,0,0,0)"},
                            name=trace_name,
                            hoverinfo="skip",
                            showlegend=(facet_idx == 0),
                        ),
                        row=row,
                        col=col,
                    )
                else:
                    if group_cols:
                        color = SETTING_COLORS[group_idx % len(SETTING_COLORS)]
                    else:
                        color = SETTING_COLORS[t_idx % len(SETTING_COLORS)]
                    fig.add_trace(
                        go.Scatter(
                            x=data[x_col],
                            y=data[t_col],
                            mode=t_mode,
                            name=trace_name,
                            line=dict(color=color),
                            showlegend=(facet_idx == 0),
                            hovertemplate=(
                                "Band: %{x}<br>BLTI: %{y:.4f}<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=col,
                    )

        if is_show_hline:
            fig.add_hline(
                y=0,
                line_width=1,
                line_color="black",
                row=row,
                col=col,
            )

    fig.update_xaxes(
        tickmode="array",
        tickvals=data[x_col].to_list(),
        ticktext=list(COL_META[X_COL].values.values()),
        ticks="outside",
        ticklen=5,
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text=COL_META[x_col].label if x_col in COL_META else x_col.upper(),
    )

    fig.update_yaxes(
        matches="y",
        showticklabels=True,
        ticks="outside",
        ticklen=5,
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text=COL_META[y_col].label
        if y_col in COL_META
        else COL_META[Y_COL].label,
    )

    fig.update_layout(
        height=400 * nrows,
        hovermode="closest",
        template="plotly_white",
        legend_title="Sampling Method" if group_cols else "",
    )

    return fig


def register_callback(app: dash.Dash, lf: pl.LazyFrame, debug: bool = False) -> None:
    @app.callback(
        Output("main-chart", "figure"),
        Input(
            {"type": "filter-dropdown", "column": ALL},
            "value",
        ),
        Input("facet-dropdown", "value"),
        Input("stats-dropdown", "value"),
    )
    def update_main_chart(filter_vals, sel_facets, sel_stats):
        filters = ctx.inputs_list[0]

        df = get_data(lf, filters, sel_facets, GROUP_COLS, sel_stats, BASELINES, debug)

        facet_cols = sel_facets
        group_cols = GROUP_COLS
        if sel_stats == STATISTIC_DELTA:
            traces = [(TRACE_MODE_LOLLIPOP, DELTA_COL, None, None)]
        elif sel_stats == STATISTIC_PERCENTILE:
            facet_cols = facet_cols + group_cols
            group_cols = None
            traces = [
                (TRACE_MODE_FILL, "p10", "p90", "10-90% envelope"),
                (TRACE_MODE_LINE_MARKER, "median", None, None),
                (TRACE_MODE_LINE_MARKER, "mean", None, None),
            ]
        else:
            traces = [(TRACE_MODE_LINE_MARKER, Y_COL, None, None)]

        return all_fig(df, facet_cols, group_cols, traces)

    @app.callback(
        Output("download-csv", "data"),
        Input("download-csv-btn", "n_clicks"),
        State(
            {"type": "filter-dropdown", "column": ALL},
            "value",
        ),
        State("facet-dropdown", "value"),
        State("stats-dropdown", "value"),
        prevent_initial_call=True,
    )
    def download_data(n_clicks, filter_vals, sel_facets, sel_stats):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        filters = ctx.states_list[0]
        df = get_data(lf, filters, sel_facets, GROUP_COLS, sel_stats, BASELINES)
        return {
            "content": df.write_csv(),
            "filename": "data.csv",
            "type": "text/csv",
        }


def main() -> int:
    args = build_argument_parser().parse_args()

    print("Starting...")

    lf = load_data(args.data_dir)

    if lf is None:
        raise FileNotFoundError(f"No parquet or CSV files found in {args.data_dir}")

    filter_values = load_col_data(lf, FILTER_COLS)

    app = dash.Dash(__name__, title="BLTI Analysis")

    app.layout = make_layout(filter_values, FACET_COLS, STATISTIC_MODES)

    register_callback(app, lf, args.debug)

    app.run(
        debug=args.debug,
        host="0.0.0.0",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
