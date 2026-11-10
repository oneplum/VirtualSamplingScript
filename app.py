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
from common import COL_META, ColMeta, FrameBasic, get_label
from data import DELTA_COL, X_COL, Y_COL, SEQ_COLS, METRIC_COLS, load_col_data, load_data, sort_data, to_long, process_data

ALL_VAR_COLS = [*SEQ_COLS, "frame_idx"]
FILTER_COLS = ["resolution", *SEQ_COLS]
FILTER_DEFAULT = {
    "resolution": ["512"],
    "dataset_name": ["Head1"],
    "level": [0],
    "lighting_enabled": [True],
    "method": ["lin"],
    "true_samples": [15],
    "virtual_sampling_method": ["none"],
    "virtual_samples": [0],
    "trans_type": ["rot"],
    "trans_axis": ["y"],
}

FACET_COLS = ALL_VAR_COLS
FACET_DEFAULT = ["lighting_enabled"]

GROUP_COLS = ALL_VAR_COLS
GROUP_DEFAULT = [
    "method",
    "virtual_sampling_method",
]

X_COLS = ALL_VAR_COLS
Y_COLS = METRIC_COLS
STATS_COL = "stats"

# BASELINES = {
#     "method": "lin",
#     "virtual_sampling_method": "none",
# }

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
TRACE_MODE_LINE = "lines"
TRACE_MODE_LINE_MARKER = "lines+markers"
TRACE_MODE_MARKER = "markers"
TRACE_MODE_LOLLIPOP = "lollipop"
TRACE_MODES = [
    TRACE_MODE_LINE,
    TRACE_MODE_LINE_MARKER,
    TRACE_MODE_LOLLIPOP,
    TRACE_MODE_FILL
]

STATISTIC_MEAN = "mean"
STATISTIC_MEDIAN = "median"
STATISTIC_MAX = "max"
STATISTIC_PERCENTILE_10 = "p10"
STATISTIC_PERCENTILE_90 = "p90"
STATISTIC_MODES = [STATISTIC_MEAN, STATISTIC_MEDIAN, STATISTIC_PERCENTILE_10, STATISTIC_PERCENTILE_90]


SETTING_COLORS = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24


def _find_idx(lst: list[Any], val: Any):
    return lst.index(val) if val in lst else -1


def _label(col_names: list[str], col_values: list[Any] | tuple[Any, ...]):
    labels = []

    vmethod_idx = _find_idx(col_names, "virtual_sampling_method")

    if len(col_names) == 1 and vmethod_idx == 0:
        return get_label("virtual_sampling_method", col_values[vmethod_idx])

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
                        f"{col_values[vs]} Virtual {get_label(col, val)} Samples"
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
            labels.append(get_label(col, val))
        else:
            labels.append(val)

    return ", ".join(labels)


def make_layout(
    filter_data: dict[str, list],
    facet_cols: list[str] | None = None,
    group_cols: list[str] | None = None,
    x_cols: list[str] | None = None,
    y_cols: list[str] | None = None,
    stats_modes: list[str] | None = None,
    trace_modes: list[str] | None = None,
) -> html.Div:
    filter_div = [
        html.H4("Filter By:"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            f"{get_label(filter_col)}: ",
                            style={"fontWeight": "bold"},
                        ),
                        dcc.Dropdown(
                            id={
                                "type": "filter-dropdown",
                                "column": filter_col,
                            },
                            options=[
                                {
                                    "label": get_label(filter_col, val),
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
    ]

    conf_divs = []
    if facet_cols:
        conf_divs.append(
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
                                "label": get_label(facet_col),
                                "value": facet_col,
                            }
                            for facet_col in facet_cols
                        ],
                        value=FACET_DEFAULT,
                        clearable=True,
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
            )
        )

    if group_cols:
        conf_divs.append(
            html.Div(
                [
                    html.Label(
                        "Group By: ",
                        style={"fontWeight": "bold"},
                    ),
                    dcc.Dropdown(
                        id="group-dropdown",
                        options=[
                            {
                                "label": get_label(group_col),
                                "value": group_col,
                            }
                            for group_col in group_cols
                        ],
                        value=GROUP_DEFAULT,
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
            )
        )

    if x_cols:
        conf_divs.append(
            html.Div(
                [
                    html.Label(
                        "Axis X: ",
                        style={"fontWeight": "bold"},
                    ),
                    dcc.Dropdown(
                        id="xcol-dropdown",
                        options=[
                            {
                                "label": get_label(x_col),
                                "value": x_col,
                            }
                            for x_col in x_cols
                        ],
                        clearable=True,
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
            )
        )

    if y_cols:
        conf_divs.append(
            html.Div(
                [
                    html.Label(
                        "Axis Y: ",
                        style={"fontWeight": "bold"},
                    ),
                    dcc.Dropdown(
                        id="ycol-dropdown",
                        options=[
                            {
                                "label": get_label(y_col),
                                "value": y_col,
                            }
                            for y_col in y_cols
                        ],
                        clearable=True,
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
            )
        )

    if stats_modes:
        conf_divs.append(
            html.Div(
                [
                    html.Label(
                        "Statistic Mode: ",
                        style={"fontWeight": "bold"},
                    ),
                    dcc.Dropdown(
                        id="smode-dropdown",
                        options=[
                            {
                                "label": get_label("stats_mode", s_mode),
                                "value": s_mode,
                            }
                            for s_mode in stats_modes
                        ],
                        # value=stats_modes[0],
                        clearable=True,
                        debounce=True,
                        closeOnSelect=False,
                        multi=True,
                    ),
                ],
                style={
                    "minWidth": "0",
                    "flex": "1",
                    "margin": "auto 2px",
                }
            )
        )


    if trace_modes:
        conf_divs.append(
            html.Div(
                [
                    html.Label(
                        "Trace Mode: ",
                        style={"fontWeight": "bold"},
                    ),
                    dcc.Dropdown(
                        id="tmode-dropdown",
                        options=[
                            {
                                "label": get_label("trace_mode", t_mode),
                                "value": t_mode,
                            }
                            for t_mode in trace_modes
                        ],
                        value=trace_modes[0],
                        clearable=False,
                    ),
                ],
                style={
                    "minWidth": "0",
                    "flex": "1",
                    "margin": "auto 2px",
                }
            )
        )

    conf_divs.append(
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
        )
    )

    filter_div.append(
        html.Div(
            conf_divs,
            style={
                "width": "100%",
                "display": "flex",
            },
        )
    )
            
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
                filter_div,
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
    x_cols: Any | None = None,
    y_cols: Any | None = None,
    stats: Any | None = None,
    debug: bool = False,
    is_long: bool = False,
) -> pl.LazyFrame:
    filter_by = {
        filter_ipt["id"]["column"]: filter_ipt["value"]
        for filter_ipt in filters
        if filter_ipt["value"]
    }

    cols = [*facets, *groups]
    if x_cols:
        cols.extend(x_cols)
    group_by = list(dict.fromkeys(
        col for col in cols if col in ALL_VAR_COLS
    ))

    if not y_cols:
        y_cols = METRIC_COLS
    if not stats:
        stats = [STATISTIC_MEAN]
    aggs = (stats, y_cols)        

    dlf = process_data(
        lf,
        filter_by,
        group_by,
        aggs,
        stats_col=STATS_COL,
        use_frame=(x_cols and "frame_idx" in x_cols),
        debug=debug,
    )
    group_by.append(STATS_COL)

    if is_long:
        dlf = (
            to_long(dlf, group_by, "metric")
            .with_columns(
                pl.col("metric").str.split("_").list.last().alias(X_COL),
            ).drop("metric")
        )
        group_by.append(X_COL)
        dlf = sort_data(dlf, group_by)

    return dlf


def all_fig(
    df: pl.DataFrame,
    x_col: str,
    y_cols: list[str],
    stats: list[str],
    facet_cols: list[str] | None = None,
    group_cols: list[str] | None = None,
    group_title: str = "",
    trace_mode: str = TRACE_MODE_LINE
) -> go.Figure:
    if df.is_empty():
        return go.Figure().update_layout(title="No data")

    sel_cols = [x_col, *y_cols, STATS_COL, *facet_cols, *group_cols]

    df = df.select(sel_cols)

    facets = (
        df.partition_by(facet_cols, maintain_order=True, as_dict=True)
        if facet_cols
        else {(): df}
    )
    n_facets = len(facets)
    ncols = min(2, n_facets)
    nrows = math.ceil(n_facets / ncols)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            _label(facet_cols, facet_val) for facet_val, _ in facets.items()
        ],
    )

    x_data = list(dict.fromkeys(df[x_col]))
    x_map = {v: k for k, v in enumerate(x_data)}

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

            color = SETTING_COLORS[group_idx % len(SETTING_COLORS)]

            if trace_mode == TRACE_MODE_LOLLIPOP:
                is_show_hline = True
                x_data = (
                    np.array([x_map[v] for v in data[x_col]], dtype=float)
                    + (group_idx - (n_groups - 1) / 2) * width
                )
                for y_col in y_cols:
                    diff = data[y_col].to_numpy()
                    stem_x = []
                    stem_y = []
                    for x, y in zip(x_data, diff):
                        stem_x.extend([x, x, None])
                        stem_y.extend([0, y, None])

                    fig.add_trace(
                        go.Scatter(
                            x=stem_x,
                            y=stem_y,
                            mode="lines",
                            line=dict(width=2, color=color),
                            showlegend=False,
                            hoverinfo="skip",
                            legendgroup=group_label,
                        ),
                        row=row,
                        col=col,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=x_data,
                            y=diff,
                            mode="markers",
                            name=group_label,
                            marker=dict(size=9, color=color),
                            legendgroup=group_label,
                            showlegend=(facet_idx == 0),
                            hovertemplate=(
                                f"{group_label}"
                                "<br>"
                                "BLTI: %{y:.4f}"
                                "<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=col,
                    )
            elif trace_mode == TRACE_MODE_FILL:
                data = data.sort(x_col)
                print(data)

                for y_col in y_cols:
                    c_data = data.filter(pl.col(STATS_COL) == stats[0]).sort(x_col).select([x_col, y_col])
                    x = c_data[x_col].to_list()
                    top = c_data[y_col].to_list()
                    bottom = data.filter(pl.col(STATS_COL) == stats[1]).sort(x_col)[y_col].to_list()
                    print(top)
                    print(bottom)


                    fig.add_trace(
                        go.Scatter(
                            x=x + x[::-1],
                            y=bottom + top[::-1],
                            fill="toself",
                            fillcolor="rgba(100, 150, 255, 0.2)",
                            line={"color": "rgba(0,0,0,0)"},
                            name=group_label,
                            legendgroup=group_label,
                            hoverinfo="skip",
                            showlegend=(facet_idx == 0),
                        ),
                        row=row,
                        col=col,
                    )
                    for stats_idx, stats_val in enumerate(stats[2:]):
                        color = SETTING_COLORS[(group_idx + stats_idx) % len(SETTING_COLORS)]
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=data.filter(pl.col(STATS_COL) == stats_val).sort(x_col)[y_col].to_list(),
                                mode="lines+markers",
                                name=group_label,
                                legendgroup=group_label,
                                line=dict(color=color),
                                showlegend=(facet_idx == 0),
                                hovertemplate=(
                                    f"{group_label}<br>BLTI: %{{y:.4f}}<extra></extra>"
                                ),
                            ),
                            row=row,
                            col=col,
                        )
            else:
                for y_col in y_cols:
                    fig.add_trace(
                        go.Scatter(
                            x=data[x_col],
                            y=data[y_col],
                            mode=trace_mode,
                            name=group_label,
                            legendgroup=group_label,
                            line=dict(color=color),
                            # line_dash=STATS_COL,
                            showlegend=(facet_idx == 0),
                            hovertemplate=(
                                f"{group_label}<br>BLTI: %{{y:.4f}}<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=col,
                    )

    fig.update_xaxes(
        matches="x",
        tickmode="array",
        tickvals=list(x_map.values()),
        ticktext=[get_label(x_col, val) for val in x_map.keys()],
        showticklabels=True,
        ticks="outside",
        ticklen=5,
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text=get_label(x_col),
    )

    fig.update_yaxes(
        matches="y",
        showticklabels=True,
        ticks="outside",
        ticklen=5,
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text=get_label(y_col)
    )

    legent_title = group_title

    fig.update_layout(
        # height=400 * nrows,
        hovermode="x",
        template="plotly_white",
        legend_title=legent_title,
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
        Input("group-dropdown", "value"),
        Input("xcol-dropdown", "value"),
        Input("ycol-dropdown", "value"),
        Input("smode-dropdown", "value"),
        Input("tmode-dropdown", "value"),
    )
    def update_main_chart(filter_vals, sel_facets, sel_groups, sel_cols, sel_ycols, sel_stats, sel_trace):
        filters = ctx.inputs_list[0]
        
        if not sel_cols:
            sel_cols = [X_COL]

        x_exist = any(X_COL in cols for cols in (sel_cols, sel_facets, sel_groups))

        dlf = get_data(
            lf, filters, sel_facets, sel_groups, sel_cols, sel_ycols, sel_stats, debug=debug, is_long=x_exist
        )

        if len(sel_cols) > 1:
            x_col = "-".join(sel_cols)
            dlf = dlf.with_columns(pl.concat_str([pl.col(s_col) for s_col in sel_cols], separator="-").alias(x_col)).drop(sel_cols)
        else:
            x_col = sel_cols[0]

        if x_exist or (not sel_ycols):
            sel_ycols = [Y_COL]

        if not sel_stats:
            sel_stats = [STATISTIC_MEAN]

        facet_cols = sel_facets
        group_cols = sel_groups
        group_title=" ".join([get_label(col) for col in sel_groups]) if sel_groups else ""

        df = dlf.collect()

        return all_fig(df, x_col, sel_ycols, sel_stats, facet_cols, group_cols, group_title=group_title, trace_mode=sel_trace)

    @app.callback(
        Output("download-csv", "data"),
        Input("download-csv-btn", "n_clicks"),
        State(
            {"type": "filter-dropdown", "column": ALL},
            "value",
        ),
        State("facet-dropdown", "value"),
        State("group-dropdown", "value"),
        State("xcol-dropdown", "value"),
        State("ycol-dropdown", "value"),
        State("smode-dropdown", "value"),
        prevent_initial_call=True,
    )
    def download_data(
        n_clicks, filter_vals, sel_facets, sel_groups, sel_cols, sel_ycols, sel_stats
    ):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        filters = ctx.states_list[0]
        
        lf = get_data(
            lf,
            filters,
            sel_facets,
            sel_groups,
            sel_cols,
            sel_ycols,
            sel_stats,
            debug=debug
        )
        df = lf.collect()

        return {
            "content": df.write_csv(),
            "filename": "data.csv",
            "type": "text/csv",
        }


def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Compute BLTI for image sequences.")

    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        required=True,
        help="Path to blti data",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash debug mode.",
    )

    return parser


def main() -> int:
    args = build_argument_parser().parse_args()

    print("Starting...")

    lf512 = load_data(args.data_dir / "r512").with_columns(pl.lit("512").alias("resolution"))
    lf1024 = load_data(args.data_dir / "r1024").with_columns(pl.lit("1024").alias("resolution"))

    lf = pl.concat([lf512, lf1024])

    if lf is None:
        raise FileNotFoundError(f"No parquet or CSV files found in {args.data_dir}")

    filter_values = load_col_data(lf, FILTER_COLS)

    app = dash.Dash(__name__, title="BLTI Analysis")

    app.layout = make_layout(
        filter_values, FACET_COLS, GROUP_COLS, X_COLS, Y_COLS, STATISTIC_MODES, TRACE_MODES
    )

    register_callback(app, lf, args.debug)

    app.run(
        debug=args.debug,
        host="0.0.0.0",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
