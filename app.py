#!/usr/bin/env python3

import argparse
import math
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
from common import COL_META, ColMeta, get_label
from data import DELTA_COL, X_COL, Y_COL, SEQ_COLS, METRIC_COLS, load_col_data, load_data, sort_data, to_long, process_data

BAND_COL = "band_idx"
STATS_COL = "stats"

SEQ_COLS.append("resolution")
FRAME_COLS = [*SEQ_COLS, "frame_idx"]
ALL_VAR_COLS = [*FRAME_COLS, BAND_COL, STATS_COL]

FILTER_COLS = FRAME_COLS
FILTER_DEFAULT = {
    # "resolution": ["512"],
    # "dataset_name": ["Head1"],
    # "level": [0],
    # "lighting_enabled": [False],
    # "method": ["lin"],
    # "true_samples": [15],
    # "virtual_sampling_method": ["none"],
    # "virtual_samples": [0],
    # "trans_type": ["rot"],
    # "trans_axis": ["y"],
    # "frame_idx": [1]
}

FACET_COLS = ALL_VAR_COLS
FACET_DEFAULT = ["lighting_enabled"]

GROUP_COLS = ALL_VAR_COLS
GROUP_DEFAULT = ["virtual_sampling_method",]

X_COLS = ALL_VAR_COLS
X_COL_DEFAULT = ["method"]

METRIC_COL_DEFAULT = METRIC_COLS

COL_META[BAND_COL] = ColMeta(
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
    TRACE_MODE_LOLLIPOP,
    # TRACE_MODE_LINE,
    TRACE_MODE_LINE_MARKER,
    TRACE_MODE_FILL
]

STATISTIC_MEAN = "mean"
STATISTIC_MEDIAN = "median"
STATISTIC_MAX = "max"
STATISTIC_MIN = "min"
STATISTIC_PERCENTILE_10 = "p10"
STATISTIC_PERCENTILE_90 = "p90"
STATISTIC_MODES = [
    STATISTIC_MEAN, 
    STATISTIC_MEDIAN, 
    STATISTIC_MAX, 
    STATISTIC_MIN, 
    STATISTIC_PERCENTILE_10, 
    STATISTIC_PERCENTILE_90
]

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

    return ", ".join(map(str, labels))


def make_layout(
    filter_data: dict[str, list],
    facet_cols: list[str] | None = None,
    group_cols: list[str] | None = None,
    x_cols: list[str] | None = None,
    metric_cols: list[str] | None = None,
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
                        # value=X_COL_DEFAULT,
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

    if metric_cols:
        conf_divs.append(
            html.Div(
                [
                    html.Label(
                        "Metric: ",
                        style={"fontWeight": "bold"},
                    ),
                    dcc.Dropdown(
                        id="metric-dropdown",
                        options=[
                            {
                                "label": get_label(metric_col),
                                "value": metric_col,
                            }
                            for metric_col in metric_cols
                        ],
                        # value=METRIC_COL_DEFAULT,
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
                        value=stats_modes[0],
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
    x_cols: Any,
    metric_cols: Any,
    stats: Any,
    debug: bool = False,
) -> pl.LazyFrame:
    filter_by = {
        filter_ipt["id"]["column"]: filter_ipt["value"]
        for filter_ipt in filters
        if filter_ipt["value"]
    }

    cols = dict.fromkeys([*(facets or []), *(groups or []), *(x_cols or [])])
    for syn_col in [BAND_COL, STATS_COL]:
        if syn_col in cols:
            del cols[syn_col]
    group_by = list(cols)

    if not metric_cols:
        metric_cols = METRIC_COLS

    if stats:
        aggs = (stats, metric_cols)
        select_by = None
    else:
        aggs = None
        select_by = list(dict.fromkeys([*group_by, *metric_cols]))

    dlf = process_data(
        lf,
        filter_by,
        group_by,
        aggs,
        select_by=select_by,
        use_frame=("frame_idx" in group_by),
        debug=debug,
    )

    dlf = (
        dlf.unpivot(index=group_by, variable_name="metric", value_name=Y_COL)
        .with_columns(
            pl.col("metric").str.split("_").list.last().alias(BAND_COL),
            pl.col("metric").str.split("_").list.first().alias(STATS_COL)
        )
        .drop("metric")
    )

    output_cols = dict.fromkeys([*cols, STATS_COL, BAND_COL])
    dlf = sort_data(dlf, output_cols)

    return dlf


def all_fig(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    stats: list[str],
    facet_cols: list[str] | None = None,
    group_cols: list[str] | None = None,
    group_title: str = "",
    trace_mode: str = TRACE_MODE_LINE,
    max_cols: int = 2
) -> go.Figure:
    if df.is_empty():
        return go.Figure().update_layout(title="No data")

    sel_cols = list(dict.fromkeys([x_col, y_col, STATS_COL, *facet_cols, *group_cols]))

    df = df.select(sel_cols)

    facets = (
        df.partition_by(facet_cols, maintain_order=True, as_dict=True)
        if facet_cols
        else {(): df}
    )
    n_facets = len(facets)
    ncols = min(max_cols, n_facets)
    nrows = math.ceil(n_facets / ncols)

    is_shared_y = True

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        shared_yaxes=is_shared_y,
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

                c_data = data.filter(pl.col(STATS_COL) == stats[0]).sort(x_col).select([x_col, y_col])
                x = c_data[x_col].to_list()
                top = c_data[y_col].to_list()
                bottom = data.filter(pl.col(STATS_COL) == stats[1]).sort(x_col)[y_col].to_list()


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
                fig.add_trace(
                    go.Scatter(
                        x=data[x_col],
                        y=data[y_col],
                        mode=trace_mode,
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

        if not is_shared_y:
            y_min = rows.select(pl.col(y_col).min()).item()
            y_max = rows.select(pl.col(y_col).max()).item()
            padding = max((y_max - y_min) * 0.1, abs(y_max) * 0.01, 0.000001)
            fig.update_yaxes(range=[y_min - padding, y_max + padding], tickformat=".5f", row=row, col=col)

        if is_show_hline:
            if "method" in sel_cols and "virtual_sampling_method" in sel_cols:
                y_b = rows.filter((pl.col("method") == "lin") & (pl.col("virtual_sampling_method") == "none")).select(pl.col(y_col).min()).item()
            else:
                y_b = rows.select(pl.col(y_col).min()).item()
            
            fig.add_hline(
                y=y_b,
                line_width=2,
                line_color=SETTING_COLORS[0],
                row=row,
                col=col,
            )

    fig.update_xaxes(
        matches="x",
        tickmode="array",
        tickvals=list(x_map.values()),
        # ticktext=[get_label(x_col, val) for val in x_map],
        showticklabels=True,
        ticks="outside",
        ticklen=5,
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text=get_label(x_col),
    )

    fig.update_yaxes(
        matches="y" if is_shared_y else None,
        showticklabels=True,
        ticks="outside",
        ticklen=5,
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text=get_label(y_col)
    )

    legent_title = group_title

    if is_shared_y:
        y_min = df.select(pl.col(y_col).min()).item()
        y_max = df.select(pl.col(y_col).max()).item()
        padding = max((y_max - y_min) * 0.1, abs(y_max) * 0.01, 0.000001)
        fig.update_yaxes(range=[y_min - padding, y_max + padding], tickformat=".5f")

    fig.update_layout(
        height=500 * nrows,
        hovermode="x",
        template="plotly_white",
        legend_title=legent_title,
    )

    return fig


def register_callback(app: dash.Dash, lf: pl.LazyFrame, cols: int = 2, debug: bool = False) -> None:
    @app.callback(
        Output("main-chart", "figure"),
        Input(
            {"type": "filter-dropdown", "column": ALL},
            "value",
        ),
        Input("facet-dropdown", "value"),
        Input("group-dropdown", "value"),
        Input("xcol-dropdown", "value"),
        Input("metric-dropdown", "value"),
        Input("smode-dropdown", "value"),
        Input("tmode-dropdown", "value"),
    )
    def update_main_chart(filter_vals, sel_facets, sel_groups, sel_cols, sel_metrics, sel_stats, sel_trace):
        filters = ctx.inputs_list[0]

        dlf = get_data(
            lf,
            filters,
            sel_facets,
            sel_groups,
            sel_cols,
            sel_metrics,
            sel_stats,
            debug=debug
        )

        if sel_cols:
            if len(sel_cols) > 1:
                x_col = "-".join(sel_cols)
                dlf = dlf.with_columns(pl.concat_str([pl.col(s_col) for s_col in sel_cols], separator="-").alias(x_col)).drop(sel_cols)
            else:
                x_col = sel_cols[0]
        else:
            x_col = BAND_COL

        if not sel_stats:
            sel_stats = ["blti"]

        facet_cols = sel_facets
        group_cols = sel_groups
        group_title=" ".join([get_label(col) for col in sel_groups]) if sel_groups else ""

        df = dlf.collect()

        return all_fig(df, x_col, Y_COL, sel_stats, facet_cols, group_cols, group_title=group_title, trace_mode=sel_trace, max_cols=cols)

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
        State("metric-dropdown", "value"),
        State("smode-dropdown", "value"),
        prevent_initial_call=True,
    )
    def download_data(
        n_clicks, filter_vals, sel_facets, sel_groups, sel_cols, sel_metrics, sel_stats
    ):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        filters = ctx.states_list[0]
        
        dlf = get_data(
            lf,
            filters,
            sel_facets,
            sel_groups,
            sel_cols,
            sel_metrics,
            sel_stats,
            debug=debug
        )
        df = dlf.collect()

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
        "--resolutions",
        nargs="+",
        type=int,
        default=[512, 1024],
        help="Resolutions to process (e.g. --resolutions 512 1024)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash debug mode.",
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=2,
        help="Subplot col number",
    )

    return parser


def main() -> int:
    args = build_argument_parser().parse_args()

    print("Starting...")

    lfs = []

    for resolution in args.resolutions:
        lf = load_data(args.data_dir / f"r{resolution}")

        if lf is not None:
            lfs.append(
                lf.with_columns(pl.lit(resolution).alias("resolution"))
            )

    if not lfs:
        raise FileNotFoundError(f"No parquet or CSV files found in {args.data_dir}")

    lf = pl.concat(lfs)

    filter_values = load_col_data(lf, FILTER_COLS)

    app = dash.Dash(__name__, title="BLTI Analysis")

    app.layout = make_layout(
        filter_values, FACET_COLS, GROUP_COLS, X_COLS, METRIC_COLS, STATISTIC_MODES, TRACE_MODES
    )

    register_callback(app, lf, cols=args.cols, debug=args.debug)

    app.run(
        debug=args.debug,
        host="0.0.0.0",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
