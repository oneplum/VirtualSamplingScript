#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any

import dash
import numpy as np
import polars as pl
from dash import ALL, Input, Output, State, ctx, dcc, html

from blti import BLTI_K, GAUSSIAN_SCALES
from common import COL_META, ColMeta, get_label
from data import (
    BAND_COL,
    DELTA_COL,
    METRIC_COLS,
    STATISTIC_MAX,
    STATISTIC_MEAN,
    STATISTIC_MEDIAN,
    STATISTIC_MIN,
    STATISTIC_PERCENTILE_10,
    STATISTIC_PERCENTILE_90,
    Y_COL,
    exclude_data,
    load_col_data,
    load_data,
    process_data,
    sort_data,
)
from plot import (
    TRACE_MODE_FILL,
    TRACE_MODE_LINE,
    TRACE_MODE_LINE_MARKER,
    TRACE_MODE_LOLLIPOP,
    TRACE_MODE_MARKER,
    Trace,
    all_fig,
)


def config_global_var():
    global SEQ_COLS
    global FRAME_COLS
    global ALL_VAR_COLS
    global FILTER_COLS
    global FACET_COLS
    global GROUP_COLS
    global X_COLS

    SEQ_COLS = [col for col in all_cols if col not in ["frame_idx", *METRIC_COLS]]
    FRAME_COLS = [col for col in all_cols if col not in METRIC_COLS]
    ALL_VAR_COLS = [*FRAME_COLS, BAND_COL]

    FILTER_COLS = FRAME_COLS
    FACET_COLS = ALL_VAR_COLS
    GROUP_COLS = ALL_VAR_COLS
    X_COLS = ALL_VAR_COLS

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
FACET_DEFAULT = ["lighting_enabled"]
GROUP_DEFAULT = ["virtual_sampling_method"]
X_COL_DEFAULT = ["method"]


COL_META[BAND_COL] = ColMeta(
    label="Band Center Scale (δ in pixels)",
    values={
        k: (f"{np.sqrt(GAUSSIAN_SCALES[k] * GAUSSIAN_SCALES[k + 1]):.2f}")
        for k in range(BLTI_K)
    },
)
COL_META[Y_COL] = ColMeta(label="BLTI")
COL_META[DELTA_COL] = ColMeta(label="ΔBLTI to Linear")

TRACE_MODES = [
    TRACE_MODE_LOLLIPOP,
    TRACE_MODE_LINE_MARKER,
    TRACE_MODE_MARKER,
    TRACE_MODE_FILL,
]

STATISTIC_MODES = [
    STATISTIC_MEAN,
    STATISTIC_MEDIAN,
    STATISTIC_MAX,
    STATISTIC_MIN,
    STATISTIC_PERCENTILE_10,
    STATISTIC_PERCENTILE_90,
]


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
                        value=[stats_modes[0]],
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
                },
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
                        html.Button("Download CSV", id="download-csv-btn"),
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
) -> pl.DataFrame:
    filter_by = {
        filter_ipt["id"]["column"]: filter_ipt["value"]
        for filter_ipt in filters
        if filter_ipt["value"]
    }

    cols = list(dict.fromkeys([*(facets or []), *(groups or []), *(x_cols or [])]))
    group_by = [col for col in cols if col != BAND_COL]

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
    if (not BAND_COL in cols) and (BAND_COL in dlf.collect_schema().names()):
        cols.append(BAND_COL)

    dlf = sort_data(dlf, cols)

    return dlf.collect()


def register_callback(
    app: dash.Dash, lf: pl.LazyFrame, cols: int = 2, debug: bool = False
) -> None:
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
    def update_main_chart(
        filter_vals, sel_facets, sel_groups, sel_cols, sel_metrics, sel_stats, sel_trace
    ):
        filters = ctx.inputs_list[0]

        df = get_data(
            lf,
            filters,
            sel_facets,
            sel_groups,
            sel_cols,
            sel_metrics,
            sel_stats,
            debug=debug,
        )

        if not sel_cols:
            sel_cols = [BAND_COL]

        if not sel_stats:
            sel_stats = METRIC_COLS

        traces = []
        if sel_trace == TRACE_MODE_FILL:
            traces.append(Trace((sel_stats[0], sel_stats[1]), mode=TRACE_MODE_FILL))
            for stat in sel_stats[2:]:
                traces.append(Trace(stat, mode=TRACE_MODE_LINE))
        else:
            for stat in sel_stats:
                traces.append(Trace(stat, mode=sel_trace))

        return all_fig(
            df,
            sel_cols,
            traces,
            sel_facets,
            sel_groups,
            max_cols=cols,
            row_height=500,
            shared_y=False,
        )

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

        df = get_data(
            lf,
            filters,
            sel_facets,
            sel_groups,
            sel_cols,
            sel_metrics,
            sel_stats,
            debug=debug,
        )

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

    if args.resolutions:
        lfs = []

        for resolution in args.resolutions:
            lf = load_data(args.data_dir / f"r{resolution}")

            if lf is not None:
                lfs.append(lf.with_columns(pl.lit(resolution).alias("resolution")))

        if not lfs:
            raise FileNotFoundError(f"No parquet or CSV files found in {args.data_dir}")

        lf = pl.concat(lfs)
    else:
        lf = load_data(args.data_dir)

    lf = exclude_data(lf)

    global all_cols
    all_cols = lf.collect_schema().names()
    config_global_var()

    filter_values = load_col_data(lf, FILTER_COLS)

    app = dash.Dash(__name__, title="BLTI Analysis")

    app.layout = make_layout(
        filter_values,
        FACET_COLS,
        GROUP_COLS,
        X_COLS,
        METRIC_COLS,
        STATISTIC_MODES,
        TRACE_MODES,
    )

    register_callback(app, lf, cols=args.cols, debug=args.debug)

    app.run(
        debug=args.debug,
        host="0.0.0.0",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
