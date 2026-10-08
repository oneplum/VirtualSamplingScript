#!/usr/bin/env python3

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any

import dash
import polars as pl
from dash import ALL, Input, Output, State, ctx, dcc, html

from common import FrameBasic
from data import DELTA_COL, load_col_data, load_data, process_data
from plot import COL_META, line_fig, lollipop_fig

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

PLOT_ONE_SEQ = 0
PLOT_SEQS = 1
PLOT_DIFF = 2
PLOTS = [PLOT_ONE_SEQ, PLOT_SEQS, PLOT_DIFF]
PLOT_LABELS = {
    PLOT_ONE_SEQ: "Median und P10-90",
    PLOT_SEQS: "Default",
    PLOT_DIFF: "Delta BLTI",
}
PLOT_DEFAULT = PLOT_ONE_SEQ

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


def _label(col_names: list[str], col_values: list[Any] | tuple[Any, ...]):
    labels = []

    for col, val in zip(col_names, col_values):
        if col == "level" and val == 0:
            continue

        if col == "virtual_sampling_method" and val == "none":
            continue

        if col == "virtual_samples" and val == 0:
            continue

        if col in COL_META:
            labels.append(COL_META[col].get_val_label(val))
        else:
            labels.append(val)

    return " ".join(labels)


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


def make_layout(
    filter_data: dict[str, list], facet_cols: list[str], plots: list[int]
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
                                        "Plot: ",
                                        style={"fontWeight": "bold"},
                                    ),
                                    dcc.Dropdown(
                                        id="plot-dropdown",
                                        options=[
                                            {
                                                "label": PLOT_LABELS.get(
                                                    plot_idx, plot_idx
                                                ),
                                                "value": plot_idx,
                                            }
                                            for plot_idx in plots
                                        ],
                                        value=PLOT_DEFAULT,
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
        ]
    )


def get_data(
    lf: pl.LazyFrame,
    filters: list[Any],
    sel_facet: Any,
    groups: Any,
    sel_plot: Any | None = None,
    baselines: Any | None = None,
    debug: bool = False,
) -> pl.DataFrame:
    filter_by = {
        filter_ipt["id"]["column"]: filter_ipt["value"]
        for filter_ipt in filters
        if filter_ipt["value"]
    }

    if sel_plot == PLOT_ONE_SEQ:
        use_frame = True
    else:
        use_frame = False

    if sel_plot == PLOT_DIFF:
        for col, val in baselines.items():
            if col in filter_by and val not in filter_by[col]:
                filter_by[col].append(val)
    else:
        baselines = None

    dlf = process_data(
        lf,
        filter_by,
        sel_facet,
        groups,
        baselines,
        use_frame=use_frame,
        debug=debug,
    )
    df = dlf.collect()
    return df


def plot_data(
    df: pl.DataFrame,
    facet_cols: list[str],
    facet_col: str,
    group_cols: list[str] | None = None,
    group_col: str | None = None,
) -> pl.DataFrame:
    facets = df.select(facet_cols).unique(maintain_order=True)
    facets_label = {
        "-".join(map(str, _row)): _label(facet_cols, _row)
        for _row in facets.iter_rows()
    }
    exprs = [
        pl.concat_str(facet_cols, separator="-").replace(facets_label).alias(facet_col)
    ]
    drop_cols = facet_cols

    if group_cols:
        groups = df.select(group_cols).unique(maintain_order=True)
        groups_label = {
            "-".join(map(str, _row)): _label(group_cols, _row)
            for _row in groups.iter_rows()
        }
        exprs.append(
            pl.concat_str(group_cols, separator="-")
            .replace(groups_label)
            .alias(group_col)
        )
        drop_cols.extend(group_cols)

    df = df.with_columns(exprs).drop(drop_cols)

    return df


def register_callback(app: dash.Dash, lf: pl.LazyFrame, debug: bool = False) -> None:
    @app.callback(
        Output("main-chart", "figure"),
        # Output("baseline-chart", "figure"),
        Input(
            {"type": "filter-dropdown", "column": ALL},
            "value",
        ),
        Input("facet-dropdown", "value"),
        Input("plot-dropdown", "value"),
    )
    def update_main_chart(filter_vals, sel_facets, sel_plot):
        filters = ctx.inputs_list[0]

        df = get_data(lf, filters, sel_facets, GROUP_COLS, sel_plot, BASELINES, debug)

        facet_col = "_facet"
        group_col = "Method"

        if sel_plot == PLOT_DIFF:
            df = plot_data(df, sel_facets, facet_col, GROUP_COLS, group_col)
            return lollipop_fig(df, facet_col, group_col, y_col=DELTA_COL)
        elif sel_plot == PLOT_ONE_SEQ:
            df = plot_data(df, sel_facets + GROUP_COLS, facet_col)

            return line_fig(df, facet_col, y_col="mean")
        else:
            df = plot_data(df, sel_facets, facet_col, GROUP_COLS, group_col)
            return line_fig(df, facet_col, group_col)

    @app.callback(
        Output("download-csv", "data"),
        Input("download-csv-btn", "n_clicks"),
        State(
            {"type": "filter-dropdown", "column": ALL},
            "value",
        ),
        State("facet-dropdown", "value"),
        State("plot-dropdown", "value"),
        prevent_initial_call=True,
    )
    def download_data(n_clicks, filter_vals, sel_facets, sel_plot):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        filters = ctx.states_list[0]
        df = get_data(lf, filters, sel_facets, GROUP_COLS, sel_plot, BASELINES)
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

    app.layout = make_layout(filter_values, FACET_COLS, PLOTS)

    register_callback(app, lf, args.debug)

    app.run(
        debug=args.debug,
        host="0.0.0.0",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
