#!/usr/bin/env python3

import math

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from blti import BLTI_K, GAUSSIAN_SCALES
from common import COL_META, ColMeta
from data import DELTA_COL, X_COL, Y_COL

COL_META[X_COL] = ColMeta(
    label="Band Center Scale",
    values={
        k: (f"{np.sqrt(GAUSSIAN_SCALES[k] * GAUSSIAN_SCALES[k + 1]):.2f}")
        for k in range(BLTI_K)
    },
)
COL_META[Y_COL] = ColMeta(label="BLTI")
COL_META[DELTA_COL] = ColMeta(label="ΔBLTI to Linear")

SETTING_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def lollipop_fig(
    df: pl.DataFrame,
    facet_col: str,
    group_col: str,
    x_col: str = X_COL,
    y_col: str = Y_COL,
) -> go.Figure:
    if df.is_empty():
        return go.Figure().update_layout(title="No data")

    facets = df[facet_col].unique(maintain_order=True)
    ncols = 2
    nrows = math.ceil(len(facets) / ncols)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=False,
        shared_yaxes=True,
        subplot_titles=facets.to_list(),
    )

    for facet_idx, facet_value in enumerate(facets.to_list()):
        row = facet_idx // ncols + 1
        col = facet_idx % ncols + 1

        rows = df.filter(pl.col(facet_col) == facet_value)

        groups = rows[group_col].unique(maintain_order=True)

        n_groups = len(groups)
        width = 0.6 / max(1, n_groups)

        for group_idx, group_value in enumerate(groups.to_list()):
            data = rows.filter(pl.col(group_col) == group_value)

            if data.is_empty():
                continue

            diff = data[y_col].to_numpy()
            band_x = data[x_col].to_numpy() + (group_idx - (n_groups - 1) / 2) * width

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
                    legendgroup=group_value,
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter(
                    x=band_x,
                    y=diff,
                    mode="markers",
                    name=group_value,
                    marker=dict(size=9, color=color),
                    legendgroup=group_value,
                    showlegend=(facet_idx == 0),
                    # customdata=[
                    #     [name, band_labels[_x]]
                    #     for _x in data[X_COL]
                    # ],
                    # hovertemplate=(
                    #     "Method: %{name}"
                    #     "<br>"
                    #     "Band: %{x}"
                    #     "<br>"
                    #     "ΔBLTI: %{y:.4f}"
                    #     "<extra></extra>"
                    # ),
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
            tickvals=data[x_col].to_list(),
            ticktext=list(COL_META[X_COL].values.values()),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=450 * nrows,
        hovermode="closest",
        template="plotly_white",
        legend_title=group_col,
    )

    return fig


def line_fig(
    df: pl.DataFrame,
    facet_col: str,
    group_col: str | None = None,
    x_col: str = X_COL,
    y_col: str = Y_COL,
) -> go.Figure:
    if df.is_empty():
        return go.Figure().update_layout(title="No data")

    facets = df[facet_col].unique(maintain_order=True)
    n_facets = len(facets)
    if n_facets > 10:
        return go.Figure().update_layout(title="Too many facets")

    ncols = 2
    nrows = math.ceil(df[facet_col].n_unique() / ncols)

    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        markers=True,
        facet_col=facet_col,
        facet_col_wrap=ncols,
        # facet_row_spacing=0.05,
        color=group_col,
        labels={
            x_col: COL_META[x_col].label if x_col in COL_META else x_col.upper(),
            y_col: COL_META[y_col].label if y_col in COL_META else y_col.upper(),
        },
    )
    if y_col == "mean":
        fig.data[0].update(name="Mean", showlegend=True)

    for facet_idx, facet_val in enumerate(facets):
        if all(col in df.columns for col in ["p10", "p90"]):
            rows = df.filter(pl.col(facet_col) == facet_val)

            x = rows[x_col].to_list()
            p10 = rows["p10"].to_list()
            p90 = rows["p90"].to_list()

            row_idx = facet_idx // ncols + 1
            col_idx = facet_idx % ncols + 1

            fig.add_trace(
                go.Scatter(
                    x=x + x[::-1],
                    y=p90 + p10[::-1],
                    fill="toself",
                    fillcolor="rgba(100, 150, 255, 0.2)",
                    line={"color": "rgba(0,0,0,0)"},
                    name="10-90% envelope",
                    hoverinfo="skip",
                    showlegend=(facet_idx == 0),
                ),
                row=row_idx,
                col=col_idx,
            )

        if "median" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=rows["median"],
                    mode="lines+markers",
                    name="Median",
                    line=dict(color="orange"),
                    showlegend=(facet_idx == 0),
                ),
                row=row_idx,
                col=col_idx,
            )

    fig.update_layout(
        # title={
        #     "text": "xxx",
        #     "x": 0.5,
        #     "xanchor": "center",
        # },
        template="plotly_white",
        height=(nrows * 250) + 200,
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=df[x_col].unique(maintain_order=True).to_list(),
        ticktext=list(COL_META[X_COL].values.values()),
    )

    fig.for_each_annotation(
        lambda annotation: annotation.update(text=(annotation.text.split("=")[-1]))
    )

    # fig.data = (
    #     fig.data[-len(categroy_orders[sel_facet]) :]
    #     + fig.data[: -len(categroy_orders[sel_facet])]
    # )

    return fig


# def diff_plot(df: pl.DataFrame, facet_cols: list, group_cols: list) -> None:
#     if df.is_empty():
#         print("No valid data found.")
#         return

#     facets = df.select(facet_cols).unique().sort(facet_cols)

#     ncols = 2
#     nrows = math.ceil(len(facets) / ncols)

#     fig, axes = plt.subplots(
#         nrows=nrows,
#         ncols=ncols,
#         figsize=(8 * ncols, 6 * nrows),
#         sharex=True,
#         sharey=True,
#         squeeze=False,
#     )

#     colors = plt.cm.tab10(
#         np.linspace(0, 1, max(1, df.select(group_cols).unique().height))
#     )

#     for ax, facet_idx in zip(axes.flat, facets.iter_rows()):
#         rows = df.filter(
#             pl.all_horizontal(
#                 [pl.col(col) == val for col, val in zip(facet_cols, facet_idx)]
#             )
#         ).sort([*group_cols, X_COL])

#         groups = rows.select(group_cols).unique().sort(group_cols)

#         n_groups = len(groups)
#         width = 0.6 / max(1, n_groups)

#         for i, (group_idx, color) in enumerate(zip(groups.iter_rows(), colors)):
#             data = rows.filter(
#                 pl.all_horizontal(
#                     [pl.col(col) == val for col, val in zip(group_cols, group_idx)]
#                 )
#             ).sort(X_COL)

#             if data.is_empty():
#                 continue

#             x = data[X_COL].to_numpy()

#             x = x + (i - (n_groups - 1) / 2) * width

#             diff = data["diff"].to_numpy()

#             # lollipop stem
#             ax.vlines(
#                 x,
#                 0,
#                 diff,
#                 color=color,
#                 alpha=0.6,
#                 linewidth=1.5,
#             )

#             # lollipop head
#             ax.scatter(
#                 x,
#                 diff,
#                 color=color,
#                 s=60,
#                 label="-".join(map(str, group_idx)),
#                 zorder=3,
#             )

#         ax.axhline(
#             0,
#             color="black",
#             linewidth=1,
#         )

#         ax.set_xticks(range(BLTI_K))
#         ax.set_xticklabels(X_LABELS)

#         ax.set_title("-".join(map(str, facet_idx)))

#         ax.grid(
#             axis="y",
#             alpha=0.2,
#         )

#     for ax in axes.flat[len(facets) :]:
#         ax.remove()

#     for ax in axes[-1, :]:
#         ax.set_xlabel("Band Center")

#     for ax in axes[:, 0]:
#         ax.set_ylabel("ΔBLTI to Linear")

#     handles, labels = axes.flat[0].get_legend_handles_labels()
#     if handles:
#         fig.legend(
#             handles,
#             labels,
#             title="Method",
#             loc="lower right",
#             bbox_to_anchor=(0.98, 0.02),
#         )

#     plt.tight_layout()
#     plt.show()


# def seq_plot(df: pl.DataFrame):
#     groups = df[GROUP_COL].unique().sort()

#     ncols = 2
#     nrows = math.ceil(len(groups) / ncols)

#     fig, axes = plt.subplots(
#         nrows=nrows,
#         ncols=ncols,
#         figsize=(5 * ncols, 4 * nrows),
#         sharex=True,
#         sharey=True,
#         squeeze=False,
#     )

#     for ax, group_idx in zip(axes.flat, groups):
#         group = df.filter(pl.col(GROUP_COL) == group_idx).sort(X_COL)

#         x = group[X_COL].to_list()
#         mean = group["mean"].to_numpy()
#         median = group["median"].to_numpy()
#         p10 = group["p10"].to_numpy()
#         p90 = group["p90"].to_numpy()

#         ax.plot(
#             x,
#             mean,
#             marker="o",
#             label="Mean",
#         )

#         ax.plot(
#             x,
#             median,
#             # marker="o",
#             label="Median",
#         )

#         ax.fill_between(
#             range(len(x)),
#             p10,
#             p90,
#             alpha=0.2,
#             label="P10–P90",
#         )

#         ax.set_title(group_idx)
#         ax.grid(alpha=0.2)

#     for ax in axes.flat[len(groups) :]:
#         ax.remove()

#     fig.supxlabel("Band Center")
#     fig.supylabel("BLTI")

#     axes.flat[0].legend()

#     plt.tight_layout()
#     plt.show()
