import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from common import COL_META, get_label

PALETTE = pc.qualitative.Plotly
DASHES = ["solid", "dash", "dot", "dashdot"]

TRACE_MODE_FILL = "fill"
TRACE_MODE_LINE = "lines"
TRACE_MODE_LINE_MARKER = "lines+markers"
TRACE_MODE_MARKER = "markers"
TRACE_MODE_LOLLIPOP = "lollipop"


@dataclass(frozen=True)
class Trace:
    y: str | tuple[str, str]
    label: str | None = None
    mode: str = TRACE_MODE_LINE_MARKER


def _cols(t: Trace) -> tuple[str, ...]:
    return (t.y,) if isinstance(t.y, str) else tuple(t.y)


def _label_of(t: Trace) -> str:
    return t.label or (t.y if isinstance(t.y, str) else f"{t.y[0]}–{t.y[1]}")


def _find_idx(lst: list[Any], val: Any):
    return lst.index(val) if val in lst else -1


def _key(cols: list[str]) -> pl.Expr:
    if not cols:
        return pl.lit("")

    expressions = []

    for col in cols:
        expr = pl.col(col)

        if col in COL_META and COL_META[col].values:
            expr = expr.replace(COL_META[col].values, default=expr)

        expressions.append(expr.cast(pl.String))

    return pl.concat_str(expressions, separator=" | ")


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _stems(x: np.ndarray, y: np.ndarray):
    n = len(x)
    sx, sy = np.full(3 * n, np.nan), np.full(3 * n, np.nan)
    sx[0::3] = sx[1::3] = x
    sy[0::3], sy[1::3] = 0.0, y
    return sx, sy


def all_fig(
    df: pl.DataFrame,
    x_cols: list[str],
    traces: list[Trace],
    facet_cols: list[str] | None = None,
    group_cols: list[str] | None = None,
    max_cols: int = 2,
    shared_y: bool = True,
    gl: bool = False,  # Scattergl for very large point counts
    row_height: int = 450,
) -> go.Figure:
    if df.is_empty():
        return go.Figure().update_layout(title="No data")

    df = df.with_row_index("_id")

    facet_cols, group_cols = list(facet_cols or []), list(group_cols or [])
    ycols = list(dict.fromkeys(c for t in traces for c in _cols(t)))

    x_keys = (
        df.select(x_cols)
        .unique(maintain_order=True)
        .with_columns(
            _x=pl.int_range(pl.len(), dtype=pl.Int32),
            _xlabel=_key(x_cols),
        )
    )
    x_ticks = dict(zip(x_keys["_x"].to_list(), x_keys["_xlabel"].to_list()))

    df = (
        df.select(
            list(dict.fromkeys(["_id", *x_cols, *ycols, *facet_cols, *group_cols]))
        )
        .with_columns(
            _f=_key(facet_cols),
            _g=_key(group_cols),
        )
        .join(x_keys.select([*x_cols, "_x"]), on=x_cols, how="left")
        .sort("_id")
    )
    groups = df["_g"].unique(maintain_order=True).to_list()
    n_groups = len(groups)
    color_of = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
    dodge_w = 0.6 / n_groups

    facets = df.partition_by("_f", maintain_order=True, as_dict=True)
    ncols = min(max_cols, len(facets))
    nrows = math.ceil(len(facets) / ncols)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        shared_yaxes=shared_y,
        subplot_titles=[k[0] for k in facets],
    )
    Scatter = go.Scattergl if gl else go.Scatter
    seen: set = set()

    def legend(key) -> bool:
        if key in seen:
            return False
        seen.add(key)
        return True

    show_hline = False

    for fi, (fkey, fdf) in enumerate(facets.items()):
        r, c = fi // ncols + 1, fi % ncols + 1
        by_g = {
            k[0]: v
            for k, v in fdf.partition_by(
                "_g", maintain_order=True, as_dict=True
            ).items()
        }

        for gi, g in enumerate(groups):
            gdf = by_g.get(g)
            if gdf is None:
                continue
            color = color_of[g]
            x = gdf["_x"].to_numpy().astype(float)
            Y = {col: gdf[col].to_numpy().astype(float) for col in ycols}

            for ti, t in enumerate(traces):
                name = " | ".join(
                    p for p in (g, _label_of(t) if len(traces) > 1 else "") if p
                ) or _label_of(t)
                dash = DASHES[ti % len(DASHES)]
                add = lambda tr: fig.add_trace(tr, row=r, col=c)

                if t.mode == TRACE_MODE_FILL:
                    lo, hi = t.y
                    add(
                        Scatter(  # lower edge, invisible, anchor for tonexty
                            x=x,
                            y=Y[lo],
                            mode=TRACE_MODE_LINE,
                            line=dict(width=0),
                            legendgroup=name,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                    add(
                        Scatter(
                            x=x,
                            y=Y[hi],
                            mode=TRACE_MODE_LINE,
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor=_rgba(color, 0.2),
                            name=name,
                            legendgroup=name,
                            showlegend=legend(name),
                            hoverinfo="skip",
                        )
                    )
                elif t.mode == TRACE_MODE_LOLLIPOP:
                    show_hline = True
                    xp = x + (gi - (n_groups - 1) / 2) * dodge_w
                    sx, sy = _stems(xp, Y[t.y])
                    add(
                        Scatter(
                            x=sx,
                            y=sy,
                            mode=TRACE_MODE_LINE,
                            line=dict(width=2, color=color),
                            legendgroup=name,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                    add(
                        Scatter(
                            x=xp,
                            y=Y[t.y],
                            mode=TRACE_MODE_MARKER,
                            marker=dict(size=9, color=color),
                            name=name,
                            legendgroup=name,
                            showlegend=legend(name),
                            hovertemplate=f"{name}<br>%{{y:.4f}}<extra></extra>",
                        )
                    )
                else:
                    add(
                        Scatter(
                            x=x,
                            y=Y[t.y],
                            mode=t.mode,
                            name=name,
                            legendgroup=name,
                            line=dict(color=color, dash=dash),
                            marker=dict(color=color),
                            showlegend=legend(name),
                            hovertemplate=f"{name}<br>%{{y:.4f}}<extra></extra>",
                        )
                    )

        if show_hline:
            bres = fdf.get_column(ycols[0])
            bval = bres[0] if len(bres) > 0 else None
            if bval:
                fig.add_hline(
                    y=bval,
                    line_width=1,
                    line_color="black",
                    line_dash="dash",
                    row=r,
                    col=c,
                )

        if not shared_y:
            arr = fdf.select(ycols).to_numpy().astype(float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                lo, hi = float(arr.min()), float(arr.max())
                pad = max((hi - lo) * 0.1, abs(hi) * 0.01, 1e-6)
                fig.update_yaxes(range=[lo - pad, hi + pad], row=r, col=c)

    fig.update_xaxes(
        matches="x",
        tickmode="array",
        tickvals=list(x_ticks.keys()),
        ticktext=list(x_ticks.values()),
        # tickangle=-45,
        showticklabels=True,
        ticks="outside",
        ticklen=5,
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text=" | ".join([get_label(x_col) for x_col in x_cols]),
    )
    fig.update_yaxes(
        matches="y" if shared_y else None,
        showticklabels=True,
        ticks="outside",
        ticklen=5,
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text=get_label("blti"),
        tickformat=".4f",
    )

    if shared_y:
        arr = df.select(ycols).to_numpy().astype(float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            lo, hi = float(arr.min()), float(arr.max())
            pad = max((hi - lo) * 0.1, abs(hi) * 0.01, 1e-6)
            fig.update_yaxes(range=[lo - pad, hi + pad])

    fig.update_layout(
        height=row_height * nrows,
        hovermode="x",
        template="plotly_white",
        legend_title=" | ".join([get_label(group_col) for group_col in group_cols])
        if group_cols
        else None,
    )
    return fig
