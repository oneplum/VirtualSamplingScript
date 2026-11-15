import math
from dataclasses import dataclass

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

PALETTE = pc.qualitative.Plotly  # hex colours
DASHES = ["solid", "dash", "dot", "dashdot"]


@dataclass(frozen=True)
class Trace:
    kind: str  # "line" | "lollipop" | "band"
    y: str | tuple[str, str]  # column, or (lo, hi) for a band
    label: str | None = None
    mode: str = "lines+markers"  # only used by "line"


def _cols(t: Trace) -> tuple[str, ...]:
    return (t.y,) if isinstance(t.y, str) else tuple(t.y)


def _label_of(t: Trace) -> str:
    return t.label or (t.y if isinstance(t.y, str) else f"{t.y[0]}–{t.y[1]}")


def _key(cols: list[str]) -> pl.Expr:
    if not cols:
        return pl.lit("")
    return pl.concat_str([pl.col(c).cast(pl.Utf8) for c in cols], separator=" | ")


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
    x_col: str,
    traces: list[Trace],
    facet_cols: list[str] | None = None,
    group_cols: list[str] | None = None,
    group_title: str = "",
    max_cols: int = 2,
    shared_y: bool = True,
    gl: bool = False,  # Scattergl for very large point counts
    row_height: int = 450,
) -> go.Figure:
    if df.is_empty():
        return go.Figure().update_layout(title="No data")

    facet_cols, group_cols = list(facet_cols or []), list(group_cols or [])
    ycols = list(dict.fromkeys(c for t in traces for c in _cols(t)))

    # ---- 1. prepare everything in Polars --------------------------------
    x_map = {v: i for i, v in enumerate(df[x_col].unique().sort())}
    df = (
        df.select(list(dict.fromkeys([x_col, *ycols, *facet_cols, *group_cols])))
        .with_columns(
            _f=_key(facet_cols),
            _g=_key(group_cols),
            _x=pl.col(x_col).replace_strict(x_map, return_dtype=pl.Int32),
        )
        .sort(["_g", "_x"], maintain_order=True)
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

    # ---- 2. draw --------------------------------------------------------
    for fi, (fkey, fdf) in enumerate(facets.items()):
        r, c = fi // ncols + 1, fi % ncols + 1
        by_g = {k[0]: v for k, v in fdf.partition_by("_g", as_dict=True).items()}

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

                if t.kind == "line":
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

                elif t.kind == "band":
                    lo, hi = t.y
                    add(
                        Scatter(  # lower edge, invisible, anchor for tonexty
                            x=x,
                            y=Y[lo],
                            mode="lines",
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
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor=_rgba(color, 0.2),
                            name=name,
                            legendgroup=name,
                            showlegend=legend(name),
                            hoverinfo="skip",
                        )
                    )

                elif t.kind == "lollipop":
                    xp = x + (gi - (n_groups - 1) / 2) * dodge_w
                    sx, sy = _stems(xp, Y[t.y])
                    add(
                        Scatter(
                            x=sx,
                            y=sy,
                            mode="lines",
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
                            mode="markers",
                            marker=dict(size=9, color=color),
                            name=name,
                            legendgroup=name,
                            showlegend=legend(name),
                            hovertemplate=f"{name}<br>%{{y:.4f}}<extra></extra>",
                        )
                    )
                else:
                    raise ValueError(f"unknown trace kind: {t.kind!r}")

    # ---- 3. axes / layout -----------------------------------------------
    fig.update_xaxes(
        matches="x",
        tickmode="array",
        tickvals=list(x_map.values()),
        ticktext=[str(v) for v in x_map],
        showticklabels=True,
        ticks="outside",
        ticklen=5,
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text=x_col,
    )
    fig.update_yaxes(
        matches="y" if shared_y else None,
        showticklabels=True,
        ticks="outside",
        ticklen=5,
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text=ycols[0] if len(ycols) == 1 else "value",
        tickformat=".5f",
    )

    arr = df.select(ycols).to_numpy().astype(float)
    arr = arr[np.isfinite(arr)]
    if arr.size:
        lo, hi = float(arr.min()), float(arr.max())
        if any(t.kind == "lollipop" for t in traces):
            lo, hi = min(lo, 0.0), max(hi, 0.0)
            fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="black")
        pad = max((hi - lo) * 0.1, abs(hi) * 0.01, 1e-6)
        if shared_y:
            fig.update_yaxes(range=[lo - pad, hi + pad])

    fig.update_layout(
        height=row_height * nrows,
        hovermode="x",
        template="plotly_white",
        legend_title=group_title,
    )
    return fig
