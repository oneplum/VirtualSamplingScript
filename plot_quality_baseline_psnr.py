#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_LOCAL_CACHE_DIR = _SCRIPT_DIR / ".cache"
_LOCAL_MPLCONFIG_DIR = _SCRIPT_DIR / ".matplotlib"
_LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_LOCAL_MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_LOCAL_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


from common import load_quality_images, warn, QualityImage


METHOD_LABELS = {
    "lin": "linear",
    "quadB": "Quadratic B-Spline",
    "cubicB": "Cubic B-Spline",
    "quadBf": "Quadratic B-Spline Prefiltered",
    "cubicBf": "Cubic B-Spline Prefiltered",
    "cr": "Catmull-Rom Spline",
}

DATASET_LABELS = {
    "Head1": "Visible Human Head, Skin Transfer Function",
    "Head2": "Visible Human Head, Bones Transfer Function",
    "Tree1": "Christmas Tree, close-up",
    "Tree2": "Christmas Tree, wide shot",
    "Aneurism1": "Aneurism, Pulse Transfer-Function",
    "Aneurism2": "Aneurism, Smoothstep Transfer-Function",
    "Sphere": "Spherical Distance Function, multi Pulse Transfer Function",
    "ML1": "Marschner-Lobb, Pulse Transfer-Function",
    "ML2": "Marschner-Lobb, Smoothstep Transfer-Function",
}

METHOD_ORDER = ["lin", "quadB", "quadBf", "cubicB", "cubicBf", "cr"]
BASELINE_VARIANT_ORDER = ["baseline1", "baseline10"]
BASELINE_VARIANT_LABELS = {
    "baseline1": "Baseline 1",
    "baseline10": "Baseline 10",
}
BASELINE_VARIANT_MARKERS = {
    "baseline1": "o",
    "baseline10": "D",
}
BASELINE_VARIANT_COLORS = {
    "baseline1": "#1f77b4",
    "baseline10": "#d62728",
}


@dataclass
class PsNrRow:
    dataset_name: str
    dataset_label: str
    reference_kind: str
    lighting_enabled: bool
    method: str
    method_label: str
    level: int
    rotation: int
    baseline_variant: str
    baseline_variant_label: str
    reference_label: str
    reference_count: int
    target_label: str
    psnr_db: float


@dataclass
class AveragedPsNrRow:
    dataset_name: str
    dataset_label: str
    reference_kind: str
    lighting_enabled: bool
    method: str
    method_label: str
    level: int
    baseline_variant: str
    baseline_variant_label: str
    rotation_count: int
    mean_psnr_db: float
    stddev_psnr_db: float


@dataclass
class LevelPlotRow:
    level: int
    lighting_enabled: bool
    method: str
    method_label: str
    baseline_variant: str
    baseline_variant_label: str
    dataset_count: int
    mean_psnr_db: float
    stddev_psnr_db: float


def human_method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def human_dataset_label(dataset_name: str) -> str:
    return DATASET_LABELS.get(dataset_name, dataset_name)


def lighting_label(lighting_enabled: bool) -> str:
    return "Lighting On" if lighting_enabled else "Lighting Off"


def _format_level_list(levels: list[int]) -> str:
    if not levels:
        return "No levels"
    if len(levels) == 1:
        return f"Level {levels[0]}"
    return "Levels " + ", ".join(str(level) for level in levels)


def _average_scope_text(excluded_datasets: set[str]) -> str:
    if not excluded_datasets:
        return ""
    return f" | Excluding {len(excluded_datasets)} dataset(s)"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"


def _load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64)


def compute_psnr(reference_path: Path, target_path: Path) -> float:
    reference = _load_rgb_image(reference_path)
    target = _load_rgb_image(target_path)
    if reference.shape != target.shape:
        raise ValueError(
            f"image shapes differ: {reference.shape} vs {target.shape} "
            f"for {reference_path} and {target_path}"
        )
    mse = np.mean((reference - target) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def build_reference_index(
    images: list[QualityImage], *, discrete_baseline_method: str, discrete_reference_mode: str
) -> dict[tuple[str, bool, int], dict[str | None, QualityImage]]:
    reference_index: dict[tuple[str, bool, int], dict[str | None, QualityImage]] = {}
    analytic_datasets = {
        image.dataset_name for image in images if image.image_kind == "legacy_baseline_analytic"
    }

    for image in images:
        if image.dataset_name in analytic_datasets:
            if image.image_kind == "legacy_baseline_analytic":
                reference_index.setdefault(
                    (image.dataset_name, image.lighting_enabled, image.rotation), {}
                )[None] = image
        else:
            if (
                image.image_kind == "legacy_baseline_discrete"
                and (
                    discrete_reference_mode == "per_method"
                    or discrete_reference_mode == "average_all"
                    or image.method == discrete_baseline_method
                )
            ):
                reference_index.setdefault(
                    (image.dataset_name, image.lighting_enabled, image.rotation), {}
                )[image.method] = image
    return reference_index


def compute_baseline_psnr_rows(
    images: list[QualityImage],
    *,
    discrete_baseline_method: str,
    discrete_reference_mode: str,
    selected_baseline_variant: str,
    warning_stream,
) -> list[PsNrRow]:
    reference_index = build_reference_index(
        images,
        discrete_baseline_method=discrete_baseline_method,
        discrete_reference_mode=discrete_reference_mode,
    )
    rows: list[PsNrRow] = []
    analytic_datasets = {
        image.dataset_name for image in images if image.image_kind == "legacy_baseline_analytic"
    }

    for image in images:
        if image.image_kind != "baseline_reference":
            continue
        if image.baseline_variant not in BASELINE_VARIANT_LABELS:
            continue
        if image.baseline_variant != selected_baseline_variant:
            continue
        if image.level is None:
            continue

        available_references = reference_index.get(
            (image.dataset_name, image.lighting_enabled, image.rotation),
            {},
        )

        references: list[QualityImage]
        reference_description: str
        if image.dataset_name in analytic_datasets:
            reference = available_references.get(None)
            references = [reference] if reference is not None else []
            reference_description = "analytic"
        elif discrete_reference_mode == "per_method":
            reference = available_references.get(image.method)
            references = [reference] if reference is not None else []
            reference_description = image.method
        elif discrete_reference_mode == "average_all":
            references = [
                available_references[method]
                for method in METHOD_ORDER
                if method in available_references
            ]
            reference_description = "all_methods"
        else:
            reference = available_references.get(discrete_baseline_method)
            references = [reference] if reference is not None else []
            reference_description = discrete_baseline_method

        if not references:
            warn(
                (
                    f"missing baseline reference image for dataset={image.dataset_name}, "
                    f"lighting={image.lighting_enabled}, rotation={image.rotation}, "
                    f"reference_method={reference_description}"
                ),
                stream=warning_stream,
            )
            continue

        psnr_values: list[float] = []
        reference_labels: list[str] = []
        for reference in references:
            try:
                psnr_values.append(compute_psnr(reference.path, image.path))
            except ValueError as error:
                warn(str(error), stream=warning_stream)
                continue
            reference_labels.append(reference.label)

        if not psnr_values:
            continue

        psnr_db = statistics.fmean(psnr_values)
        if not math.isfinite(psnr_db):
            warn(
                (
                    f"psnr value error for dataset={image.dataset_name}, "
                    f"lighting={image.lighting_enabled}, rotation={image.rotation}, "
                    f"method: {image.method}, references: {", ".join(reference_labels)}"
                ),
                stream=warning_stream,
            )
            continue

        rows.append(
            PsNrRow(
                dataset_name=image.dataset_name,
                dataset_label=image.dataset_label,
                reference_kind=references[0].image_kind,
                lighting_enabled=image.lighting_enabled,
                method=image.method,
                method_label=image.method_label,
                level=image.level,
                rotation=image.rotation,
                baseline_variant=image.baseline_variant,
                baseline_variant_label=BASELINE_VARIANT_LABELS[image.baseline_variant],
                reference_label=", ".join(reference_labels),
                reference_count=len(reference_labels),
                target_label=image.label,
                psnr_db=psnr_db,
            )
        )

    rows.sort(
        key=lambda row: (
            row.dataset_name,
            row.lighting_enabled,
            METHOD_ORDER.index(row.method) if row.method in METHOD_ORDER else len(METHOD_ORDER),
            row.level,
            BASELINE_VARIANT_ORDER.index(row.baseline_variant),
            row.rotation,
        )
    )
    return rows


def average_psnr_rows(rows: list[PsNrRow]) -> list[AveragedPsNrRow]:
    grouped: dict[tuple[str, str, bool, str, int, str], list[PsNrRow]] = {}
    for row in rows:
        key = (
            row.dataset_name,
            row.reference_kind,
            row.lighting_enabled,
            row.method,
            row.level,
            row.baseline_variant,
        )
        grouped.setdefault(key, []).append(row)

    averaged_rows: list[AveragedPsNrRow] = []
    for grouped_rows in grouped.values():
        first = grouped_rows[0]
        psnr_values = [row.psnr_db for row in grouped_rows]
        averaged_rows.append(
            AveragedPsNrRow(
                dataset_name=first.dataset_name,
                dataset_label=first.dataset_label,
                reference_kind=first.reference_kind,
                lighting_enabled=first.lighting_enabled,
                method=first.method,
                method_label=first.method_label,
                level=first.level,
                baseline_variant=first.baseline_variant,
                baseline_variant_label=first.baseline_variant_label,
                rotation_count=len(grouped_rows),
                mean_psnr_db=statistics.fmean(psnr_values),
                stddev_psnr_db=statistics.pstdev(psnr_values) if len(psnr_values) > 1 else 0.0,
            )
        )

    averaged_rows.sort(
        key=lambda row: (
            row.dataset_name,
            row.lighting_enabled,
            METHOD_ORDER.index(row.method) if row.method in METHOD_ORDER else len(METHOD_ORDER),
            row.level,
            BASELINE_VARIANT_ORDER.index(row.baseline_variant),
        )
    )
    return averaged_rows


def _reference_text_for_mode(
    reference_kind: str,
    *,
    discrete_baseline_method: str,
    discrete_reference_mode: str,
) -> str:
    if reference_kind == "legacy_baseline_analytic":
        return "analytic baseline"
    if discrete_reference_mode == "per_method":
        return "per-method baseline"
    if discrete_reference_mode == "average_all":
        return "average over all baselines"
    return f"baseline: {human_method_label(discrete_baseline_method)}"


def _combined_reference_text(
    *, discrete_baseline_method: str, discrete_reference_mode: str
) -> str:
    analytic_text = "analytic baseline"
    if discrete_reference_mode == "per_method":
        discrete_text = "per-method baseline"
    elif discrete_reference_mode == "average_all":
        discrete_text = "average over all baselines"
    else:
        discrete_text = f"baseline: {human_method_label(discrete_baseline_method)}"
    return f"{analytic_text} / {discrete_text}"


def _write_csv_rows(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    row_dicts = [asdict(row) for row in rows]
    fieldnames = list(row_dicts[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_dicts)


def _write_reproduction_command(output_dir: Path) -> None:
    command_text = shlex.join(["python3", str(Path(sys.argv[0]).resolve()), *sys.argv[1:]])
    (output_dir / "reproduce_command.txt").write_text(
        command_text + "\n",
        encoding="utf-8",
    )


def filter_rows_by_levels[T](rows: list[T], *, levels: set[int]) -> list[T]:
    return [row for row in rows if getattr(row, "level", None) in levels]


def aggregate_rows_for_level_layout(rows: list[AveragedPsNrRow]) -> list[LevelPlotRow]:
    grouped: dict[tuple[int, bool, str, str], list[AveragedPsNrRow]] = {}
    for row in rows:
        key = (
            row.level,
            row.lighting_enabled,
            row.method,
            row.baseline_variant,
        )
        grouped.setdefault(key, []).append(row)

    level_rows: list[LevelPlotRow] = []
    for grouped_rows in grouped.values():
        first = grouped_rows[0]
        psnr_values = [row.mean_psnr_db for row in grouped_rows]
        level_rows.append(
            LevelPlotRow(
                level=first.level,
                lighting_enabled=first.lighting_enabled,
                method=first.method,
                method_label=first.method_label,
                baseline_variant=first.baseline_variant,
                baseline_variant_label=first.baseline_variant_label,
                dataset_count=len(grouped_rows),
                mean_psnr_db=statistics.fmean(psnr_values),
                stddev_psnr_db=statistics.pstdev(psnr_values) if len(psnr_values) > 1 else 0.0,
            )
        )

    level_rows.sort(
        key=lambda row: (
            row.level,
            row.lighting_enabled,
            METHOD_ORDER.index(row.method) if row.method in METHOD_ORDER else len(METHOD_ORDER),
        )
    )
    return level_rows


def plot_averaged_psnr(
    rows: list[AveragedPsNrRow],
    *,
    output_dir: Path,
    discrete_baseline_method: str,
    discrete_reference_mode: str,
    output_format: str,
    include_csv: bool,
    show_error_bars: bool,
) -> None:
    grouped: dict[tuple[str, str, bool], list[AveragedPsNrRow]] = {}
    for row in rows:
        grouped.setdefault((row.dataset_name, row.reference_kind, row.lighting_enabled), []).append(
            row
        )

    for (dataset_name, reference_kind, lighting_enabled), dataset_rows in grouped.items():
        fig, ax = plt.subplots(figsize=(12.0, 5.2), constrained_layout=True)

        categories = [
            (method, level)
            for method in METHOD_ORDER
            for level in sorted({row.level for row in dataset_rows if row.method == method})
        ]
        x_positions = {category: index for index, category in enumerate(categories)}

        for baseline_variant in BASELINE_VARIANT_ORDER:
            variant_rows = [row for row in dataset_rows if row.baseline_variant == baseline_variant]
            if not variant_rows:
                continue

            first_series = True
            for method in METHOD_ORDER:
                method_rows = sorted(
                    [row for row in variant_rows if row.method == method],
                    key=lambda row: row.level,
                )
                if not method_rows:
                    continue

                x_values = []
                y_values = []
                y_errors = []
                for row in method_rows:
                    x_values.append(
                        x_positions[(row.method, row.level)]
                        + (-0.12 if baseline_variant == "baseline1" else 0.12)
                    )
                    y_values.append(row.mean_psnr_db)
                    y_errors.append(row.stddev_psnr_db)

                plot_kwargs = {
                    "linestyle": "-",
                    "linewidth": 1.25,
                    "marker": BASELINE_VARIANT_MARKERS[baseline_variant],
                    "color": BASELINE_VARIANT_COLORS[baseline_variant],
                    "label": BASELINE_VARIANT_LABELS[baseline_variant] if first_series else None,
                }
                if show_error_bars:
                    ax.errorbar(
                        x_values,
                        y_values,
                        yerr=y_errors,
                        capsize=4,
                        **plot_kwargs,
                    )
                else:
                    ax.plot(x_values, y_values, **plot_kwargs)

                first_series = False

        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(
            [f"{human_method_label(method)}\nL{level}" for method, level in categories],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(
            f"{dataset_rows[0].dataset_label} | {lighting_label(lighting_enabled)} | "
            f"{_reference_text_for_mode(reference_kind, discrete_baseline_method=discrete_baseline_method, discrete_reference_mode=discrete_reference_mode)}"
        )
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)

        basename = (
            "quality_baseline_psnr_"
            f"{_slugify(dataset_name)}_"
            f"{'lit' if lighting_enabled else 'unlit'}_"
            f"ref_{_slugify('per_method' if discrete_reference_mode == 'per_method' and reference_kind != 'legacy_baseline_analytic' else 'average_all' if discrete_reference_mode == 'average_all' and reference_kind != 'legacy_baseline_analytic' else discrete_baseline_method)}"
        )
        if output_format in ("png", "both"):
            fig.savefig(output_dir / f"{basename}.png", dpi=220)
        if output_format in ("pdf", "both"):
            fig.savefig(output_dir / f"{basename}.pdf")
        if include_csv:
            _write_csv_rows(output_dir / f"{basename}.csv", dataset_rows)
        plt.close(fig)


def plot_level_summary_psnr(
    rows: list[LevelPlotRow],
    *,
    output_dir: Path,
    discrete_baseline_method: str,
    discrete_reference_mode: str,
    target_levels: list[int],
    excluded_datasets: set[str],
    output_format: str,
    include_csv: bool,
    show_error_bars: bool,
) -> None:
    grouped: dict[int, list[LevelPlotRow]] = {}
    for row in rows:
        grouped.setdefault(row.level, []).append(row)

    lighting_colors = {
        False: "#1f77b4",
        True: "#d62728",
    }
    lighting_markers = {
        False: "o",
        True: "D",
    }

    for level, level_rows in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
        x_values = list(range(len(METHOD_ORDER)))
        ax.set_xticks(x_values)
        ax.set_xticklabels(
            [human_method_label(method) for method in METHOD_ORDER],
            rotation=30,
            ha="right",
            fontsize=10,
        )

        for lighting_enabled in (False, True):
            series_rows = sorted(
                [row for row in level_rows if row.lighting_enabled == lighting_enabled],
                key=lambda row: METHOD_ORDER.index(row.method)
                if row.method in METHOD_ORDER
                else len(METHOD_ORDER),
            )
            if not series_rows:
                continue

            series_x = [METHOD_ORDER.index(row.method) for row in series_rows]
            series_y = [row.mean_psnr_db for row in series_rows]
            series_err = [row.stddev_psnr_db for row in series_rows]

            plot_kwargs = {
                "linestyle": "-",
                "linewidth": 1.5,
                "marker": lighting_markers[lighting_enabled],
                "color": lighting_colors[lighting_enabled],
                "label": lighting_label(lighting_enabled),
            }
            if show_error_bars:
                ax.errorbar(
                    series_x,
                    series_y,
                    yerr=series_err,
                    capsize=4,
                    **plot_kwargs,
                )
            else:
                ax.plot(series_x, series_y, **plot_kwargs)

        dataset_count = max((row.dataset_count for row in level_rows), default=0)
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(
            f"Level {level} | All datasets averaged ({dataset_count}) | "
            f"{_combined_reference_text(discrete_baseline_method=discrete_baseline_method, discrete_reference_mode=discrete_reference_mode)}"
            f"{_average_scope_text(excluded_datasets)}"
        )
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)

        basename = (
            "quality_baseline_psnr_level_"
            f"{level}_"
            f"ref_{_slugify('per_method' if discrete_reference_mode == 'per_method' else 'average_all' if discrete_reference_mode == 'average_all' else discrete_baseline_method)}"
        )
        if output_format in ("png", "both"):
            fig.savefig(output_dir / f"{basename}.png", dpi=220)
        if output_format in ("pdf", "both"):
            fig.savefig(output_dir / f"{basename}.pdf")
        if include_csv:
            _write_csv_rows(output_dir / f"{basename}.csv", level_rows)
        plt.close(fig)


def plot_dataset_level_psnr(
    rows: list[AveragedPsNrRow],
    *,
    output_dir: Path,
    discrete_baseline_method: str,
    discrete_reference_mode: str,
    output_format: str,
    include_csv: bool,
    show_error_bars: bool,
) -> None:
    grouped: dict[tuple[str, str, int], list[AveragedPsNrRow]] = {}
    for row in rows:
        grouped.setdefault((row.dataset_name, row.reference_kind, row.level), []).append(row)

    lighting_colors = {
        False: "#1f77b4",
        True: "#d62728",
    }
    lighting_markers = {
        False: "o",
        True: "D",
    }

    for (dataset_name, reference_kind, level), dataset_level_rows in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
        x_values = list(range(len(METHOD_ORDER)))
        ax.set_xticks(x_values)
        ax.set_xticklabels(
            [human_method_label(method) for method in METHOD_ORDER],
            rotation=30,
            ha="right",
            fontsize=10,
        )

        for lighting_enabled in (False, True):
            series_rows = sorted(
                [row for row in dataset_level_rows if row.lighting_enabled == lighting_enabled],
                key=lambda row: METHOD_ORDER.index(row.method)
                if row.method in METHOD_ORDER
                else len(METHOD_ORDER),
            )
            if not series_rows:
                continue

            series_x = [METHOD_ORDER.index(row.method) for row in series_rows]
            series_y = [row.mean_psnr_db for row in series_rows]
            series_err = [row.stddev_psnr_db for row in series_rows]

            plot_kwargs = {
                "linestyle": "-",
                "linewidth": 1.5,
                "marker": lighting_markers[lighting_enabled],
                "color": lighting_colors[lighting_enabled],
                "label": lighting_label(lighting_enabled),
            }
            if show_error_bars:
                ax.errorbar(
                    series_x,
                    series_y,
                    yerr=series_err,
                    capsize=4,
                    **plot_kwargs,
                )
            else:
                ax.plot(series_x, series_y, **plot_kwargs)

        ax.set_ylabel("PSNR (dB)")
        ax.set_title(
            f"{human_dataset_label(dataset_name)} | Level {level} | "
            f"{_reference_text_for_mode(reference_kind, discrete_baseline_method=discrete_baseline_method, discrete_reference_mode=discrete_reference_mode)}"
        )
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)

        basename = (
            "quality_baseline_psnr_"
            f"{_slugify(dataset_name)}_"
            f"level_{level}_"
            f"ref_{_slugify('per_method' if discrete_reference_mode == 'per_method' and reference_kind != 'legacy_baseline_analytic' else 'average_all' if discrete_reference_mode == 'average_all' and reference_kind != 'legacy_baseline_analytic' else discrete_baseline_method)}"
        )
        if output_format in ("png", "both"):
            fig.savefig(output_dir / f"{basename}.png", dpi=220)
        if output_format in ("pdf", "both"):
            fig.savefig(output_dir / f"{basename}.pdf")
        if include_csv:
            _write_csv_rows(output_dir / f"{basename}.csv", dataset_level_rows)
        plt.close(fig)


def plot_lighting_summary_psnr(
    rows: list[LevelPlotRow],
    *,
    output_dir: Path,
    discrete_baseline_method: str,
    discrete_reference_mode: str,
    target_levels: list[int],
    excluded_datasets: set[str],
    output_format: str,
    include_csv: bool,
    show_error_bars: bool,
) -> None:
    grouped: dict[bool, list[LevelPlotRow]] = {}
    for row in rows:
        if row.level in target_levels:
            grouped.setdefault(row.lighting_enabled, []).append(row)

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    markers = ["o", "D", "s", "^", "v", "P"]
    level_colors = {level: palette[index % len(palette)] for index, level in enumerate(target_levels)}
    level_markers = {level: markers[index % len(markers)] for index, level in enumerate(target_levels)}

    for lighting_enabled, lighting_rows in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
        x_values = list(range(len(METHOD_ORDER)))
        ax.set_xticks(x_values)
        ax.set_xticklabels(
            [human_method_label(method) for method in METHOD_ORDER],
            rotation=30,
            ha="right",
            fontsize=10,
        )

        for level in target_levels:
            series_rows = sorted(
                [
                    row
                    for row in lighting_rows
                    if row.level == level
                ],
                key=lambda row: METHOD_ORDER.index(row.method)
                if row.method in METHOD_ORDER
                else len(METHOD_ORDER),
            )
            if not series_rows:
                continue

            series_x = [METHOD_ORDER.index(row.method) for row in series_rows]
            series_y = [row.mean_psnr_db for row in series_rows]
            series_err = [row.stddev_psnr_db for row in series_rows]

            plot_kwargs = {
                "linestyle": "-",
                "linewidth": 1.5,
                "marker": level_markers[level],
                "color": level_colors[level],
                "label": f"Level {level}",
            }
            if show_error_bars:
                ax.errorbar(
                    series_x,
                    series_y,
                    yerr=series_err,
                    capsize=4,
                    **plot_kwargs,
                )
            else:
                ax.plot(series_x, series_y, **plot_kwargs)

        dataset_count = max((row.dataset_count for row in lighting_rows), default=0)
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(
            f"{lighting_label(lighting_enabled)} | {_format_level_list(target_levels)} | "
            f"All datasets averaged ({dataset_count}) | "
            f"{_combined_reference_text(discrete_baseline_method=discrete_baseline_method, discrete_reference_mode=discrete_reference_mode)}"
            f"{_average_scope_text(excluded_datasets)}"
        )
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)

        basename = (
            "quality_baseline_psnr_"
            f"{'lit' if lighting_enabled else 'unlit'}_"
            f"levels_{'_'.join(str(level) for level in target_levels)}_"
            f"ref_{_slugify('per_method' if discrete_reference_mode == 'per_method' else 'average_all' if discrete_reference_mode == 'average_all' else discrete_baseline_method)}"
        )
        if output_format in ("png", "both"):
            fig.savefig(output_dir / f"{basename}.png", dpi=220)
        if output_format in ("pdf", "both"):
            fig.savefig(output_dir / f"{basename}.pdf")
        if include_csv:
            _write_csv_rows(output_dir / f"{basename}.csv", lighting_rows)
        plt.close(fig)


def plot_dataset_lighting_psnr(
    rows: list[AveragedPsNrRow],
    *,
    output_dir: Path,
    discrete_baseline_method: str,
    discrete_reference_mode: str,
    target_levels: list[int],
    output_format: str,
    include_csv: bool,
    show_error_bars: bool,
) -> None:
    grouped: dict[tuple[str, str, bool], list[AveragedPsNrRow]] = {}
    for row in rows:
        if row.level in target_levels:
            grouped.setdefault((row.dataset_name, row.reference_kind, row.lighting_enabled), []).append(
                row
            )

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    markers = ["o", "D", "s", "^", "v", "P"]
    level_colors = {level: palette[index % len(palette)] for index, level in enumerate(target_levels)}
    level_markers = {level: markers[index % len(markers)] for index, level in enumerate(target_levels)}

    for (dataset_name, reference_kind, lighting_enabled), dataset_rows in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
        x_values = list(range(len(METHOD_ORDER)))
        ax.set_xticks(x_values)
        ax.set_xticklabels(
            [human_method_label(method) for method in METHOD_ORDER],
            rotation=30,
            ha="right",
            fontsize=10,
        )

        for level in target_levels:
            series_rows = sorted(
                [row for row in dataset_rows if row.level == level],
                key=lambda row: METHOD_ORDER.index(row.method)
                if row.method in METHOD_ORDER
                else len(METHOD_ORDER),
            )
            if not series_rows:
                continue

            series_x = [METHOD_ORDER.index(row.method) for row in series_rows]
            series_y = [row.mean_psnr_db for row in series_rows]
            series_err = [row.stddev_psnr_db for row in series_rows]

            plot_kwargs = {
                "linestyle": "-",
                "linewidth": 1.5,
                "marker": level_markers[level],
                "color": level_colors[level],
                "label": f"Level {level}",
            }
            if show_error_bars:
                ax.errorbar(
                    series_x,
                    series_y,
                    yerr=series_err,
                    capsize=4,
                    **plot_kwargs,
                )
            else:
                ax.plot(series_x, series_y, **plot_kwargs)

        ax.set_ylabel("PSNR (dB)")
        ax.set_title(
            f"{human_dataset_label(dataset_name)} | {lighting_label(lighting_enabled)} | {_format_level_list(target_levels)} | "
            f"{_reference_text_for_mode(reference_kind, discrete_baseline_method=discrete_baseline_method, discrete_reference_mode=discrete_reference_mode)}"
        )
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)

        basename = (
            "quality_baseline_psnr_"
            f"{_slugify(dataset_name)}_"
            f"{'lit' if lighting_enabled else 'unlit'}_"
            f"levels_{'_'.join(str(level) for level in target_levels)}_"
            f"ref_{_slugify('per_method' if discrete_reference_mode == 'per_method' and reference_kind != 'legacy_baseline_analytic' else 'average_all' if discrete_reference_mode == 'average_all' and reference_kind != 'legacy_baseline_analytic' else discrete_baseline_method)}"
        )
        if output_format in ("png", "both"):
            fig.savefig(output_dir / f"{basename}.png", dpi=220)
        if output_format in ("pdf", "both"):
            fig.savefig(output_dir / f"{basename}.pdf")
        if include_csv:
            _write_csv_rows(output_dir / f"{basename}.csv", dataset_rows)
        plt.close(fig)


def _clear_generated_plot_files(directory: Path) -> None:
    for path in directory.glob("quality_baseline_psnr_*.png"):
        path.unlink()
    for path in directory.glob("quality_baseline_psnr_*.pdf"):
        path.unlink()
    for path in directory.glob("quality_baseline_psnr_*.csv"):
        path.unlink()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute PSNR against dataset baselines for baseline1/baseline10 images."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("scripts/quality images"),
        help="Directory containing the generated screenshot images.",
    )
    parser.add_argument(
        "--discrete-baseline-method",
        choices=METHOD_ORDER,
        default="lin",
        help="Reference baseline method for sampled datasets. Ignored for analytic datasets.",
    )
    parser.add_argument(
        "--discrete-reference-mode",
        choices=("shared", "per_method", "average_all"),
        default="shared",
        help=(
            "How to choose the baseline for sampled datasets: one shared baseline method "
            "for all methods, each method's own baseline image, or the average over all baseline methods."
        ),
    )
    parser.add_argument(
        "--rotation-mode",
        choices=("averaged", "per_rotation"),
        default="averaged",
        help="Whether to average PSNR over rotations or keep per-rotation rows in the summary.",
    )
    parser.add_argument(
        "--baseline-variant",
        choices=BASELINE_VARIANT_ORDER,
        default="baseline10",
        help="Which baseline-sampling variant to plot and summarize.",
    )
    parser.add_argument(
        "--output-format",
        choices=("png", "pdf", "both"),
        default="both",
        help="Which plot file format to write.",
    )
    parser.add_argument(
        "--include-csv",
        action="store_true",
        help="Write a CSV file next to each generated plot with the plotted data.",
    )
    parser.add_argument(
        "--plot-layout",
        choices=("by_dataset", "by_level", "by_lighting"),
        default="by_dataset",
        help=(
            "How to organize the averaged PSNR plots."
        ),
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="+",
        choices=sorted(DATASET_LABELS),
        default=[],
        help="Datasets to exclude from the cross-dataset averages while still generating per-dataset breakdown plots.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Levels to include in the plots and summaries. Default: all levels, except by_lighting keeps its historical default of levels 1 and 2.",
    )
    parser.add_argument(
        "--include-error-bars",
        action="store_true",
        help="Draw standard-deviation error bars on the averaged PSNR plots.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("plots/plot_quality_baseline_psnr"),
        help="Directory for plots and summary JSON.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    images = load_quality_images(args.images_dir, warning_stream=sys.stderr)
    print(f"Found {len(images)} folders.")

    psnr_rows = compute_baseline_psnr_rows(
        images,
        discrete_baseline_method=args.discrete_baseline_method,
        discrete_reference_mode=args.discrete_reference_mode,
        selected_baseline_variant=args.baseline_variant,
        warning_stream=sys.stderr,
    )
    available_levels = sorted({row.level for row in psnr_rows})
    if args.levels is not None:
        target_levels = sorted(set(args.levels))
    elif args.plot_layout == "by_lighting":
        target_levels = [level for level in (1, 2) if level in available_levels]
    else:
        target_levels = available_levels

    if not target_levels:
        parser.error("No matching levels are available for the selected images and plot layout.")

    excluded_datasets = set(args.exclude_datasets)

    filtered_psnr_rows = filter_rows_by_levels(psnr_rows, levels=set(target_levels))
    averaged_rows = average_psnr_rows(filtered_psnr_rows)
    averaged_rows_for_summary = [
        row for row in averaged_rows if row.dataset_name not in excluded_datasets
    ]
    level_plot_rows = aggregate_rows_for_level_layout(averaged_rows_for_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _clear_generated_plot_files(args.output_dir)
    _write_reproduction_command(args.output_dir)
    if args.plot_layout == "by_dataset":
        plot_averaged_psnr(
            averaged_rows,
            output_dir=args.output_dir,
            discrete_baseline_method=args.discrete_baseline_method,
            discrete_reference_mode=args.discrete_reference_mode,
            output_format=args.output_format,
            include_csv=args.include_csv,
            show_error_bars=args.include_error_bars,
        )
    elif args.plot_layout == "by_level":
        plot_level_summary_psnr(
            level_plot_rows,
            output_dir=args.output_dir,
            discrete_baseline_method=args.discrete_baseline_method,
            discrete_reference_mode=args.discrete_reference_mode,
            target_levels=target_levels,
            excluded_datasets=excluded_datasets,
            output_format=args.output_format,
            include_csv=args.include_csv,
            show_error_bars=args.include_error_bars,
        )
        plot_dataset_level_psnr(
            averaged_rows,
            output_dir=args.output_dir,
            discrete_baseline_method=args.discrete_baseline_method,
            discrete_reference_mode=args.discrete_reference_mode,
            output_format=args.output_format,
            include_csv=args.include_csv,
            show_error_bars=args.include_error_bars,
        )
    else:
        plot_lighting_summary_psnr(
            level_plot_rows,
            output_dir=args.output_dir,
            discrete_baseline_method=args.discrete_baseline_method,
            discrete_reference_mode=args.discrete_reference_mode,
            target_levels=target_levels,
            excluded_datasets=excluded_datasets,
            output_format=args.output_format,
            include_csv=args.include_csv,
            show_error_bars=args.include_error_bars,
        )
        plot_dataset_lighting_psnr(
            averaged_rows,
            output_dir=args.output_dir,
            discrete_baseline_method=args.discrete_baseline_method,
            discrete_reference_mode=args.discrete_reference_mode,
            target_levels=target_levels,
            output_format=args.output_format,
            include_csv=args.include_csv,
            show_error_bars=args.include_error_bars,
        )

    summary_payload = {
        "images_dir": str(args.images_dir),
        "discrete_baseline_method": args.discrete_baseline_method,
        "discrete_reference_mode": args.discrete_reference_mode,
        "baseline_variant": args.baseline_variant,
        "exclude_datasets": sorted(excluded_datasets),
        "levels": target_levels,
        "output_format": args.output_format,
        "include_csv": args.include_csv,
        "plot_layout": args.plot_layout,
        "rotation_mode": args.rotation_mode,
        "psnr_rows": [asdict(row) for row in filtered_psnr_rows],
        "averaged_psnr_rows": [asdict(row) for row in averaged_rows],
        "averaged_psnr_rows_for_summary": [asdict(row) for row in averaged_rows_for_summary],
        "level_plot_rows": [asdict(row) for row in level_plot_rows],
    }
    (args.output_dir / "quality_baseline_psnr_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Wrote {len(averaged_rows)} averaged PSNR rows and plots to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
