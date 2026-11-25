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

VIRTUAL_METHOD_LABELS = {
    "linvs": "linear",
    "crvs": "Catmull-Rom Spline",
    "hermvs": "Hermite Spline",
    "monhermvs": "Monotone Hermite Spline",
}

METHOD_ORDER = ["lin", "quadB", "quadBf", "cubicB", "cubicBf", "cr"]
VIRTUAL_METHOD_ORDER = ["linvs", "crvs", "hermvs", "monhermvs"]
VIRTUAL_METHOD_COLORS = {
    "linvs": "#1f77b4",
    "crvs": "#d62728",
    "hermvs": "#2ca02c",
    "monhermvs": "#ff7f0e",
}
VIRTUAL_METHOD_MARKERS = {
    "linvs": "o",
    "crvs": "D",
    "hermvs": "s",
    "monhermvs": "^",
}


@dataclass
class VirtualPsNrRow:
    dataset_name: str
    reference_kind: str
    lighting_enabled: bool
    method: str
    level: int
    rotation: int
    virtual_sampling_method: str
    virtual_samples: int
    reference_label: str
    reference_count: int
    target_label: str
    psnr_db: float


@dataclass
class AveragedVirtualPsNrRow:
    method: str
    lighting_enabled: bool
    virtual_sampling_method: str
    virtual_samples: int
    observation_count: int
    dataset_count: int
    level_count: int
    mean_psnr_db: float
    stddev_psnr_db: float


@dataclass
class DatasetAveragedVirtualPsNrRow:
    dataset_name: str
    dataset_label: str
    method: str
    lighting_enabled: bool
    virtual_sampling_method: str
    virtual_samples: int
    observation_count: int
    level_count: int
    mean_psnr_db: float
    stddev_psnr_db: float


def human_method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def human_dataset_label(dataset_name: str) -> str:
    return DATASET_LABELS.get(dataset_name, dataset_name)


def human_virtual_method_label(method: str) -> str:
    return VIRTUAL_METHOD_LABELS.get(method, method)


def lighting_label(lighting_enabled: bool) -> str:
    return "Lighting On" if lighting_enabled else "Lighting Off"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"


def _load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def compute_psnr_arrays(reference: np.ndarray, target: np.ndarray) -> float:
    if reference.shape != target.shape:
        raise ValueError(
            f"image shapes differ: {reference.shape} vs {target.shape}"
        )
    mse = np.mean((reference - target) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def build_reference_array_cache(
    reference_index: dict[tuple[str, bool, int, int, str], QualityImage]
) -> dict[Path, np.ndarray]:
    cache: dict[Path, np.ndarray] = {}
    for reference in reference_index.values():
        if reference.path not in cache:
            cache[reference.path] = _load_rgb_image(reference.path)
    return cache


def build_reference_index(
    images: list[QualityImage],
    *,
    baseline_true_samples: int,
) -> dict[tuple[str, bool, int, int, str], QualityImage]:
    reference_index: dict[tuple[str, bool, int, int, str], QualityImage] = {}
    expected_baseline_variant = f"baseline{baseline_true_samples}"
    for image in images:
        if (
            image.image_kind == "baseline_reference"
            and image.baseline_variant == expected_baseline_variant
            and image.level is not None
        ):
            reference_index[
                (
                    image.dataset_name,
                    image.lighting_enabled,
                    image.rotation,
                    image.level,
                    image.method,
                )
            ] = image

    return reference_index


def compute_virtual_psnr_rows(
    images: list[QualityImage],
    *,
    baseline_true_samples: int,
    selected_methods: set[str] | None,
    selected_virtual_methods: set[str] | None,
    selected_levels: set[int] | None,
    max_virtual_samples: int | None,
    warning_stream,
) -> list[VirtualPsNrRow]:
    expected_baseline_variant = f"baseline{baseline_true_samples}"
    reference_index = build_reference_index(
        images,
        baseline_true_samples=baseline_true_samples,
    )
    reference_array_cache = build_reference_array_cache(reference_index)
    rows: list[VirtualPsNrRow] = []

    for image in images:
        if image.image_kind != "virtual_sampled":
            continue
        if image.level is None or image.virtual_sampling_method is None or image.virtual_samples is None:
            continue
        if max_virtual_samples is not None and image.virtual_samples > max_virtual_samples:
            continue
        if selected_methods is not None and image.method not in selected_methods:
            continue
        if (
            selected_virtual_methods is not None
            and image.virtual_sampling_method not in selected_virtual_methods
        ):
            continue
        if selected_levels is not None and image.level not in selected_levels:
            continue

        reference = reference_index.get(
            (
                image.dataset_name,
                image.lighting_enabled,
                image.rotation,
                image.level,
                image.method,
            )
        )
        if reference is None:
            warn(
                (
                    f"missing baseline reference image for dataset={image.dataset_name}, "
                    f"lighting={image.lighting_enabled}, rotation={image.rotation}, "
                    f"level={image.level}, reference_method={image.method}, "
                    f"reference_variant={expected_baseline_variant}"
                ),
                stream=warning_stream,
            )
            continue

        target_array = _load_rgb_image(image.path)
        try:
            psnr_db = compute_psnr_arrays(reference_array_cache[reference.path], target_array)
        except ValueError as error:
            warn(
                f"{error} for {reference.path} and {image.path}",
                stream=warning_stream,
            )
            continue

        rows.append(
            VirtualPsNrRow(
                dataset_name=image.dataset_name,
                reference_kind=reference.image_kind,
                lighting_enabled=image.lighting_enabled,
                method=image.method,
                level=image.level,
                rotation=image.rotation,
                virtual_sampling_method=image.virtual_sampling_method,
                virtual_samples=image.virtual_samples,
                reference_label=reference.label,
                reference_count=1,
                target_label=image.label,
                psnr_db=psnr_db,
            )
        )

    rows.sort(
        key=lambda row: (
            METHOD_ORDER.index(row.method) if row.method in METHOD_ORDER else len(METHOD_ORDER),
            row.lighting_enabled,
            VIRTUAL_METHOD_ORDER.index(row.virtual_sampling_method)
            if row.virtual_sampling_method in VIRTUAL_METHOD_ORDER
            else len(VIRTUAL_METHOD_ORDER),
            row.virtual_samples,
            row.dataset_name,
            row.level,
            row.rotation,
        )
    )
    return rows


def average_virtual_psnr_rows(
    rows: list[VirtualPsNrRow],
    *,
    excluded_datasets: set[str] | None = None,
) -> list[AveragedVirtualPsNrRow]:
    grouped: dict[tuple[str, bool, str, int], list[VirtualPsNrRow]] = {}
    for row in rows:
        if excluded_datasets is not None and row.dataset_name in excluded_datasets:
            continue
        key = (
            row.method,
            row.lighting_enabled,
            row.virtual_sampling_method,
            row.virtual_samples,
        )
        grouped.setdefault(key, []).append(row)

    averaged_rows: list[AveragedVirtualPsNrRow] = []
    for grouped_rows in grouped.values():
        first = grouped_rows[0]
        psnr_values = [row.psnr_db for row in grouped_rows]
        averaged_rows.append(
            AveragedVirtualPsNrRow(
                method=first.method,
                lighting_enabled=first.lighting_enabled,
                virtual_sampling_method=first.virtual_sampling_method,
                virtual_samples=first.virtual_samples,
                observation_count=len(grouped_rows),
                dataset_count=len({row.dataset_name for row in grouped_rows}),
                level_count=len({row.level for row in grouped_rows}),
                mean_psnr_db=statistics.fmean(psnr_values),
                stddev_psnr_db=statistics.pstdev(psnr_values) if len(psnr_values) > 1 else 0.0,
            )
        )

    averaged_rows.sort(
        key=lambda row: (
            METHOD_ORDER.index(row.method) if row.method in METHOD_ORDER else len(METHOD_ORDER),
            row.lighting_enabled,
            VIRTUAL_METHOD_ORDER.index(row.virtual_sampling_method)
            if row.virtual_sampling_method in VIRTUAL_METHOD_ORDER
            else len(VIRTUAL_METHOD_ORDER),
            row.virtual_samples,
        )
    )
    return averaged_rows


def average_virtual_psnr_rows_by_dataset(
    rows: list[VirtualPsNrRow],
) -> list[DatasetAveragedVirtualPsNrRow]:
    grouped: dict[tuple[str, str, bool, str, int], list[VirtualPsNrRow]] = {}
    for row in rows:
        key = (
            row.dataset_name,
            row.method,
            row.lighting_enabled,
            row.virtual_sampling_method,
            row.virtual_samples,
        )
        grouped.setdefault(key, []).append(row)

    averaged_rows: list[DatasetAveragedVirtualPsNrRow] = []
    for grouped_rows in grouped.values():
        first = grouped_rows[0]
        psnr_values = [row.psnr_db for row in grouped_rows]
        averaged_rows.append(
            DatasetAveragedVirtualPsNrRow(
                dataset_name=first.dataset_name,
                dataset_label=human_dataset_label(first.dataset_name),
                method=first.method,
                lighting_enabled=first.lighting_enabled,
                virtual_sampling_method=first.virtual_sampling_method,
                virtual_samples=first.virtual_samples,
                observation_count=len(grouped_rows),
                level_count=len({row.level for row in grouped_rows}),
                mean_psnr_db=statistics.fmean(psnr_values),
                stddev_psnr_db=statistics.pstdev(psnr_values) if len(psnr_values) > 1 else 0.0,
            )
        )

    averaged_rows.sort(
        key=lambda row: (
            row.dataset_name,
            METHOD_ORDER.index(row.method) if row.method in METHOD_ORDER else len(METHOD_ORDER),
            row.lighting_enabled,
            VIRTUAL_METHOD_ORDER.index(row.virtual_sampling_method)
            if row.virtual_sampling_method in VIRTUAL_METHOD_ORDER
            else len(VIRTUAL_METHOD_ORDER),
            row.virtual_samples,
        )
    )
    return averaged_rows


def _reference_text(baseline_true_samples: int) -> str:
    return f"baseline{baseline_true_samples} of same method and level"


def _level_text(
    selected_levels: set[int] | None,
    *,
    observed_level_count: int,
) -> str:
    if not selected_levels:
        return f"levels averaged ({observed_level_count})"
    ordered_levels = sorted(selected_levels)
    if len(ordered_levels) == 1:
        return f"Level {ordered_levels[0]}"
    return "Levels " + ", ".join(str(level) for level in ordered_levels)


def _average_dataset_text(dataset_count: int, excluded_datasets: set[str] | None) -> str:
    if not excluded_datasets:
        return f"All datasets averaged ({dataset_count})"
    return (
        f"Datasets averaged ({dataset_count}, excluding "
        + ", ".join(sorted(excluded_datasets))
        + ")"
    )


def _write_csv_rows(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def plot_virtual_psnr(
    rows: list[AveragedVirtualPsNrRow],
    *,
    output_dir: Path,
    baseline_true_samples: int,
    selected_levels: set[int] | None,
    excluded_datasets: set[str] | None,
    output_format: str,
    show_error_bars: bool,
) -> None:
    grouped: dict[tuple[str, bool], list[AveragedVirtualPsNrRow]] = {}
    for row in rows:
        grouped.setdefault((row.method, row.lighting_enabled), []).append(row)

    for (method, lighting_enabled), method_rows in grouped.items():
        fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)

        for virtual_method in VIRTUAL_METHOD_ORDER:
            series_rows = sorted(
                [row for row in method_rows if row.virtual_sampling_method == virtual_method],
                key=lambda row: row.virtual_samples,
            )
            if not series_rows:
                continue

            x_values = [row.virtual_samples for row in series_rows]
            y_values = [row.mean_psnr_db for row in series_rows]
            y_errors = [row.stddev_psnr_db for row in series_rows]
            plot_kwargs = {
                "linestyle": "-",
                "linewidth": 1.5,
                "marker": VIRTUAL_METHOD_MARKERS[virtual_method],
                "color": VIRTUAL_METHOD_COLORS[virtual_method],
                "label": human_virtual_method_label(virtual_method),
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

        dataset_count = max((row.dataset_count for row in method_rows), default=0)
        ax.set_xlabel("Virtual samples")
        ax.set_ylabel("PSNR (dB)")
        level_text = _level_text(
            selected_levels,
            observed_level_count=max((row.level_count for row in method_rows), default=0),
        )
        ax.set_title(
            f"{human_method_label(method)} | {lighting_label(lighting_enabled)} | "
            f"{_average_dataset_text(dataset_count, excluded_datasets)}, {level_text} | "
            f"{_reference_text(baseline_true_samples)}"
        )
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)

        basename = (
            "quality_virtual_psnr_"
            f"{_slugify(method)}_"
            f"{'lit' if lighting_enabled else 'unlit'}_"
            f"ref_baseline{baseline_true_samples}"
        )
        _write_csv_rows(output_dir / f"{basename}.csv", method_rows)
        if output_format in ("png", "both"):
            fig.savefig(output_dir / f"{basename}.png", dpi=220)
        if output_format in ("pdf", "both"):
            fig.savefig(output_dir / f"{basename}.pdf")
        plt.close(fig)


def plot_dataset_virtual_psnr(
    rows: list[DatasetAveragedVirtualPsNrRow],
    *,
    output_dir: Path,
    baseline_true_samples: int,
    selected_levels: set[int] | None,
    output_format: str,
    show_error_bars: bool,
) -> None:
    grouped: dict[tuple[str, str, bool], list[DatasetAveragedVirtualPsNrRow]] = {}
    for row in rows:
        grouped.setdefault((row.dataset_name, row.method, row.lighting_enabled), []).append(row)

    for (dataset_name, method, lighting_enabled), method_rows in grouped.items():
        fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)

        for virtual_method in VIRTUAL_METHOD_ORDER:
            series_rows = sorted(
                [row for row in method_rows if row.virtual_sampling_method == virtual_method],
                key=lambda row: row.virtual_samples,
            )
            if not series_rows:
                continue

            x_values = [row.virtual_samples for row in series_rows]
            y_values = [row.mean_psnr_db for row in series_rows]
            y_errors = [row.stddev_psnr_db for row in series_rows]
            plot_kwargs = {
                "linestyle": "-",
                "linewidth": 1.5,
                "marker": VIRTUAL_METHOD_MARKERS[virtual_method],
                "color": VIRTUAL_METHOD_COLORS[virtual_method],
                "label": human_virtual_method_label(virtual_method),
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

        ax.set_xlabel("Virtual samples")
        ax.set_ylabel("PSNR (dB)")
        level_text = _level_text(
            selected_levels,
            observed_level_count=max((row.level_count for row in method_rows), default=0),
        )
        ax.set_title(
            f"{human_dataset_label(dataset_name)} | {human_method_label(method)} | "
            f"{lighting_label(lighting_enabled)} | {level_text} | "
            f"{_reference_text(baseline_true_samples)}"
        )
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)

        basename = (
            "quality_virtual_psnr_"
            f"{_slugify(dataset_name)}_"
            f"{_slugify(method)}_"
            f"{'lit' if lighting_enabled else 'unlit'}_"
            f"ref_baseline{baseline_true_samples}"
        )
        _write_csv_rows(output_dir / f"{basename}.csv", method_rows)
        if output_format in ("png", "both"):
            fig.savefig(output_dir / f"{basename}.png", dpi=220)
        if output_format in ("pdf", "both"):
            fig.savefig(output_dir / f"{basename}.pdf")
        plt.close(fig)


def _clear_generated_plot_files(directory: Path) -> None:
    for path in directory.glob("quality_virtual_psnr_*.png"):
        path.unlink()
    for path in directory.glob("quality_virtual_psnr_*.pdf"):
        path.unlink()
    for path in directory.glob("quality_virtual_psnr_*.csv"):
        path.unlink()


def _write_reproduction_command(output_dir: Path) -> None:
    command_text = shlex.join(["python3", str(Path(sys.argv[0]).resolve()), *sys.argv[1:]])
    (output_dir / "reproduce_command.txt").write_text(
        command_text + "\n",
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute PSNR for virtual-sampled images against the selected baseline."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing the generated screenshot images.",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=10,
        help="Use baselineN of the same method and level as reference, for example 10 selects baseline10.",
    )
    parser.add_argument(
        "--method",
        choices=METHOD_ORDER,
        default=None,
        help="Restrict the analysis to one true-sampling method.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_ORDER,
        default=None,
        help="Restrict the analysis to a selected set of true-sampling methods.",
    )
    parser.add_argument(
        "--virtual-methods",
        nargs="+",
        choices=VIRTUAL_METHOD_ORDER,
        default=None,
        help="Restrict the analysis to a selected set of virtual-sampling methods.",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Restrict the analysis to one level instead of averaging across levels.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Restrict the analysis to a selected set of levels.",
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="+",
        choices=sorted(DATASET_LABELS.keys()),
        default=None,
        help="Exclude these datasets from the overall averages while still generating per-dataset plots.",
    )
    parser.add_argument(
        "--max-virtual-samples",
        type=int,
        default=None,
        help="Only include virtual-sampled images up to this virtual-sample count.",
    )
    parser.add_argument(
        "--include-error-bars",
        action="store_true",
        help="Draw standard-deviation error bars on the averaged PSNR plots.",
    )
    parser.add_argument(
        "--output-format",
        choices=("png", "pdf", "both"),
        default="both",
        help="Which plot file format to write.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("plots/plot_quality_virtual_samples_psnr"),
        help="Directory for plots and summary JSON.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    selected_methods: set[str] | None = None
    if args.methods is not None:
        selected_methods = set(args.methods)
    elif args.method is not None:
        selected_methods = {args.method}

    selected_virtual_methods: set[str] | None = None
    if args.virtual_methods is not None:
        selected_virtual_methods = set(args.virtual_methods)

    selected_levels: set[int] | None = None
    if args.levels is not None:
        selected_levels = set(args.levels)
    elif args.level is not None:
        selected_levels = {args.level}

    excluded_datasets: set[str] | None = None
    if args.exclude_datasets is not None:
        excluded_datasets = set(args.exclude_datasets)

    images = load_quality_images(args.images_dir, warning_stream=sys.stderr)
    print(f"Found {len(images)} folders.")

    psnr_rows = compute_virtual_psnr_rows(
        images,
        baseline_true_samples=args.baseline_samples,
        selected_methods=selected_methods,
        selected_virtual_methods=selected_virtual_methods,
        selected_levels=selected_levels,
        max_virtual_samples=args.max_virtual_samples,
        warning_stream=sys.stderr,
    )
    averaged_rows = average_virtual_psnr_rows(
        psnr_rows,
        excluded_datasets=excluded_datasets,
    )
    dataset_averaged_rows = average_virtual_psnr_rows_by_dataset(psnr_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _clear_generated_plot_files(args.output_dir)
    _write_reproduction_command(args.output_dir)
    plot_virtual_psnr(
        averaged_rows,
        output_dir=args.output_dir,
        baseline_true_samples=args.baseline_samples,
        selected_levels=selected_levels,
        excluded_datasets=excluded_datasets,
        output_format=args.output_format,
        show_error_bars=args.include_error_bars,
    )
    plot_dataset_virtual_psnr(
        dataset_averaged_rows,
        output_dir=args.output_dir,
        baseline_true_samples=args.baseline_samples,
        selected_levels=selected_levels,
        output_format=args.output_format,
        show_error_bars=args.include_error_bars,
    )

    summary_payload = {
        "images_dir": str(args.images_dir),
        "reference_mode": "baselineN_same_method_same_level",
        "baseline_samples": args.baseline_samples,
        "method": args.method,
        "methods": sorted(selected_methods) if selected_methods is not None else None,
        "virtual_methods": (
            sorted(selected_virtual_methods) if selected_virtual_methods is not None else None
        ),
        "level": args.level,
        "levels": sorted(selected_levels) if selected_levels is not None else None,
        "excluded_datasets": (
            sorted(excluded_datasets) if excluded_datasets is not None else None
        ),
        "max_virtual_samples": args.max_virtual_samples,
        "output_format": args.output_format,
        "psnr_rows": [asdict(row) for row in psnr_rows],
        "averaged_psnr_rows": [asdict(row) for row in averaged_rows],
        "dataset_averaged_psnr_rows": [asdict(row) for row in dataset_averaged_rows],
    }
    (args.output_dir / "quality_virtual_psnr_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(averaged_rows)} averaged PSNR rows and plots to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
