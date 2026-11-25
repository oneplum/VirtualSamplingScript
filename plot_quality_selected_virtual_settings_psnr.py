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
DATASET_ORDER = list(DATASET_LABELS.keys())

VIRTUAL_METHOD_LABELS = {
    "linvs": "linear",
    "crvs": "Catmull-Rom Spline",
    "hermvs": "Hermite Spline",
    "monhermvs": "Monotone Hermite Spline",
}

METHOD_ORDER = ["lin", "quadB", "quadBf", "cubicB", "cubicBf", "cr"]
VIRTUAL_METHOD_ORDER = ["linvs", "crvs", "hermvs", "monhermvs"]
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


@dataclass(frozen=True)
class SelectedSetting:
    method: str
    virtual_sampling_method: str | None
    virtual_samples: int | None
    uses_virtual_sampling: bool
    setting_key: str
    setting_label: str


@dataclass
class PsNrRow:
    dataset_name: str
    dataset_label: str
    reference_kind: str
    lighting_enabled: bool
    level: int
    rotation: int
    setting_key: str
    setting_label: str
    method: str
    virtual_sampling_method: str
    virtual_samples: int
    reference_label: str
    reference_count: int
    target_label: str
    psnr_db: float


@dataclass
class DatasetAveragedPsNrRow:
    dataset_name: str
    dataset_label: str
    reference_kind: str
    lighting_enabled: bool
    level: int
    setting_key: str
    setting_label: str
    method: str
    virtual_sampling_method: str
    virtual_samples: int
    rotation_count: int
    mean_psnr_db: float
    stddev_psnr_db: float


@dataclass
class LevelAveragedPsNrRow:
    reference_kind: str
    lighting_enabled: bool
    level: int
    setting_key: str
    setting_label: str
    method: str
    virtual_sampling_method: str
    virtual_samples: int
    dataset_count: int
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
        return np.asarray(image.convert("RGB"), dtype=np.float64)


def compute_psnr(reference_path: Path, target_path: Path) -> float:
    reference = _load_rgb_image(reference_path)
    target = _load_rgb_image(target_path)
    if reference.shape != target.shape:
        raise ValueError(
            f"image shapes differ: {reference.shape} vs {target.shape} for {reference_path} and {target_path}"
        )
    mse = np.mean((reference - target) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def _write_csv_rows(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    row_dicts = [asdict(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)


def _write_reproduction_command(output_dir: Path) -> None:
    command_text = shlex.join(["python3", str(Path(sys.argv[0]).resolve()), *sys.argv[1:]])
    (output_dir / "reproduce_command.txt").write_text(command_text + "\n", encoding="utf-8")


def parse_selected_setting(value: str) -> SelectedSetting:
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            f"invalid setting {value!r}; expected METHOD:none or METHOD:VIRTUAL_METHOD:COUNT"
        )
    if len(parts) == 2:
        method, virtual_method = parts
        virtual_samples_text = None
    else:
        method, virtual_method, virtual_samples_text = parts
    if method not in METHOD_ORDER:
        raise argparse.ArgumentTypeError(f"unknown method {method!r}")
    if virtual_method == "none":
        if virtual_samples_text not in (None, "0"):
            raise argparse.ArgumentTypeError(
                f"no-virtual setting {value!r} must not provide a count other than 0"
            )
        setting_key = f"{method}:none"
        setting_label = f"{human_method_label(method)}\nNo virtual samples"
        return SelectedSetting(
            method=method,
            virtual_sampling_method=None,
            virtual_samples=None,
            uses_virtual_sampling=False,
            setting_key=setting_key,
            setting_label=setting_label,
        )
    if virtual_method not in VIRTUAL_METHOD_ORDER:
        raise argparse.ArgumentTypeError(f"unknown virtual method {virtual_method!r}")
    if virtual_samples_text is None:
        raise argparse.ArgumentTypeError(
            f"invalid setting {value!r}; virtual-sampled settings need a count"
        )
    if not virtual_samples_text.isdigit():
        raise argparse.ArgumentTypeError(
            f"invalid virtual sample count {virtual_samples_text!r}"
        )
    virtual_samples = int(virtual_samples_text)
    setting_key = f"{method}:{virtual_method}:{virtual_samples}"
    setting_label = (
        f"{human_method_label(method)}\n"
        f"{human_virtual_method_label(virtual_method)}\n"
        f"VS {virtual_samples}"
    )
    return SelectedSetting(
        method=method,
        virtual_sampling_method=virtual_method,
        virtual_samples=virtual_samples,
        uses_virtual_sampling=True,
        setting_key=setting_key,
        setting_label=setting_label,
    )


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


def compute_selected_virtual_setting_psnr_rows(
    images: list[QualityImage],
    *,
    settings: list[SelectedSetting],
    baseline_true_samples: int,
    discrete_baseline_method: str,
    discrete_reference_mode: str,
    warning_stream,
) -> list[PsNrRow]:
    setting_lookup = {setting.setting_key: setting for setting in settings}
    reference_index = build_reference_index(
        images,
        discrete_baseline_method=discrete_baseline_method,
        discrete_reference_mode=discrete_reference_mode,
    )
    analytic_datasets = {
        image.dataset_name for image in images if image.image_kind == "legacy_baseline_analytic"
    }
    rows: list[PsNrRow] = []

    for image in images:
        setting: SelectedSetting | None
        setting_key: str
        if (
            image.image_kind == "virtual_sampled"
            and image.level is not None
            and image.virtual_sampling_method is not None
            and image.virtual_samples is not None
        ):
            setting_key = (
                f"{image.method}:{image.virtual_sampling_method}:{image.virtual_samples}"
            )
        elif (
            image.image_kind == "baseline_reference"
            and image.level is not None
            and image.baseline_variant == f"baseline{baseline_true_samples}"
        ):
            setting_key = f"{image.method}:none"
        else:
            continue

        setting = setting_lookup.get(setting_key)
        if setting is None:
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

        rows.append(
            PsNrRow(
                dataset_name=image.dataset_name,
                dataset_label=image.dataset_label,
                reference_kind=references[0].image_kind,
                lighting_enabled=image.lighting_enabled,
                level=image.level,
                rotation=image.rotation,
                setting_key=setting.setting_key,
                setting_label=setting.setting_label,
                method=image.method,
                virtual_sampling_method=(
                    image.virtual_sampling_method if image.virtual_sampling_method is not None else "none"
                ),
                virtual_samples=(
                    image.virtual_samples if image.virtual_samples is not None else 0
                ),
                reference_label=", ".join(reference_labels),
                reference_count=len(reference_labels),
                target_label=image.label,
                psnr_db=statistics.fmean(psnr_values),
            )
        )

    setting_order = {setting.setting_key: index for index, setting in enumerate(settings)}
    rows.sort(
        key=lambda row: (
            row.dataset_name,
            row.level,
            row.lighting_enabled,
            setting_order[row.setting_key],
            row.rotation,
        )
    )
    return rows


def average_rows_by_dataset(rows: list[PsNrRow]) -> list[DatasetAveragedPsNrRow]:
    grouped: dict[tuple[str, str, bool, int, str], list[PsNrRow]] = {}
    for row in rows:
        key = (
            row.dataset_name,
            row.reference_kind,
            row.lighting_enabled,
            row.level,
            row.setting_key,
        )
        grouped.setdefault(key, []).append(row)

    averaged_rows: list[DatasetAveragedPsNrRow] = []
    for grouped_rows in grouped.values():
        first = grouped_rows[0]
        psnr_values = [row.psnr_db for row in grouped_rows]
        averaged_rows.append(
            DatasetAveragedPsNrRow(
                dataset_name=first.dataset_name,
                dataset_label=first.dataset_label,
                reference_kind=first.reference_kind,
                lighting_enabled=first.lighting_enabled,
                level=first.level,
                setting_key=first.setting_key,
                setting_label=first.setting_label,
                method=first.method,
                virtual_sampling_method=first.virtual_sampling_method,
                virtual_samples=first.virtual_samples,
                rotation_count=len(grouped_rows),
                mean_psnr_db=statistics.fmean(psnr_values),
                stddev_psnr_db=statistics.pstdev(psnr_values) if len(psnr_values) > 1 else 0.0,
            )
        )

    return averaged_rows


def average_rows_by_level(rows: list[DatasetAveragedPsNrRow]) -> list[LevelAveragedPsNrRow]:
    grouped: dict[tuple[bool, int, str], list[DatasetAveragedPsNrRow]] = {}
    for row in rows:
        key = (
            row.lighting_enabled,
            row.level,
            row.setting_key,
        )
        grouped.setdefault(key, []).append(row)

    averaged_rows: list[LevelAveragedPsNrRow] = []
    for grouped_rows in grouped.values():
        first = grouped_rows[0]
        psnr_values = [row.mean_psnr_db for row in grouped_rows]
        averaged_rows.append(
            LevelAveragedPsNrRow(
                reference_kind="combined",
                lighting_enabled=first.lighting_enabled,
                level=first.level,
                setting_key=first.setting_key,
                setting_label=first.setting_label,
                method=first.method,
                virtual_sampling_method=first.virtual_sampling_method,
                virtual_samples=first.virtual_samples,
                dataset_count=len(grouped_rows),
                mean_psnr_db=statistics.fmean(psnr_values),
                stddev_psnr_db=statistics.pstdev(psnr_values) if len(psnr_values) > 1 else 0.0,
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


def _dataset_sort_key(dataset_name: str) -> tuple[int, str]:
    if dataset_name in DATASET_ORDER:
        return (DATASET_ORDER.index(dataset_name), dataset_name)
    return (len(DATASET_ORDER), dataset_name)


def _setting_order(settings: list[SelectedSetting]) -> dict[str, int]:
    return {setting.setting_key: index for index, setting in enumerate(settings)}


def _plot_setting_rows(
    rows: list[DatasetAveragedPsNrRow] | list[LevelAveragedPsNrRow],
    *,
    settings: list[SelectedSetting],
    title: str,
    output_base: Path,
    output_format: str,
    show_error_bars: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(max(9.0, 1.6 * len(settings)), 4.8), constrained_layout=True)
    setting_order = _setting_order(settings)
    sorted_rows = sorted(rows, key=lambda row: setting_order[row.setting_key])
    x_values = [setting_order[row.setting_key] for row in sorted_rows]
    y_values = [row.mean_psnr_db for row in sorted_rows]
    y_errors = [row.stddev_psnr_db for row in sorted_rows]

    for row in sorted_rows:
        index = setting_order[row.setting_key]
        color = SETTING_COLORS[index % len(SETTING_COLORS)]
        if show_error_bars:
            ax.errorbar(
                [index],
                [row.mean_psnr_db],
                yerr=[row.stddev_psnr_db],
                fmt="o",
                color=color,
                capsize=4,
            )
        else:
            ax.scatter([index], [row.mean_psnr_db], color=color)

    ax.set_xticks(list(range(len(settings))))
    ax.set_xticklabels([setting.setting_label for setting in settings], fontsize=9)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)

    _write_csv_rows(output_base.with_suffix(".csv"), sorted_rows)
    if output_format in ("png", "both"):
        fig.savefig(output_base.with_suffix(".png"), dpi=220)
    if output_format in ("pdf", "both"):
        fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_level_dataset_breakdown_psnr(
    rows: list[DatasetAveragedPsNrRow],
    *,
    settings: list[SelectedSetting],
    output_dir: Path,
    discrete_baseline_method: str,
    discrete_reference_mode: str,
    output_format: str,
    show_error_bars: bool,
) -> None:
    grouped: dict[tuple[bool, int], list[DatasetAveragedPsNrRow]] = {}
    for row in rows:
        grouped.setdefault((row.lighting_enabled, row.level), []).append(row)

    setting_order = _setting_order(settings)
    dataset_names = sorted(
        {row.dataset_name for row in rows},
        key=_dataset_sort_key,
    )

    for (lighting_enabled, level), plot_rows in sorted(grouped.items()):
        fig, ax = plt.subplots(
            figsize=(max(10.5, 1.4 * len(dataset_names)), 4.8),
            constrained_layout=True,
        )
        x_positions = {dataset_name: index for index, dataset_name in enumerate(dataset_names)}

        for setting in settings:
            series_rows = sorted(
                [row for row in plot_rows if row.setting_key == setting.setting_key],
                key=lambda row: _dataset_sort_key(row.dataset_name),
            )
            if not series_rows:
                continue

            series_x = [x_positions[row.dataset_name] for row in series_rows]
            series_y = [row.mean_psnr_db for row in series_rows]
            series_err = [row.stddev_psnr_db for row in series_rows]
            color = SETTING_COLORS[setting_order[setting.setting_key] % len(SETTING_COLORS)]
            plot_kwargs = {
                "color": color,
                "label": setting.setting_label.replace("\n", " | "),
            }
            if show_error_bars:
                ax.errorbar(
                    series_x,
                    series_y,
                    yerr=series_err,
                    capsize=4,
                    fmt="o",
                    **plot_kwargs,
                )
            else:
                ax.scatter(series_x, series_y, **plot_kwargs)

        ax.set_xticks(list(range(len(dataset_names))))
        ax.set_xticklabels(
            [human_dataset_label(dataset_name) for dataset_name in dataset_names],
            rotation=35,
            ha="right",
            fontsize=9,
        )
        ax.set_ylabel("PSNR (dB)")
        basename = (
            "quality_selected_virtual_settings_"
            f"level_{level}_"
            f"{'lit' if lighting_enabled else 'unlit'}_"
            "datasets_"
            f"ref_{_slugify('average_all' if discrete_reference_mode == 'average_all' else 'per_method' if discrete_reference_mode == 'per_method' else discrete_baseline_method)}"
        )
        title = (
            f"Datasets compared | Level {level} | "
            f"{lighting_label(lighting_enabled)} | "
            f"{_combined_reference_text(discrete_baseline_method=discrete_baseline_method, discrete_reference_mode=discrete_reference_mode)}"
        )
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

        _write_csv_rows(output_dir / f"{basename}.csv", sorted(
            plot_rows,
            key=lambda row: (_dataset_sort_key(row.dataset_name), setting_order[row.setting_key]),
        ))
        if output_format in ("png", "both"):
            fig.savefig((output_dir / basename).with_suffix(".png"), dpi=220)
        if output_format in ("pdf", "both"):
            fig.savefig((output_dir / basename).with_suffix(".pdf"))
        plt.close(fig)


def plot_level_summary_psnr(
    rows: list[LevelAveragedPsNrRow],
    *,
    settings: list[SelectedSetting],
    output_dir: Path,
    discrete_baseline_method: str,
    discrete_reference_mode: str,
    output_format: str,
    show_error_bars: bool,
) -> None:
    grouped: dict[tuple[bool, int], list[LevelAveragedPsNrRow]] = {}
    for row in rows:
        grouped.setdefault((row.lighting_enabled, row.level), []).append(row)

    for (lighting_enabled, level), plot_rows in sorted(grouped.items()):
        basename = (
            "quality_selected_virtual_settings_"
            f"level_{level}_"
            f"{'lit' if lighting_enabled else 'unlit'}_"
            f"ref_{_slugify('average_all' if discrete_reference_mode == 'average_all' else 'per_method' if discrete_reference_mode == 'per_method' else discrete_baseline_method)}"
        )
        dataset_count = max((row.dataset_count for row in plot_rows), default=0)
        title = (
            f"All datasets averaged ({dataset_count}) | Level {level} | "
            f"{lighting_label(lighting_enabled)} | "
            f"{_combined_reference_text(discrete_baseline_method=discrete_baseline_method, discrete_reference_mode=discrete_reference_mode)}"
        )
        _plot_setting_rows(
            plot_rows,
            settings=settings,
            title=title,
            output_base=output_dir / basename,
            output_format=output_format,
            show_error_bars=show_error_bars,
        )


def _clear_generated_outputs(directory: Path) -> None:
    for pattern in (
        "quality_selected_virtual_settings_*.png",
        "quality_selected_virtual_settings_*.pdf",
        "quality_selected_virtual_settings_*.csv",
    ):
        for path in directory.glob(pattern):
            path.unlink()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare selected virtual-sampling settings against the final baseline quality."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing the generated screenshot images.",
    )
    parser.add_argument(
        "--setting",
        dest="settings",
        action="append",
        type=parse_selected_setting,
        required=True,
        help="Selected setting in the form METHOD:none or METHOD:VIRTUAL_METHOD:COUNT. Repeat for multiple settings.",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=10,
        help="Use baselineN images for no-virtual settings, for example 10 selects baseline10.",
    )
    parser.add_argument(
        "--discrete-reference-mode",
        choices=("shared", "per_method", "average_all"),
        default="average_all",
        help="How to choose the discrete-dataset baseline reference.",
    )
    parser.add_argument(
        "--discrete-baseline-method",
        choices=METHOD_ORDER,
        default="cubicBf",
        help="Discrete baseline method used when --discrete-reference-mode shared is selected.",
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
        default=Path("plots/plot_quality_selected_virtual_settings_psnr"),
        help="Directory for plots, CSV files, summary JSON, and reproduction command.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    settings: list[SelectedSetting] = []
    seen_keys: set[str] = set()
    for setting in args.settings:
        if setting.setting_key in seen_keys:
            continue
        seen_keys.add(setting.setting_key)
        settings.append(setting)

    images = load_quality_images(args.images_dir, warning_stream=sys.stderr)
    print(f"Found {len(images)} folders.")

    psnr_rows = compute_selected_virtual_setting_psnr_rows(
        images,
        settings=settings,
        baseline_true_samples=args.baseline_samples,
        discrete_baseline_method=args.discrete_baseline_method,
        discrete_reference_mode=args.discrete_reference_mode,
        warning_stream=sys.stderr,
    )
    dataset_averaged_rows = average_rows_by_dataset(psnr_rows)
    level_averaged_rows = average_rows_by_level(dataset_averaged_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _clear_generated_outputs(args.output_dir)
    _write_reproduction_command(args.output_dir)
    plot_level_dataset_breakdown_psnr(
        dataset_averaged_rows,
        settings=settings,
        output_dir=args.output_dir,
        discrete_baseline_method=args.discrete_baseline_method,
        discrete_reference_mode=args.discrete_reference_mode,
        output_format=args.output_format,
        show_error_bars=args.include_error_bars,
    )
    plot_level_summary_psnr(
        level_averaged_rows,
        settings=settings,
        output_dir=args.output_dir,
        discrete_baseline_method=args.discrete_baseline_method,
        discrete_reference_mode=args.discrete_reference_mode,
        output_format=args.output_format,
        show_error_bars=args.include_error_bars,
    )

    summary_payload = {
        "images_dir": str(args.images_dir),
        "baseline_samples": args.baseline_samples,
        "discrete_reference_mode": args.discrete_reference_mode,
        "discrete_baseline_method": args.discrete_baseline_method,
        "settings": [asdict(setting) for setting in settings],
        "output_format": args.output_format,
        "psnr_rows": [asdict(row) for row in psnr_rows],
        "dataset_averaged_psnr_rows": [asdict(row) for row in dataset_averaged_rows],
        "level_averaged_psnr_rows": [asdict(row) for row in level_averaged_rows],
    }
    (args.output_dir / "quality_selected_virtual_settings_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Wrote {len(dataset_averaged_rows)} dataset-averaged rows and "
        f"{len(level_averaged_rows)} level-summary rows to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
