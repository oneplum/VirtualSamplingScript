#!/usr/bin/env python3

import csv
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, TextIO

import numpy as np
import polars as pl
from PIL import Image

from parse_performance_log import ParsedLabel

OutputType = Literal["csv", "parquet"]

METHOD_LABELS = {
    "lin": "Trilinear",
    "quadB": "Quadratic B-Spline",
    "quadBf": "Quadratic B-Spline Prefiltered",
    "cubicB": "Cubic B-Spline",
    "cubicBf": "Cubic B-Spline Prefiltered",
    "cr": "Catmull-Rom Spline",
}
METHOD_ORDER = list(METHOD_LABELS.keys())

DATASET_LABELS = {
    "Head1": "Visible Human Head, Skin Transfer Function",
    "Head2": "Visible Human Head, Bones Transfer Function",
    "Tree1": "Christmas Tree, close-up",
    "Tree2": "Christmas Tree, wide shot",
    "Aneurism1": "Aneurysm, Pulse Transfer-Function",
    "Aneurism2": "Aneurysm, Smoothstep Transfer-Function",
    "Sphere": "Spherical Distance Function, multi Pulse Transfer Function",
    "ML1": "Marschner-Lobb, Pulse Transfer-Function",
    "ML2": "Marschner-Lobb, Smoothstep Transfer-Function",
}
DATASET_ORDER = list(DATASET_LABELS.keys())

VIRTUAL_METHOD_LABELS = {
    "none": "None",
    "linvs": "Linear",
    "crvs": "Catmull-Rom Spline",
    "hermvs": "Hermite Spline",
    "monhermvs": "Monotone Hermite Spline",
}
VIRTUAL_METHOD_ORDER = list(VIRTUAL_METHOD_LABELS.keys())

TRANS_TYPE_LABELS = {"rot": "Rotation", "trans": "Translate"}
TRANS_AXIS_LABELS = {
    "x": "Axis-X",
    "y": "Axis-Y",
    "z": "Axis-Z",
    "diag": "Axis-Diagonal",
}

SAMPLING_METHOD_LABEL = {}
for m in METHOD_ORDER:
    for vm in VIRTUAL_METHOD_ORDER:
        s = METHOD_LABELS.get(m, m)
        if vm != "none":
            s = f"{s} + {VIRTUAL_METHOD_LABELS.get(vm, vm)}"
        SAMPLING_METHOD_LABEL[f"{m}-{vm}"] = s
SAMPLING_METHOD_ORDER = list(SAMPLING_METHOD_LABEL.keys())


@dataclass
class FrameBasic:
    dataset_name: str
    level: int
    method: str
    lighting_enabled: bool
    true_samples: int
    virtual_sampling_method: str
    virtual_samples: int
    trans_type: str = field(default="rot", kw_only=True)
    trans_axis: str = field(default="y", kw_only=True)


@dataclass(frozen=True)
class ColMeta:
    label: str
    values: dict[Any, str] = field(default_factory=dict)
    order: list[Any] = field(default_factory=list)

    def get_val_label(self, value: Any) -> str:
        return self.values.get(value, str(value))

    def order_vals(self, values) -> list[Any]:
        curr = set(values)
        if self.order:
            return [val for val in self.order if val in curr]
        return sorted(curr)


@dataclass
class QualityImage:
    path: Path
    label: str
    dataset_name: str
    dataset_label: str
    image_kind: str
    level: int | None
    method: str
    method_label: str
    lighting_enabled: bool
    rotation: int
    baseline_variant: str | None
    true_samples: int | None
    virtual_sampling_method: str | None
    virtual_samples: int | None


COL_META = {
    "dataset_name": ColMeta(
        label="Dataset", values=DATASET_LABELS, order=DATASET_ORDER
    ),
    "level": ColMeta(
        label="Downsamples",
        values={i: f"Downloaded {i}" for i in range(3)},
        order=list(range(3)),
    ),
    "method": ColMeta(label="True Method", values=METHOD_LABELS, order=METHOD_ORDER),
    "lighting_enabled": ColMeta(
        label="Illumination",
        values={
            True: "Lighting enabled",
            False: "Lighting disabled",
        },
        order=[False, True],
    ),
    "true_samples": ColMeta(
        label="Sampling Rate",
        values={1: "1x Samples", 5: "5x Samples", 10: "10x Samples", 15: "15x Samples"},
    ),
    "virtual_sampling_method": ColMeta(
        label="Virtual Method",
        values=VIRTUAL_METHOD_LABELS,
        order=VIRTUAL_METHOD_ORDER,
    ),
    "virtual_samples": ColMeta(
        label="Virtual Samples", values={i: f"{i} Samples" for i in range(20)}
    ),
    "trans_type": ColMeta(
        label="Transform Type",
        values=TRANS_TYPE_LABELS,
        order=list(TRANS_TYPE_LABELS.keys()),
    ),
    "trans_axis": ColMeta(
        label="Transform Axis",
        values=TRANS_AXIS_LABELS,
        order=list(TRANS_AXIS_LABELS.keys()),
    ),
    "sampling_method": ColMeta(
        label="Method",
        values=SAMPLING_METHOD_LABEL,
        order=list(SAMPLING_METHOD_LABEL.keys()),
    ),
}


def get_label(col: str, val: str | None = None) -> str:
    if val is None:
        return COL_META[col].label if col in COL_META else col

    return COL_META[col].get_val_label(val) if col in COL_META else val


def covert_value(value: Any, target_type: type):
    if isinstance(value, target_type):
        return value

    if target_type == bool and isinstance(value, str):
        return value.lower() == "true"

    return target_type(value)


def warn(message: str, *, stream: TextIO) -> None:
    print(f"Warning: {message}", file=stream)


def parse_name(path_name: str) -> FrameBasic:
    parts = path_name.split("-")
    basic_fields = fields(FrameBasic)
    min_lens = min(len(parts), len(basic_fields))
    values = {}
    for key in range(min_lens):
        field_name = basic_fields[key].name
        field_value = parts[key]
        if field_name == "lighting_enabled":
            values[field_name] = field_value == "l"
        elif field_name == "virtual_sampling_method":
            values[field_name] = field_value if field_value != "n" else "none"
        else:
            values[field_name] = covert_value(parts[key], basic_fields[key].type)
    return FrameBasic(**values)


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(output_path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    df = pl.DataFrame(rows)
    df.write_parquet(output_path)


def write_to_file(
    output_path: Path, rows: list[dict], output_type: OutputType = "csv"
) -> None:
    if not rows:
        return

    if output_type == "parquet":
        write_parquet(output_path, rows)
    else:
        write_csv(output_path, rows)


def parse_measurement_label(path: Path) -> tuple[ParsedLabel | None, str | None]:
    frame_basic = parse_name(path.name)
    if frame_basic.trans_type != "rot" or frame_basic.trans_axis != "y":
        return None, "exclude other motion except rotation x"

    image_kind = ""
    if frame_basic.level == 0:
        if frame_basic.virtual_sampling_method == "none":
            baseline_source = "analytic" if frame_basic.method == "analytic" else "discrete"
            if frame_basic.true_samples == 8:
                baseline_variant = "legacy_baseline"
                image_kind = f"{baseline_variant}_{baseline_source}"
            else:
                baseline_variant = f"baseline{frame_basic.true_samples}"
                image_kind = "baseline_reference"
        else:
            return None, "not include quality process"
    else:
        image_kind = "virtual_sampled" if frame_basic.virtual_sampling_method == "n" else "downsampled"
        baseline_source = None
        baseline_variant = None

    return ParsedLabel(
        dataset_name=frame_basic.dataset_name,
        image_kind=image_kind,
        level=frame_basic.level,
        method=frame_basic.method,
        lighting_enabled=frame_basic.lighting_enabled,
        virtual_sampling_method=frame_basic.virtual_sampling_method if frame_basic.virtual_sampling_method != "none" else None,
        virtual_samples=frame_basic.virtual_samples if frame_basic.virtual_samples > 0 else None,
        rotation=0,
        baseline_source=baseline_source,
        baseline_variant=baseline_variant,
        true_samples=frame_basic.true_samples if frame_basic.virtual_sampling_method != "none" else None,
    ), None


def parse_roation(label: str) -> tuple[int | None, str | None]:
    if not label.endswith(".png"):
        return None, "label does not end with .png"

    parts = label[:-4].split("-")
    return parts[-1], None


def load_quality_images(images_dir: Path, *, warning_stream) -> list[QualityImage]:
    
    images: list[QualityImage] = []

    images_dir = sorted(path for path in images_dir.iterdir() if path.is_dir())
    total = len(images_dir)
    print(f"Found {total} folders.")

    for image_dir in images_dir:
        parsed_label, error = parse_measurement_label(image_dir)
        if error is not None or parsed_label is None:
            warn(
                f"skipping unrecognized quality image name {image_dir.name!r} ({error})",
                stream=warning_stream,
            )
            continue

        for path in sorted(image_dir.glob("*.png")):
            rotation, error = parse_roation(path.name)
            if error is not None or rotation is None:
                warn(
                    f"skipping rotation error {path.name!r} ({error})",
                    stream=warning_stream,
                )
                continue
            
            images.append(
                QualityImage(
                    path=path,
                    label=path.name,
                    dataset_name=parsed_label.dataset_name,
                    dataset_label=DATASET_LABELS.get(parsed_label.dataset_name, parsed_label.dataset_name),
                    image_kind=parsed_label.image_kind,
                    level=parsed_label.level,
                    method=parsed_label.method,
                    method_label=METHOD_LABELS.get(parsed_label.method, parsed_label.method),
                    lighting_enabled=parsed_label.lighting_enabled,
                    rotation=rotation,
                    baseline_variant=parsed_label.baseline_variant,
                    true_samples=parsed_label.true_samples,
                    virtual_sampling_method=parsed_label.virtual_sampling_method,
                    virtual_samples=parsed_label.virtual_samples,
                )
            )
    return images