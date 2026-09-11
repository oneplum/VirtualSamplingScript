#!/usr/bin/env python3

import csv
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, TextIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

OutputType = Literal["csv", "parquet"]

METHOD_LABELS = {
    "lin": "linear",
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
    "Aneurism1": "Aneurism, Pulse Transfer-Function",
    "Aneurism2": "Aneurism, Smoothstep Transfer-Function",
    "Sphere": "Spherical Distance Function, multi Pulse Transfer Function",
    "ML1": "Marschner-Lobb, Pulse Transfer-Function",
    "ML2": "Marschner-Lobb, Smoothstep Transfer-Function",
}
DATASET_ORDER = list(DATASET_LABELS.keys())

VIRTUAL_METHOD_LABELS = {
    "none": "None",
    "linvs": "linear",
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

    # @classmethod
    # def from_obj(cls, obj, dic: dict[str, Any] = {}):
    #     for obj_field in fields(cls):
    #         value = getattr(obj, obj_field.name, None)
    #         if value is not None:
    #             dic[obj_field.name] = covert_value(value, obj_field.type)
    #     return cls(**dic)


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

    pq.write_table(
        pa.Table.from_pylist(rows),
        output_path,
        compression="zstd",
    )


def write_to_file(
    output_path: Path, rows: list[dict], output_type: OutputType = "csv"
) -> None:
    if not rows:
        return

    if output_type == "parquet":
        write_parquet(output_path, rows)
    else:
        write_csv(output_path, rows)
