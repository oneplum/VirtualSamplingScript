#!/usr/bin/env python3

from pathlib import Path
from typing import TextIO, Any
from PIL import Image
import numpy as np
from dataclasses import dataclass, fields
import csv

PATTERN = "*.png"

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
    "linvs": "linear",
    "crvs": "Catmull-Rom Spline",
    "hermvs": "Hermite Spline",
    "monhermvs": "Monotone Hermite Spline"
}
VIRTUAL_METHOD_ORDER = list(VIRTUAL_METHOD_LABELS.keys())

@dataclass
class FrameBasic:
    dataset_name: str
    level: int
    method: str
    lighting_enabled: bool
    true_samples: int
    virtual_sampling_method: str
    virtual_samples: int

    @classmethod
    def from_obj(cls, obj, dic={}):
        for field in fields(cls):
            value = getattr(obj, field.name, None)
            if value is not None:
                dic[field.name] = covert_value(value, field.type)
        return cls(**dic)

    def get_dataset_label(self) -> str:
        return DATASET_LABELS.get(self.dataset_name, self.dataset_name)
    
    def get_method_label(self) -> str:
        return METHOD_LABELS.get(self.method, self.method)

    def get_lighting_label(self) -> str:
        return "Lighting On" if self.lighting_enabled else "Lighting Off"

    def get_virtual_sampling_method_label(self) -> str:
        return VIRTUAL_METHOD_LABELS.get(self.virtual_sampling_method, "")

@dataclass
class FrameImage(FrameBasic):
    path: Path
    label: str
    number: int

@dataclass
class FrameSeq(FrameBasic):
    path: Path
    label: str
    frame_images: list[FrameImage]

def covert_value(value: Any, target_type: type):
    if isinstance(value, target_type):
        return value

    if target_type == bool and isinstance(value, str):
        return value.lower() == "true"

    return target_type(value)

def warn(message: str, *, stream: TextIO) -> None:
    print(f"Warning: {message}", file=stream)

def parse_name(path_name: str) -> FrameBasic | None:
    parts = path_name.split("-")
    basic_fields = fields(FrameBasic)
    if len(parts) < len(basic_fields):
        return None
    values = {}
    for key, basic_field in enumerate(basic_fields):
        if basic_field.name == "lighting_enabled":
            values[basic_field.name] = (
                True if parts[key] == "l" else False
            )
        elif basic_field.name == "virtual_sampling_method":
            values[basic_field.name] = (
                parts[key] if parts[key] != 'n' else "None"
            )
        else:
            values[basic_field.name] = covert_value(parts[key], basic_field.type)
    return FrameBasic(**values)

def load_frames_images(images_dir: Path, *, warning_stream) -> list[FrameImage]:
    images: list[FrameImage] = []
    i:int = 0
    for path in sorted(images_dir.glob(PATTERN), key=lambda x: int(x.stem.split("-")[-1])):
        frameBasic = parse_name(path_name=path.name)
        if frameBasic is None:
            warn(
                f"invalid frames directory",
                stream=warning_stream,
            )
            continue

        parts = path.name[:-4].split("-")
        if len(parts) < 8:
            warn(
                f"expected at least 8 name parts, got {len(parts)}",
                stream=warning_stream,
            )
            continue

        if not parts[7].isdigit() or int(parts[7]) != i:
            warn(
                f"missing frame data at position {i}.",
                stream=warning_stream,
            )
            return images
        i += 1

        images.append(FrameImage.from_obj(frameBasic, {
            "number": covert_value(parts[7], int),
            "path": path,
            "label": path.name
            }))

    return images

def load_frames_seqs(path: Path, is_one_seq: bool = False, *, warning_stream) -> list[FrameSeq]:
    seqs: list[FrameSeq] = []
    paths = path.iterdir() if not is_one_seq else [path]
    for cpath in paths:
        if not cpath.is_dir():
            warn(
                f"{cpath.name} is not a direcotry",
                stream=warning_stream
            )
            continue

        frameBasic = parse_name(path_name=cpath.name)
        if frameBasic is None:
            warn(
                f"{cpath.name} - invalid directory",
                stream=warning_stream
            )
            continue

        seqs.append(FrameSeq.from_obj(frameBasic, {
            "path": cpath,
            "label": cpath.name,
            "frame_images": load_frames_images(cpath, warning_stream=warning_stream)
            }))
    return seqs

def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64)

def write_csv_rows(path: Path, row_dicts: list[dict[str, Any]]) -> None:
    if not row_dicts:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)

def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)
    