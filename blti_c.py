#!/usr/bin/env python3

"""
input: directory path of sequences
output: csv or parquet
sequence name format: volname-level-method-lighting-rate-vmethod-subdiv-transforms-axis
row column: dataset_name,level,method,lighting_enabled,true_samples,virtual_sampling_method,virtual_samples,trans_type,trans_axis,frame_idx,blti_0,blti_1,blti_2,blti_3,blti_4,blti_5
"""

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import load_rgb_image, parse_name, warn, write_to_file

BATCH_SIZE = 100


PATTERN = "*.png"
BLTI_K = 6
BLTI_SIGMA0 = 1.0
BLTI_ALPHA = 2**0.5
BLTI_P = 95.0
BLTI_GAMMA = 2.0
BLTI_EPS = 1e-6

GAUSSIAN_SCALES = tuple(BLTI_SIGMA0 * (BLTI_ALPHA**k) for k in range(BLTI_K + 1))


def _luminance_image(path: Path) -> np.ndarray:
    rgb = load_rgb_image(path) / 255.0
    limage = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return limage


def _edge_weights(frame: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.hypot(gx, gy)

    q_p = np.percentile(gradient, BLTI_P)
    if q_p == 0:
        return np.zeros_like(gradient, dtype=np.float64)

    normalized = np.clip(gradient / q_p, 0.0, 1.0)

    return normalized**BLTI_GAMMA


def _weighted_rms(weight: np.ndarray, X: np.ndarray) -> float:
    sum_weight = np.sum(weight)
    if sum_weight < BLTI_EPS:
        return float(np.sqrt(np.mean(X**2)))

    return float(np.sqrt(np.sum(weight * X**2) / sum_weight))


def _compute_bands(frame: np.ndarray) -> list[np.ndarray]:
    bands: list[np.ndarray] = []

    curr = cv2.GaussianBlur(frame, (0, 0), sigmaX=GAUSSIAN_SCALES[0])

    for sigma in GAUSSIAN_SCALES[1:]:
        next = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma)
        bands.append(curr - next)
        curr = next

    return bands

def _load_and_blur(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    img = _luminance_image(path)
    bands = _compute_bands(img)

    return img, bands


def show_img(data: list[tuple[np.ndarray, str, str | None]], title: str, path: Path) -> int:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )
    c = len(data)
    col = min(2, c)
    row = math.ceil(c / col)
    fig, axes = plt.subplots(row, col, figsize=(10,10), squeeze=False)

    for ax, (px, label, cmap) in zip(axes.flat, data):
        if cmap == "hot":
            im = ax.imshow(px, cmap='hot')
            ax.set_title(label)
        elif cmap == "gray":
            im = ax.imshow(px, cmap='gray', vmin=0, vmax=1)
            ax.set_title(label)
        else:
            im = ax.imshow(px)
            ax.set_title(label)

        fig.colorbar(im, ax=ax)
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(path / title, bbox_inches='tight', dpi=220)

    plt.close(fig)

    return 0


def compute_blti(paths: list[Path], output_dir: Path) -> dict[int, list[float]]:
    if len(paths) < 3:
        return {}

    bltis: dict[int, list[float]] = {}

    with ThreadPoolExecutor(max_workers=2) as io_executor:
        data = [io_executor.submit(_load_and_blur, p) for p in paths[:2]]

        prev_image, prev_bands = data[0].result()
        curr_image, curr_bands = data[1].result()

        next_data = io_executor.submit(_load_and_blur, paths[2])

        for t in range(1, len(paths) - 1):
            next_image, next_bands = next_data.result()

            if t < len(paths) - 2:
                next_data = io_executor.submit(_load_and_blur, paths[t+2])

            weight = _edge_weights(curr_image)

            blti_t: list[float] = []

            for k in range(BLTI_K):
                band_prev = prev_bands[k]
                band_curr = curr_bands[k]
                band_next = next_bands[k]

                R_t = band_curr - 0.5 * (band_prev + band_next)
                S_t = 0.5 * (band_next - band_prev)

                if k == 0:
                    show_img([
                        (load_rgb_image(paths[t]) / 255.0, "Original Image", None),
                        (weight, "Edge Weight", "gray"), 
                        (weight * R_t ** 2, "weight * R_t ** 2", "hot"), 
                        (weight * S_t ** 2, "weight * S_t ** 2", "hot")
                    ], f"{paths[t].name}", output_dir)

                blti_t_k = _weighted_rms(weight, R_t) / max(
                    _weighted_rms(weight, S_t), BLTI_EPS
                )
                blti_t.append(blti_t_k)

            bltis[t] = blti_t

            prev_image, prev_bands = curr_image, curr_bands
            curr_image, curr_bands = next_image, next_bands

    return bltis


def compute_sequence(path: Path, output_dir: Path, frame_start: int = 0, frame_end:int = -1) -> tuple[str, list[dict[str, Any]], str]:
    try:
        frame_paths = sorted(
            path.glob(PATTERN), key=lambda x: int(x.stem.rsplit("-", 1)[-1])
        )

        frame_paths = frame_paths[frame_start: frame_end]

        if len(frame_paths) < 3:
            return (
                path.name,
                [],
                f"Expected at least 3 images, found {len(frame_paths)}.",
            )

        frame_bltis = compute_blti(frame_paths, output_dir)
        frame_basic = parse_name(path.name)

        res: list[dict[str, Any]] = []

        for frame_idx, frame_blti in frame_bltis.items():
            frame_obj = asdict(frame_basic)
            frame_obj["frame_idx"] = frame_idx

            for i, blti in enumerate(frame_blti):
                frame_obj[f"blti_{i}"] = blti

            res.append(frame_obj)

        return path.name, res, ""

    except Exception as exc:
        return path.name, [], f"{type(exc).__name__}: {exc}"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute BLTI for image sequences.")

    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing image sequences.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("temporal_data"),
        help="Directory for results.",
    )

    parser.add_argument(
        "-fs",
        "--frame-start",
        type=int,
        default=0,
        help="Start frame idx.",
    )

    parser.add_argument(
        "-fe",
        "--frame-end",
        type=int,
        default=-1,
        help="End frame idx.",
    )

    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")


    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    name, rows, error = compute_sequence(args.input_dir, args.output_dir, args.frame_start, args.frame_end)

    if error:
        warn(
            f"{name}: {error}",
            stream=sys.stderr,
        )

        return 0

    output_path = (
        args.output_dir / f"{name}.csv"
    )

    write_to_file(output_path, rows, "csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
