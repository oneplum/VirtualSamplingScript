#!/usr/bin/env python3

"""
input: directory path of sequences
output: csv or parquet
sequence name format: volname-level-method-lighting-rate-vmethod-subdiv-transforms-axis
row column: dataset_name,level,method,lighting_enabled,true_samples,virtual_sampling_method,virtual_samples,trans_type,trans_axis,frame_idx,blti_0,blti_1,blti_2,blti_3,blti_4,blti_5
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

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
    normalized = np.clip(gradient / max(q_p, BLTI_EPS), 0.0, 1.0)

    return normalized**BLTI_GAMMA


def _weighted_rms(weight: np.ndarray, X: np.ndarray) -> float:
    sum_weight = np.sum(weight)
    if sum_weight < BLTI_EPS:
        return float(np.sqrt(np.mean(X**2)))

    return float(np.sqrt(np.sum(weight * X**2) / sum_weight))


def _compute_bands(frame: np.ndarray) -> list[np.ndarray]:
    bands: list[np.ndarray] = []

    prev = cv2.GaussianBlur(frame, (0, 0), sigmaX=GAUSSIAN_SCALES[0])

    for sigma in GAUSSIAN_SCALES[1:]:
        curr = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma)
        bands.append(prev - curr)
        prev = curr

    return bands


def compute_blti(paths: list[Path]) -> dict[int, list[float]]:
    if len(paths) < 3:
        return {}

    prev_image = _luminance_image(paths[0])
    curr_image = _luminance_image(paths[1])

    prev_bands = _compute_bands(prev_image)
    curr_bands = _compute_bands(curr_image)

    bltis: dict[int, list[float]] = {}
    for t in range(1, len(paths) - 1):
        next_image = _luminance_image(paths[t + 1])
        next_bands = _compute_bands(next_image)

        weight = _edge_weights(curr_image)

        blti_t: list[float] = []
        for k in range(BLTI_K):
            band_prev = prev_bands[k]
            band_curr = curr_bands[k]
            band_next = next_bands[k]

            R_t = band_curr - 0.5 * (band_prev + band_next)
            S_t = 0.5 * (band_next - band_prev)

            blti_t_k = _weighted_rms(weight, R_t) / max(
                _weighted_rms(weight, S_t), BLTI_EPS
            )
            blti_t.append(blti_t_k)

        bltis[t] = blti_t

        prev_image = curr_image
        curr_image = next_image

        prev_bands = curr_bands
        curr_bands = next_bands

    return bltis


def compute_sequence(path: Path) -> tuple[str, list[dict[str, Any]], str]:
    try:
        frame_paths = sorted(
            path.glob(PATTERN), key=lambda x: int(x.stem.rsplit("-", 1)[-1])
        )

        if len(frame_paths) < 3:
            return (
                path.name,
                [],
                f"Expected at least 3 images, found {len(frame_paths)}.",
            )

        frame_bltis = compute_blti(frame_paths)
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
        "--output-format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Output file format.",
    )

    parser.add_argument(
        "--multi-process",
        action="store_true",
        help="Enable multiprocessing",
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=8,
        help="Number of parallel sequence workers.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of sequences per output file.",
    )

    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    workers = min(args.workers, os.cpu_count() or 1)

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sequence_paths = sorted(path for path in args.input_dir.iterdir() if path.is_dir())

    total = len(sequence_paths)

    print(f"Found {total} sequences.")
    if args.multi_process:
        print(f"Workers: {workers}")
    print(f"Batch size: {args.batch_size} sequences")

    chunksize = max(1, (os.cpu_count() or 1) // 2)

    batch_rows: list[dict[str, Any]] = []
    batch_sequences = 0
    part_idx = 0
    processed = 0
    failed = 0
    start = time.perf_counter()
    if args.multi_process:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(
            compute_sequence,
            sequence_paths,
            chunksize=chunksize,
        )
    else:
        executor = None
        results = (compute_sequence(path) for path in sequence_paths)

    try:
        for name, rows, error in results:
            processed += 1

            if error:
                failed += 1
                warn(
                    f"{name}: {error}",
                    stream=sys.stderr,
                )
                continue

            # print(len(rows))
            batch_sequences += 1
            batch_rows.extend(rows)

            if (
                batch_sequences >= args.batch_size or processed == total
            ) and batch_rows:
                output_path = (
                    args.output_dir / f"part-{part_idx:05d}.{args.output_format}"
                )

                write_to_file(output_path, batch_rows, args.output_format)

                print(
                    f"[{processed}/{total}] "
                    f"Saved {output_path.name} "
                    f"({len(batch_rows):,} rows)"
                )

                batch_rows.clear()
                batch_sequences = 0
                part_idx += 1

    finally:
        if executor is not None:
            executor.shutdown()
    elapsed = time.perf_counter() - start

    print(f"{args.output_format}: time={elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
