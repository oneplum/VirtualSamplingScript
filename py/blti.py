#!/usr/bin/env python3

import argparse
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.ndimage import sobel
import sys
from dataclasses import dataclass, fields
from typing import Any

from common import FrameBasic, FrameSeq, load_frames_seqs, warn, load_rgb_image, write_csv_rows

PATTERN = "*.png"
BLTI_K = 6
BLTI_SIGMA0 = 1.0
BLTI_ALPHA = 2 ** 0.5
BLTI_P = 95.0
BLTI_GAMMA = 2.0
BLTI_EPS_DENOM = 1e-6

@dataclass
class Bandstat:
    mean: float
    median: float
    p10: float
    p90: float

@dataclass
class BltiFrame(FrameBasic):
    number: int
    frame_bltis: list[float]

    def to_csv_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for field in fields(self):
            if field.name == "frame_bltis":
                for i, blti in enumerate(self.frame_bltis):
                    row[f"blti_{i}"] = blti
                continue
            row[field.name] = getattr(self, field.name)
        return row

#     @classmethod
#     def init_from_row(self, row: dict[str, str]):
#         for field in fields(self):
#             if field.name == "frame_bltis":
#                 bands: list[float] = []
#                 for k, v in row.items():
#                     if k.startswith("blti_"):
#                         bands.append(float(v))
#                 self.bands = bands
#                 continue
#             value = row.get(field.name)
#             if value is not None:
#                 setattr(self, field.name, covert_value(value, field.type))
#         return self

@dataclass
class BltiSeq(FrameBasic):
    bandstats: list[Bandstat]
    seq_bltis: list[BltiFrame]

    def to_csv_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for field in fields(self):
            if field.name == "seq_bltis":
                continue
            if field.name == "bandstats":
                for i, bandstatObj in enumerate(self.bandstats):
                    for bandstatField in fields(bandstatObj):
                        row[f"blti_{i}_{bandstatField.name}"] = getattr(bandstatObj, bandstatField.name)
                continue
            row[field.name] = getattr(self, field.name)
        return row

def luminance_image(path: Path) -> np.ndarray:
    rgb = load_rgb_image(path) / 255.0
    limage = (
        0.2126 * rgb[..., 0] +
        0.7152 * rgb[..., 1] +
        0.0722 * rgb[..., 2]
    )
    return limage

def gaussian_scales(K: int = BLTI_K) -> list[float]:
    if K < 0:
        K = BLTI_K
    sigmas = [BLTI_SIGMA0 * (BLTI_ALPHA ** k) for k in range(K + 1)]
    return sigmas

def gradient_magnitude(frame: np.ndarray) -> np.ndarray:
    G_x = sobel(frame, axis=1)
    G_y = sobel(frame, axis=0)
    return np.sqrt(G_x ** 2 + G_y ** 2) #.astype(np.float64)

def edge_weights(frame: np.ndarray) -> np.ndarray:
    g_t = gradient_magnitude(frame)
    q_p = np.percentile(g_t, BLTI_P)
    g_sim = np.clip(g_t / (q_p + BLTI_EPS_DENOM), 0.0, 1.0)
    return g_sim ** BLTI_GAMMA #.astype(np.float64)

def weighted_rms(weight: np.ndarray, X: np.ndarray) -> np.ndarray:
    sum_weight = np.sum(weight)
    if sum_weight < BLTI_EPS_DENOM:
        return np.sqrt(np.mean(X ** 2))
    else:
        return np.sqrt(np.sum(weight * X ** 2) / sum_weight)

def compute_blti(paths: list[Path], K: int = BLTI_K) -> dict[int, np.ndarray] | None:
    I = [luminance_image(path) for path in paths]
    if len(I) < 3:
        return None

    B: list[np.ndarray] = []
    for I_t in I:
        G_t = [gaussian_filter(I_t, sigma) for sigma in gaussian_scales(K)]
        B_t = [G_t[k] - G_t[k+1] for k in range(K)]
        B.append(B_t)

    bltis: dict[int, np.ndarray] = {}
    for t in range(1, len(I) - 1):
        weight = edge_weights(I[t])
        blti_t: np.ndarray = []
        for k in range(K):
            band_prev = B[t-1][k]
            band_curr = B[t][k]
            band_next = B[t+1][k]
            R_t = band_curr - 0.5 * (band_prev + band_next)
            S_t = 0.5 * (band_next - band_prev)

            blti_t_k = weighted_rms(weight, R_t) / max(weighted_rms(weight, S_t), BLTI_EPS_DENOM)
            blti_t.append(blti_t_k)
        if len(blti_t) < K:
            continue
        bltis[t] = blti_t

    return bltis

def compute_seq_blti(frame_seq: FrameSeq, K: int = BLTI_K, *, warning_stream) -> BltiSeq | None:
    frame_seq.frame_images.sort(key=lambda row: row.number)
    if len(frame_seq.frame_images) < 3:
        warn(
            f"Expected at least 3 images in {frame_seq.label}, but found {len(frame_seq.frame_images)}.",
            warning_stream
        )
        return None

    bltis = compute_blti([image.path for image in frame_seq.frame_images])
    if bltis is None:
        warn(
            f"Compute BLTI error at position {frame_seq.label}.",
            stream=warning_stream
        )
        return None

    band_bltis: list[np.ndarray] = []
    blti_frames: list[BltiFrame] = []
    for image in frame_seq.frame_images:
        if image.number not in bltis:
            warn(
                f"No BLTI result found at position {image.number}.",
                stream=warning_stream
            )
            continue
        frame_bltis = bltis.get(image.number)
        blti_frames.append(BltiFrame.from_obj(image, {"frame_bltis": frame_bltis}))
        band_bltis.append(frame_bltis)

    return BltiSeq.from_obj(frame_seq, {"seq_bltis":blti_frames, "bandstats": [Bandstat(
                        mean=mean,
                        median=median,
                        p10=p10,
                        p90=p90,
                    ) for mean, median, p10, p90 in zip(
                        np.mean(band_bltis, axis=0),
                        np.median(band_bltis, axis=0),
                        np.percentile(band_bltis, 10, axis=0),
                        np.percentile(band_bltis, 90, axis=0)
                    )]})

def save_blti(path: Path, blti: list[BltiSeq] | list[BltiFrame]) -> None:
    write_csv_rows(path, [b.to_csv_row() for b in blti])

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="compute blti of one sequence"
    )
    parser.add_argument(
        "-i",
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing the generated screenshot images.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory for CSV file，default：images directory",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Reuse existing BLTI results instead of recomputing them.",
    )
    parser.add_argument(
        "--one-seq",
        action="store_true",
        help="only compute one sequence",
    )
    return parser

def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = args.output_dir
    else:
        output_dir = args.images_dir

    frames_seqs = load_frames_seqs(args.images_dir, args.one_seq, warning_stream=sys.stderr)

    if not frames_seqs:
        warn(
            f"No valid frame sequences found.",
            stream=sys.stderr
        )
        return 0

    K = BLTI_K

    seq_bltis: list[BltiSeq] = []
    for frame_seq in frames_seqs:
        seq_blti = compute_seq_blti(frame_seq, K, warning_stream=sys.stderr)
        if seq_blti is None:
            warn(
                f"No Blti result in {frame_seq.label}",
                stream=sys.stderr
            )
            continue

        if args.save_csv and seq_blti.seq_bltis:
            save_blti(output_dir / f"temporal_blti_{frame_seq.label}.csv", seq_blti.seq_bltis)
            print(f"Saved frame-level BLTI results to temporal_blti_{frame_seq.label}.csv")

        seq_bltis.append(seq_blti)

    if args.save_csv and seq_bltis:
        save_blti(output_dir / f"temporal_blti.csv", seq_bltis)
        print(f"Saved sequence-level BLTI results to temporal_blti.csv")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())