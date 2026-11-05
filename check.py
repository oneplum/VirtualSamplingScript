#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import blti as b
import data as d

def show_img(data: list[tuple[np.ndarray, str, str | None, tuple | None]], title: str, path: Path) -> int:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )
    c = len(data)
    col = min(3, c)
    row = math.ceil(c / col)
    fig, axes = plt.subplots(row, col, figsize=(10,10), squeeze=False)

    for ax, (px, label, cmap, conf) in zip(axes.flat, data):
        if cmap == "hot":
            im = ax.imshow(px, cmap='hot')
            fig.colorbar(im, ax=ax, shrink=0.8)
        elif cmap == "gray":
            if not conf:
                conf = (0, 1)
            ax.imshow(px, cmap='gray', vmin=conf[0], vmax=conf[1])
        else:
            ax.imshow(px)

        ax.set_title(label)
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(path / title, bbox_inches='tight', dpi=220)

    plt.close(fig)

    return 0


def compute_blti(paths: list[Path], output_dir: Path, t: int = 1, k: int = 0) -> tuple[int, int, float]:
    if len(paths) < 3:
        return {}

    prev_image, prev_bands = b._load_and_blur(paths[t-1])
    curr_image, curr_bands = b._load_and_blur(paths[t])
    next_image, next_bands = b._load_and_blur(paths[t+1])

    weight = b._edge_weights(curr_image)

    band_prev = prev_bands[k]
    band_curr = curr_bands[k]
    band_next = next_bands[k]

    R_t = band_curr - 0.5 * (band_prev + band_next)
    S_t = 0.5 * (band_next - band_prev)

    show_img([
        (b.load_rgb_image(paths[t-1]) / 255.0, "Previous Image", None, None),
        (b.load_rgb_image(paths[t]) / 255.0, "Current Image", None, None),
        (b.load_rgb_image(paths[t+1]) / 255.0, "Next Image", None, None),
        (prev_image, "Previous luminance Image", "gray", None),
        (curr_image, "Current luminance Image", "gray", None),
        (next_image, "Next luminance Image", "gray", None),
    ], f"img_{paths[t].name}", output_dir)

    show_img([
        (weight, "Edge Weight", "gray", None),
    ], f"weight_{paths[t].name}", output_dir)

    show_img([
        (band_prev, f"Previous Band {k}", "gray", (np.min(band_prev), np.max(band_prev))),
        (band_curr, f"Current Band {k}", "gray", (np.min(band_curr), np.max(band_curr))),
        (band_next, f"Next Band {k}", "gray", (np.min(band_next), np.max(band_next))),
    ], f"band_{k}_{paths[t].name}", output_dir)

    show_img([ 
        (weight * R_t ** 2, "weight * R_t ** 2", "hot", None), 
        (weight * S_t ** 2, "weight * S_t ** 2", "hot", None)
    ], f"hot_{k}_{paths[t].name}", output_dir)

    blti_t_k = b._weighted_rms(weight, R_t) / max(
        b._weighted_rms(weight, S_t), b.BLTI_EPS
    )

    return (t, k, blti_t_k)


def compute_sequence(path: Path, output_dir: Path, t: int = 1, k: int = 0) -> tuple[str, dict, str]:
    frame_paths = sorted(
        path.glob(b.PATTERN), key=lambda x: int(x.stem.rsplit("-", 1)[-1])
    )

    if len(frame_paths) != 100:
        return (
            path.name,
            [],
            f"Expected 100 images, found {len(frame_paths)}.",
        )

    frame_bltis = compute_blti(frame_paths, output_dir, t, k)
    frame_basic = b.parse_name(path.name)

    frame_obj = b.asdict(frame_basic)
    frame_obj["frame_idx"] = frame_bltis[0]
    frame_obj[f"blti_{frame_bltis[1]}"] = frame_bltis[2]

    return path.name, frame_obj, ""


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check exceptional data.")

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
        default=Path("check"),
        help="Directory for results.",
    )

    parser.add_argument(
        "-t",
        type=int,
        default=1,
        help="Frame idx",
    )

    parser.add_argument(
        "-k",
        type=int,
        default=0,
        help="Band idx",
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

    name, row, error = compute_sequence(args.input_dir, args.output_dir, args.t, args.k)

    if error:
        print(f"{name}: {error}")

        return 0

    print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
