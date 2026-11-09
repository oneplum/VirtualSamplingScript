#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import blti as b
import data as d

def show_img(data: list[tuple[np.ndarray, str, str | None, tuple | None]], title: str, path: Path) -> int:
    c = len(data)
    col = min(3, c)
    row = math.ceil(c / col)
    fig, axes = plt.subplots(row, col, figsize=(10,10), squeeze=False)

    for ax, (px, label, cmap, conf) in zip(axes.flat, data):
        if cmap == "hot":
            im = ax.imshow(px, cmap='hot')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.5)
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


def _seq_col() -> d.pl.Expr:
    return d.pl.concat_str(
        [
            d.pl.when(c == "lighting_enabled" and d.pl.col(c) == True)
            .then(d.pl.lit("l"))
            .when(c == "lighting_enabled" and d.pl.col(c) == False)
            .then(d.pl.lit("n"))
            .when(c == "virtual_sampling_method" and d.pl.col(c) == "none")
            .then(d.pl.lit("n"))
            .otherwise(d.pl.col(c).cast(d.pl.String))
            for c in d.SEQ_COLS
        ],
        separator="-"
    ).alias("seq")


def check_data(lf: d.pl.LazyFrame, path: Path) -> int:
    # step 1: sequences count
    n_seq = lf.group_by(d.SEQ_COLS).len().select(d.pl.len()).collect().item()
    print(f"{n_seq} sequences")

    # step 2: sequencies do not include 98 results
    sdf = lf.group_by(d.SEQ_COLS).len().filter(d.pl.col("len") != 98).with_columns(_seq_col()).select([d.pl.col("seq"), d.pl.col("len")]).collect()
    if len(sdf) > 0:
        print(f"found {len(sdf)} unexcepted sequences.")
        sdf.write_csv(path / "unexcepted_seq.csv")
    else:
        print("all sequences valid")

    return 0


def statistic_data(lf: d.pl.LazyFrame, path: Path) -> int: 
    d.pl.concat([
        lf.select([
            d.pl.col(c).mean().alias("mean"),
            d.pl.col(c).max().alias("max"),
            d.pl.col(c).min().alias("min"),
            d.pl.col(c).quantile(0.10).alias("p10"),
            d.pl.col(c).quantile(0.99).alias("p99"),
            d.pl.col(c).quantile(0.999).alias("p999"),
            d.pl.lit(c).alias("band_idx"),
        ])
        for c in d.METRIC_COLS
    ]).collect().write_csv(path / "statistic.csv")

    return 0


def analysis_data(lf: d.pl.LazyFrame, path: Path, dir: Path | None = None, k: int | None = None, score: float | None = None) -> int:
    if k is None:
        k = 0
        bands = d.METRIC_COLS
    else:
        bands = [d.METRIC_COLS[k]]

    for metric_col in bands:
        m_path = path / metric_col
        m_path.mkdir(
            parents=True,
            exist_ok=True,
        )
        f_cols = ["dataset_name", "level", "trans_type", "trans_axis", "lighting_enabled"]
        flf = (
            lf.filter(d.pl.col(metric_col) > (d.pl.col(metric_col).quantile(0.999) if score is None else score))
            .sort(metric_col, descending=True)
            .with_columns(_seq_col())
            .group_by(f_cols, maintain_order=True)
            .agg(
                d.pl.col("seq").first(),
                d.pl.col("frame_idx").first(),
                d.pl.col(metric_col).first()
            )
        )

        flf.select(["seq", "frame_idx", metric_col]).collect().write_csv(m_path / f"{metric_col}_p999.csv")

        if dir is not None:
            for item in flf.collect().iter_rows(named=True):
                output_dir = m_path / item["seq"]
                output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                for _p in output_dir.glob("*.png"):
                    _p.unlink()
                name, row, error = compute_sequence(dir / item["seq"], output_dir, item["frame_idx"], k)
                if error:
                    print(f"{name}: {error}")
                    return 0

                # find best
                filter_by = {col: row[col] for col in f_cols + ["frame_idx"]}
                bseq, bfidx, bb = d.filter_data(lf, filter_by).sort(d.pl.col(metric_col)).head(1).with_columns(_seq_col()).select(["seq", "frame_idx", metric_col]).collect().row(0)
                compute_sequence(dir / bseq, output_dir, bfidx, k)

                if row[metric_col] == item[metric_col]:
                    print("pass")
                else:
                    print(item[metric_col])
                    print(row)

        k += 1

    return 0


def split_parquet_file(lf: d.pl.LazyFrame, output_dir: Path) -> int:
    print(d.load_col_data(lf, ["dataset_name", "level", "lighting_enabled"]))

    df = lf.collect()
    groups = df.partition_by(["dataset_name", "level", "lighting_enabled"], as_dict=True)

    for (ds, level, lit), group in groups.items():
        print(len(group))
        group.write_parquet(output_dir / f"{ds}-{str(level)}-{"l" if lit else "n"}.parquet")

    return 0


# def diff_data(path: Path) -> int:
#     for p in path.glob("*.csv"):
#         df1 = d.pl.read_csv(p)

#         p2 = p.parent.parent / f"{p.parent.name}a" / p.name
#         df2 = d.pl.read_csv(p2)

#         if not df1.equals(df2):
#             diff1 = df1.join(df2, on=df1.columns, how="anti")
#             print(p.name)
#             print(diff1)

#     print("done")

#     return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check exceptional data.")

    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing blti data.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("check"),
        help="Directory for results.",
    )

    parser.add_argument(
        "--img-dir",
        type=Path,
        help="Directory containing image sequences.",
    )

    parser.add_argument(
        "--action",
        type=int,
        default=0,
        help="0: check sequences whether valid, 1: blti statistic data 2: analyse unexcepted data 3: split data",
    )

    parser.add_argument(
        "-k",
        type=int,
        help="Band idx",
    )

    parser.add_argument(
        "--score",
        type=float,
        help="Blti score",
    )

    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    output_dir = args.output_dir #/ "check" / args.input_dir.name
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # for _p in output_dir.glob("*.csv"):
    #     _p.unlink()

    lf = d.load_data(args.input_dir)
    if lf is None:
        print("No valid data files found.")
        return 1

    match args.action:
        case 0:
            check_data(lf, output_dir)
        case 1:
            statistic_data(lf, output_dir)
        case 2:
            analysis_data(lf, output_dir, args.img_dir, args.k, args.score)
        case 3:
            split_parquet_file(lf, output_dir)
        case _:
            check_data(lf, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
