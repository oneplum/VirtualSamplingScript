import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# 1. Configuration
# ==========================================

ap = argparse.ArgumentParser()
ap.add_argument("images_dir", type=Path)
a = ap.parse_args()

DEFAULT_SCENES = ["Head1", "Head2", "Tree1", "Tree2", "Aneurism1", "Aneurism2", "Sphere", "ML1", "ML2"]

DATASET_DIR = a.images_dir

OUTPUT_CSV = "all_sequences_spatial_frequency.csv"

NUM_SAMPLES = 1

NUM_BINS = 300

FREQUENCY_RANGES = {
    "Very_Low_50_200px": (1/200, 1/50),
    "Low_20_50px":       (1/50, 1/20),
    "Medium_5_20px":     (1/20, 1/5),
    "High_2_5px":        (1/5, 1/2),
}

# ==========================================
# 2. Analyze one image
# ==========================================

def analyze_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32) / 255.0

    H, W = gray.shape

    # Remove mean
    gray -= np.mean(gray)

    # Hann window
    window = np.outer(
        np.hanning(H),
        np.hanning(W)
    )

    gray *= window

    # 2D FFT
    fft = np.fft.fftshift(
        np.fft.fft2(gray)
    )

    power = np.abs(fft) ** 2

    fx = np.fft.fftshift(np.fft.fftfreq(W))
    fy = np.fft.fftshift(np.fft.fftfreq(H))

    FX, FY = np.meshgrid(fx, fy)

    radial_freq = np.sqrt(FX**2 + FY**2)

    # Radial spectrum
    max_freq = np.sqrt(0.5**2 + 0.5**2)

    bins = np.linspace(
        0,
        max_freq,
        NUM_BINS + 1
    )

    freq_centers = (bins[:-1] + bins[1:]) / 2

    radial_energy = np.zeros(NUM_BINS)

    for i in range(NUM_BINS):

        mask = (
            (radial_freq >= bins[i]) &
            (radial_freq < bins[i + 1])
        )

        if np.any(mask):
            radial_energy[i] = np.mean(power[mask])

    # Dominant frequency
    valid = freq_centers > 0.005

    valid_freqs = freq_centers[valid]
    valid_energy = radial_energy[valid]

    peak_idx = np.argmax(valid_energy)

    dominant_freq = valid_freqs[peak_idx]

    dominant_period = 1.0 / dominant_freq

    # Total power
    total_power = np.sum(power)

    result = {
        "Width": W,
        "Height": H,
        "Dominant_Frequency": dominant_freq,
        "Dominant_Period_px": dominant_period,
    }

    # Frequency range statistics
    for name, (f_min, f_max) in FREQUENCY_RANGES.items():

        mask = (
            (radial_freq >= f_min) &
            (radial_freq < f_max)
        )

        range_power = np.sum(power[mask])

        mean_power = (
            np.mean(power[mask])
            if np.any(mask) else 0
        )

        proportion = (
            range_power / total_power * 100
            if total_power > 0 else 0
        )

        radial_mask = (
            (freq_centers >= f_min) &
            (freq_centers < f_max)
        )

        selected_freqs = freq_centers[radial_mask]

        selected_energy = radial_energy[radial_mask]

        if len(selected_energy) > 0:

            idx = np.argmax(selected_energy)

            peak_period = 1.0 / selected_freqs[idx]

        else:
            peak_period = np.nan

        result[f"{name}_Peak_Period_px"] = peak_period

        result[f"{name}_Mean_Power"] = mean_power

        result[f"{name}_Proportion_%"] = proportion

    return result


# ==========================================
# 3. Uniformly sample one sequence
# ==========================================

def sample_sequence(sequence_dir):

    image_paths = sorted(sequence_dir.glob("*.png"), key=lambda x: int(x.stem.rsplit("-", 1)[-1]))

    total_frames = len(image_paths)

    if total_frames == 0:
        return []

    sample_count = min(NUM_SAMPLES, total_frames)

    # Uniformly sample across the entire sequence
    indices = np.linspace(
        0,
        total_frames - 1,
        sample_count,
        dtype=int
    )

    # Remove potential duplicate indices
    indices = np.unique(indices)

    return [
        (idx, image_paths[idx])
        for idx in indices
    ]


# ==========================================
# 4. Analyze all sequences
# ==========================================

all_results = []

sequence_dirs = []
for scene in DEFAULT_SCENES:
    sequence_dirs.append(a.images_dir / f"{scene}-1-quadBf-n-15-n-0-rot-y")
    # sequence_dirs.append(a.images_dir / f"{scene}-1-quadBf-n-15-n-0-rot-x")
    # sequence_dirs.append(a.images_dir / f"{scene}-1-quadBf-n-15-n-0-rot-diag")
    # sequence_dirs.append(a.images_dir / f"{scene}-1-quadBf-n-15-n-0-trans-y")
    # sequence_dirs.append(a.images_dir / f"{scene}-1-quadBf-n-15-n-0-trans-x")
    # sequence_dirs.append(a.images_dir / f"{scene}-1-quadBf-n-15-n-0-trans-z")
    # sequence_dirs.append(a.images_dir / f"{scene}-1-cubicBf-l-15-n-0-rot-y")
    # sequence_dirs.append(a.images_dir / f"{scene}-1-cubicBf-l-15-n-0-rot-x")
    # sequence_dirs.append(a.images_dir / f"{scene}-1-cubicBf-l-15-n-0-rot-diag")
    # sequence_dirs.append(a.images_dir / f"{scene}-1-cubicBf-l-15-n-0-trans-y")
    # sequence_dirs.append(a.images_dir / f"{scene}-1-cubicBf-l-15-n-0-trans-x")
    # sequence_dirs.append(a.images_dir / f"{scene}-1-cubicBf-l-15-n-0-trans-z")


for sequence_dir in sequence_dirs:

    sequence_name = sequence_dir.name

    print(f"\nAnalyzing sequence: {sequence_name}")

    sampled_images = sample_sequence(sequence_dir)

    if not sampled_images:
        print("No images found. Skipping.")
        continue

    sequence_results = []

    total_frames = len([
        p for p in sequence_dir.iterdir()
        if p.is_file() and p.suffix.lower() == "png"
    ])

    print(
        f"Total frames: {total_frames}, "
        f"Sampled frames: {len(sampled_images)}"
    )

    for sample_number, (frame_idx, image_path) in enumerate(
        sampled_images
    ):

        print(
            f"  Processing {sample_number + 1}/"
            f"{len(sampled_images)}: {image_path.name}"
        )

        result = analyze_image(image_path)

        if result is None:
            continue

        result["Sequence"] = sequence_name
        result["Frame"] = image_path.name
        result["Frame_Index"] = frame_idx
        result["Sample_Number"] = sample_number + 1

        sequence_results.append(result)

    all_results.extend(sequence_results)

    # ======================================
    # 5. Sequence summary
    # ======================================

    if sequence_results:

        df_sequence = pd.DataFrame(sequence_results)

        summary = {
            "Sequence": sequence_name,
            "Frame": "SEQUENCE_SUMMARY",
            "Frame_Index": -1,
            "Sample_Number": len(sequence_results),
            "Width": df_sequence["Width"].iloc[0],
            "Height": df_sequence["Height"].iloc[0],
        }

        numeric_columns = [
            col for col in df_sequence.columns
            if col not in [
                "Sequence",
                "Frame",
                "Frame_Index",
                "Sample_Number",
                "Width",
                "Height"
            ]
        ]

        for col in numeric_columns:

            summary[f"{col}_Mean"] = df_sequence[col].mean()

            summary[f"{col}_Min"] = df_sequence[col].min()

            summary[f"{col}_Max"] = df_sequence[col].max()

        all_results.append(summary)


# ==========================================
# 6. Save all results into ONE CSV
# ==========================================

if not all_results:

    raise RuntimeError("No images were successfully analyzed.")

df = pd.DataFrame(all_results)

df.to_csv(
    OUTPUT_CSV,
    index=False,
    float_format="%.6f"
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(f"Sequences analyzed: {len(sequence_dirs)}")
print(f"CSV output: {OUTPUT_CSV}")
print(f"Total result rows: {len(df)}")
