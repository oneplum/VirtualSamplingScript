
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==========================================
# 1. Your DoG parameters
# ==========================================

BLTI_K = 6
BLTI_SIGMA0 = 0.5
BLTI_ALPHA = 2**0.5

GAUSSIAN_SCALES = tuple(
    BLTI_SIGMA0 * (BLTI_ALPHA**k)
    for k in range(BLTI_K + 1)
)

# ==========================================
# 2. Your original DoG implementation
# ==========================================

def compute_bands(frame: np.ndarray) -> list[np.ndarray]:
    bands = []

    curr = cv2.GaussianBlur(
        frame, (0, 0), sigmaX=GAUSSIAN_SCALES[0]
    )

    for sigma in GAUSSIAN_SCALES[1:]:
        nxt = cv2.GaussianBlur(
            frame, (0, 0), sigmaX=sigma
        )

        bands.append(curr - nxt)
        curr = nxt

    return bands


# ==========================================
# 3. Generate sinusoidal test image
# ==========================================

def generate_grating(
    period: float,
    size: int = 512
) -> np.ndarray:

    x = np.arange(size, dtype=np.float32)

    # Horizontal sinusoidal pattern
    signal = np.sin(2 * np.pi * x / period)

    image = np.tile(signal, (size, 1))

    return image.astype(np.float32)


# ==========================================
# 4. Measure response of each DoG band
# ==========================================

def measure_response(
    period: float,
    size: int = 512
) -> np.ndarray:

    image = generate_grating(period, size)

    bands = compute_bands(image)

    responses = []

    # Ignore borders to reduce boundary effects
    margin = 32

    for band in bands:

        center = band[
            margin:-margin,
            margin:-margin
        ]

        # RMS response
        rms = np.sqrt(np.mean(center**2))

        responses.append(rms)

    return np.array(responses)


# ==========================================
# 5. Test different spatial periods
# ==========================================

periods = np.linspace(2.0, 40.0, 200)

all_responses = []

for period in periods:

    response = measure_response(period)

    all_responses.append(response)

all_responses = np.array(all_responses)

# Shape: [periods, bands]


# ==========================================
# 6. Plot frequency response
# ==========================================

plt.figure(figsize=(11, 6))

for k in range(BLTI_K):

    plt.plot(
        periods,
        all_responses[:, k],
        label=f"Band {k}"
    )

plt.xlabel("Spatial Period (pixels)")
plt.ylabel("RMS Response")
plt.title("DoG Spatial Frequency Response")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ==========================================
# 7. Find peak wavelength for each band
# ==========================================

print("\nPeak Spatial Period of Each Band")
print("--------------------------------")

for k in range(BLTI_K):

    idx = np.argmax(all_responses[:, k])

    peak_period = periods[idx]
    peak_response = all_responses[idx, k]

    print(
        f"Band {k}: "
        f"{peak_period:.2f} px "
        f"(RMS = {peak_response:.5f})"
    )


# ==========================================
# 8. Compare selected spatial periods
# ==========================================

test_periods = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]

print("\nBand Responses for Selected Periods")
print("-----------------------------------")

print(
    "Period | " +
    " | ".join(f"B{k}" for k in range(BLTI_K))
)

for period in test_periods:

    response = measure_response(period)

    print(
        f"{period:6.1f} | " +
        " | ".join(f"{v:.4f}" for v in response)
    )
