#!/usr/bin/env python3
"""Is BLTI dominated by the motion step? Run on ONE sequence directory of PNGs (frame index = number after the last '-').

  python blti_diagnose.py SEQ_DIR [--strides 1 2 4] [--n 20]

Prints
  (1) edge-weighted median optical-flow speed in px/frame (the step size that actually matters for BLTI)
  (2) mean BLTI per band for frame strides 1, 2, 4 (neighbours t-s, t, t+s), using exactly the definitions of blti.py.
Reading: if theta grows roughly in proportion to the stride, the sequence is motion-floor dominated;
if it stays flat, flicker dominates. Flow speed s px/frame together with the table from floor_test.py gives the expected floor.
"""
import argparse, re
from pathlib import Path
import cv2, numpy as np

K, S0, ALPHA, P, GAMMA, EPS = 6, 1.0, 2 ** 0.5, 95.0, 2.0, 1e-6
SCALES = [S0 * ALPHA ** k for k in range(K + 1)]

def lum(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR).astype(np.float64) / 255.0
    return 0.2126 * bgr[..., 2] + 0.7152 * bgr[..., 1] + 0.0722 * bgr[..., 0]

def bands(f):
    G = [cv2.GaussianBlur(f, (0, 0), sigmaX=s) for s in SCALES]
    return [G[k] - G[k + 1] for k in range(K)]

def edge_w(f):
    g = np.hypot(cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3))
    q = np.percentile(g, P)
    return np.zeros_like(g) if q == 0 else np.clip(g / q, 0, 1) ** GAMMA

def wrms(w, X):
    sw = w.sum()
    return float(np.sqrt(np.mean(X ** 2))) if sw < EPS else float(np.sqrt((w * X ** 2).sum() / sw))

def wmedian(x, w):
    o = np.argsort(x); c = np.cumsum(w[o]); return float(x[o][np.searchsorted(c, 0.5 * c[-1])])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--n", type=int, default=20, help="max number of frames t to evaluate"); a = ap.parse_args()
    paths = sorted(Path(a.seq).glob("*.png"), key=lambda p: int(re.findall(r"(\d+)$", p.stem)[0]))
    F = [lum(p) for p in paths]; T = len(F)
    # (1) edge-weighted flow speed between consecutive frames
    sp = []
    for t in np.linspace(1, T - 2, min(10, T - 2)).astype(int):
        a8 = np.clip(F[t] * 255, 0, 255).astype(np.uint8); b8 = np.clip(F[t + 1] * 255, 0, 255).astype(np.uint8)
        fl = cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 4, 21, 3, 5, 1.1, 0)
        w = edge_w(F[t]).ravel(); m = np.hypot(fl[..., 0], fl[..., 1]).ravel()
        if w.sum() > 0: sp.append(wmedian(m, w))
    print(f"{len(paths)} frames; edge-weighted median flow speed: {np.median(sp):.2f} px/frame (range {min(sp):.2f}-{max(sp):.2f})")
    # (2) BLTI vs stride
    B = [bands(f) for f in F]
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    for s in a.strides:
        ts = np.unique(np.linspace(s, T - 1 - s, min(a.n, T - 2 * s)).astype(int)); th = []
        for t in ts:
            w = edge_w(F[t])
            th.append([wrms(w, B[t][k] - .5 * (B[t - s][k] + B[t + s][k])) / max(wrms(w, .5 * (B[t + s][k] - B[t - s][k])), EPS) for k in range(K)])
        print(f"stride {s}: theta per band 0..5 = {np.mean(th, 0)}")

if __name__ == "__main__":
    main()
