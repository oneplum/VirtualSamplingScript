#!/usr/bin/env python3
"""
Turn the stage0 renders into a calibration: how many screen pixels per nominal unit (degree for rotations, scene
units for translations) the edge-weighted image content moves, for every scene and motion.

  python calibrate.py out_stage0 stage0_manifest.csv [--out calibration.json]

Speed = edge-weighted median optical-flow magnitude (the weighting BLTI itself uses: Sobel gradient, p95, gamma 2)
between consecutive frames of each 3-frame stage0 sequence. The result is px_per_unit; stage1-3 use
step = target_px_per_frame / px_per_unit.
Reliable flow needs roughly 0.5-6 px/frame; outside that range the row is flagged and --cal-rot/--cal-trans of
make_temporal_gsc.py stage0 should be changed and stage0 rendered again (it is tiny).
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def lum(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR).astype(np.float64) / 255.0
    return 0.2126 * bgr[..., 2] + 0.7152 * bgr[..., 1] + 0.0722 * bgr[..., 0]


def edge_w(f, p=95.0, gamma=2.0):
    g = np.hypot(cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3))
    q = np.percentile(g, p)
    return np.zeros_like(g) if q == 0 else np.clip(g / q, 0, 1) ** gamma


def wquant(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(c, q * c[-1]), len(x) - 1)])


def pair_speed(a, b):
    a8 = np.clip(a * 255, 0, 255).astype(np.uint8)
    b8 = np.clip(b * 255, 0, 255).astype(np.uint8)
    fl = cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 4, 21, 3, 5, 1.1, 0)
    w = edge_w(a).ravel()
    m = np.hypot(fl[..., 0], fl[..., 1]).ravel()
    if w.sum() == 0:
        return float('nan'), float('nan')
    return wquant(m, w, 0.5), wquant(m, w, 0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('manifest')
    ap.add_argument('--out', default='calibration.json')
    a = ap.parse_args()
    calib, report = {}, []
    for r in csv.DictReader(open(a.manifest)):
        d = Path(a.root) / r['seq']
        F = [lum(d / f"{r['seq']}-{k}.png") for k in range(int(r['frames']))]
        ps = [pair_speed(F[k], F[k + 1]) for k in range(len(F) - 1)]
        med, p90 = float(np.nanmean([p[0] for p in ps])), float(np.nanmean([p[1] for p in ps]))
        step = float(r['step'])
        c = med / step
        calib.setdefault(r['scene'], {})[r['motion']] = c
        flag = '' if 0.5 <= med <= 6 else '  <-- outside 0.5-6 px: re-render stage0 with another --cal-rot/--cal-trans'
        report.append(f"{r['scene']:<10} {r['motion']:<7} step {step:g} {r['step_unit']:<4} median {med:6.2f} px/frame"
                      f"  p90 {p90:6.2f}  => {c:10.3f} px/unit{flag}")
    json.dump(calib, open(a.out, 'w'), indent=1)
    print('\n'.join(report))
    print(f"wrote {a.out}")


if __name__ == '__main__':
    main()
