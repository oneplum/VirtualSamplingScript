#!/usr/bin/env python3
"""Check that the sequences produced by temporal.gsc exist and are complete.

Expected names follow temporal.gsc exactly:
  <vol>-<level>-<method>-<lit>-<rate>-<vmethod>-<subdiv>-<motion>
  true sampling   : vmethod n,  rate in {1,10,15}, subdiv 0
  virtual sampling: rate 1, vmethod in {linvs,crvs,hermvs,monhermvs}, subdiv in {1,6,12}
  motion: rot-x, rot-y, rot-diag, trans-x, trans-y, trans-z (z only for non-ortho scenes)
Files inside a folder are  <seqname>-<r>.png  (r = frame index).

usage:
  python check_sequences.py OUTPUT_DIR [--frames 1000] [--level 1] [--scenes ...] [--verify] [--csv check.csv]
Exit code 0 if everything is complete, 1 otherwise.
"""
import argparse, csv, re, sys
from pathlib import Path

ORTHO = {"Head1", "Head2", "Aneurism1", "Aneurism2"}           # no trans-z (script skips z when ortho)
DEFAULT_SCENES = ["Head1", "Head2", "Tree1", "Tree2", "Aneurism1", "Aneurism2", "Sphere", "ML1", "ML2"]
METHODS = ["lin", "quadB", "quadBf", "cubicB", "cubicBf", "cr"]
VMETHODS = ["linvs", "crvs", "hermvs", "monhermvs"]
TRUE_RATES = [1, 5, 10, 15]
SUBDIVS = [1, 3, 6, 12]
MOTIONS = ["rot-x", "rot-y", "rot-diag", "trans-x", "trans-y", "trans-z"]

def expected(scenes, level):
    out = []
    for sc in scenes:
        for lit in ("n", "l"):
            for m in METHODS:
                configs = [(r, "n", 0) for r in TRUE_RATES] + [(1, v, s) for v in VMETHODS for s in SUBDIVS]
                for rate, vm, sd in configs:
                    for mo in MOTIONS:
                        if mo == "trans-z" and sc in ORTHO:
                            continue
                        out.append(f"{sc}-{level}-{m}-{lit}-{rate}-{vm}-{sd}-{mo}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    ap.add_argument("--ext", default="png")
    ap.add_argument("--verify", action="store_true", help="open every image with PIL (slow) to detect truncated files")
    ap.add_argument("--max-print", type=int, default=60, help="max problem lines printed (all are in the CSV)")
    ap.add_argument("--csv", type=Path, default=Path("sequence_check.csv"))
    a = ap.parse_args()

    exp = expected(a.scenes, a.level)
    present = {p.name for p in a.outdir.iterdir() if p.is_dir()} if a.outdir.is_dir() else set()
    pat = re.compile(r"-(\d+)\." + re.escape(a.ext) + r"$")
    rows, bad = [], 0
    for name in exp:
        d = a.outdir / name
        if name not in present:
            rows.append((name, "MISSING_FOLDER", 0, "", "")); bad += 1; continue
        files = [f for f in d.iterdir() if f.suffix.lower() == "." + a.ext]
        idx = sorted(int(m.group(1)) for f in files if (m := pat.search(f.name)))
        note, status = "", "OK"
        if len(files) != a.frames:
            status = "INCOMPLETE" if len(files) < a.frames else "TOO_MANY"
        if idx:
            gaps = sorted(set(range(idx[0], idx[-1] + 1)) - set(idx))
            if gaps:
                status = "GAPS" if status == "OK" else status
                note = f"missing indices {gaps[:10]}{'...' if len(gaps) > 10 else ''}"
        empty = [f.name for f in files if f.stat().st_size == 0]
        if empty:
            status, note = "EMPTY_FILES", f"{len(empty)} zero-byte files (e.g. {empty[0]})"
        if a.verify and status == "OK":
            from PIL import Image
            for f in files:
                try:
                    with Image.open(f) as im: im.load()
                except Exception as e:
                    status, note = "CORRUPT", f"{f.name}: {e}"; break
        if status != "OK": bad += 1
        rows.append((name, status, len(files), idx[0] if idx else "", note))
    extra = sorted(present - set(exp))
    with a.csv.open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h); w.writerow(["sequence", "status", "n_images", "first_index", "note"]); w.writerows(rows)
    print(f"expected sequences: {len(exp)}   present folders: {len(present)}   problems: {bad}")
    from collections import Counter
    print("status counts:", dict(Counter(r[1] for r in rows)))
    shown = 0
    for r in rows:
        if r[1] != "OK":
            if shown < a.max_print: print(f"  {r[1]:15s} {r[2]:5d}  {r[0]}  {r[4]}")
            shown += 1
    if shown > a.max_print: print(f"  ... {shown - a.max_print} more in the CSV")
    if extra:
        print(f"folders not in the expected list ({len(extra)}), e.g.: {extra[:5]}")
    print(f"full table: {a.csv}")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
