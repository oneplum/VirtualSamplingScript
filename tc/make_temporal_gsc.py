#!/usr/bin/env python3
"""
Generate state-explicit volume-renderer scripts (.gsc) for the temporal-coherence study.

Design rules (why this replaces the hand-written temporal.gsc)
  1. Step size is defined in SCREEN SPACE (target px/frame, edge-weighted flow speed), converted per scene and
     motion to a nominal step through a calibration measured on real renders (stage0 -> calibrate.py).
  2. Every scene/lighting group re-emits volume, background and transfer function explicitly, and every
     sequence re-emits level, rate, subdiv, method and camera. No state can leak between loop iterations.
  3. Every test sequence has a reference sequence on the same camera path (level 0, rate 8, as in quality.gsc;
     analytic methods 78/79 for analytic volumes), so PSNR and BLTI come from the same frames.
  4. Test levels are 1 and 2 (the paper's quality script also tests only downsampled data against level 0).
  5. Windows are small and centred on the paper's chosen view; z motion only for perspective scenes
     (the camera model stays identical to the scene's other motions).

Stages
  stage0   calibration: 3 frames per (scene, motion) at a fixed small nominal step
  stage1   step ladder: speeds 2/1/0.5/0.25 px/frame, lin@1 and lin@15, plus reference (+ determinism duplicate)
  stage2   method pilot at one speed
  stage3   data collection proper (config set core|full, levels, views)

Usage
  python make_temporal_gsc.py stage0
  python calibrate.py out_stage0 stage0_manifest.csv            # after rendering stage0
  python make_temporal_gsc.py stage1 --calib calibration.json
  python make_temporal_gsc.py stage2 --calib calibration.json --speed 0.5
  python make_temporal_gsc.py stage3 --calib calibration.json --speed 0.5 --cfgset core --views 3

Outputs: <stage>.gsc and <stage>_manifest.csv (one row per sequence; this is the metadata source of truth).
Sequence directory names keep the original scheme  scene-level-method-lit-rate-vmethod-subdiv-rot|trans-axis;
references use level 0 / rate 8; analytic references use the method token 'analytic'.
Each (stage, speed, view) renders into its own root directory out_<stage>[_s<speed>][_v<view>]; run blti.py once per root.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---- method numbering, identical to quality.gsc ------------------------------------------------------
BASE = {'lin': 0, 'quadB': 13, 'quadBf': 26, 'cubicB': 39, 'cubicBf': 52, 'cr': 65}   # unlit; lit = +7
VS_UNLIT = {'linvs': 3, 'crvs': 4, 'hermvs': 5, 'monhermvs': 6}                          # + BASE[method]
VS_LIT = {'linvs': 9, 'crvs': 10, 'hermvs': 11, 'monhermvs': 12}                         # + BASE[method]
ANALYTIC = {'n': 78, 'l': 79}
METHODS = list(BASE)


def method_index(method, vm, lit):
    if method == 'analytic':
        return ANALYTIC[lit]
    if vm == 'n':
        return BASE[method] + (7 if lit == 'l' else 0)
    return BASE[method] + (VS_LIT if lit == 'l' else VS_UNLIT)[vm]


# ---- motions -----------------------------------------------------------------------------------------
MOTIONS = {
    'roty':   dict(kind='rot', ttype='rot', axis='y', cmd='addrotationy {}'),
    'rotx':   dict(kind='rot', ttype='rot', axis='x', cmd='addrotationx {}'),
    'rotd':   dict(kind='rot', ttype='rot', axis='diag', cmd='addrotationaxis 0.57735027 0.57735027 0.57735027 {}'),
    'transx': dict(kind='trans', ttype='trans', axis='x', cmd='addtranslation {} 0 0', sign=+1),
    'transy': dict(kind='trans', ttype='trans', axis='y', cmd='addtranslation 0 {} 0', sign=-1),
    'transz': dict(kind='trans', ttype='trans', axis='z', cmd='addtranslation 0 0 {}', sign=-1, persp_only=True),
}

# ---- configuration sets: (method, rate, vmethod, subdiv); vmethod 'n' = none ---------------------------
CAL = [('lin', 1, 'n', 0)]
LADDER = [('lin', 1, 'n', 0), ('lin', 15, 'n', 0)]
PILOT = ([('lin', r, 'n', 0) for r in (1, 10, 15)]
         + [(m, 1, 'n', 0) for m in ('cr', 'quadB', 'quadBf', 'cubicB', 'cubicBf')]
         + [('cr', 1, vm, 6) for vm in ('linvs', 'hermvs', 'monhermvs')])
CORE = ([('lin', r, 'n', 0) for r in (1, 5, 10, 15)]
        + [(m, 1, 'n', 0) for m in ('cr', 'quadB', 'quadBf', 'cubicB', 'cubicBf')]
        + [(m, 1, vm, s) for m in ('lin', 'cr', 'cubicBf') for vm in VS_UNLIT for s in (1, 6, 18)])
FULL = ([(m, r, 'n', 0) for m in METHODS for r in (1, 5, 10, 15)]
        + [(m, 1, vm, s) for m in METHODS for vm in VS_UNLIT for s in (1, 6, 12, 18)])
CFGSETS = {'cal': CAL, 'ladder': LADDER, 'pilot': PILOT, 'core': CORE, 'full': FULL}

STAGES = {
    'stage0': dict(scenes='all', motions=list(MOTIONS), lighting='unlit', levels=[1], cfgset='cal',
                   refs='none', speeds=[None], frames=3, dup=False),
    'stage1': dict(scenes='all', motions=list(MOTIONS), lighting='unlit',
                   levels=[1], cfgset='ladder', refs='lin', speeds=[2, 1, 0.5, 0.25], frames=100, dup=True),
    'stage2': dict(scenes=['Head1', 'Tree1', 'ML1'], motions=['rotx', 'roty', 'rotd', 'transx', 'transz'],
                   lighting='both', levels=[1], cfgset='pilot', refs='per_method', speeds=[0.5],
                   frames=100, dup=False),
    'stage3': dict(scenes='all', motions=['roty', 'rotd', 'transx', 'transz'], lighting='both',
                   levels=[1, 2], cfgset='core', refs='per_method', speeds=[0.5], frames=100, dup=False),
}


def fmt(x):
    s = f"{x:.6g}"
    return '0' if float(s) == 0 else s


def speed_tag(speed):
    return 's' + ('%g' % speed).replace('.', 'p')


def load_scenes(names):
    scenes = json.load(open(HERE / 'scenes.json'))
    by = {s['name']: s for s in scenes}
    if names == 'all':
        chosen = scenes
    else:
        bad = [n for n in names if n not in by]
        if bad:
            sys.exit(f"unknown scene(s) {bad}; available: {list(by)}")
        chosen = [by[n] for n in names]
    chosen = sorted(chosen, key=lambda s: s['idx'])           # ascending: keeps the background-state invariant
    seen_custom_bg = False
    for s in chosen:                                           # renderer default background is never re-set
        if s['bg'] is not None:
            seen_custom_bg = True
        elif seen_custom_bg:
            sys.exit(f"scene {s['name']} (default background) would follow a scene with a custom background")
    return chosen


def make_seq(a, stage, sc, lit, level, method, rate, vm, subdiv, mo, speed, view, ref, calib):
    m = MOTIONS[mo]
    if speed is None:
        step = a.cal_rot if m['kind'] == 'rot' else a.cal_trans
    else:
        try:
            c = calib[sc['name']][mo]
        except KeyError:
            sys.exit(f"no calibration for scene {sc['name']} motion {mo}; run calibrate.py (stage0) first")
        step = speed / c
    inc = step if m['kind'] == 'rot' else m['sign'] * step
    start = -(a.frames - 1) / 2.0 * inc
    if m['kind'] == 'rot' and a.views > 1:
        start += view * 360.0 / a.views
    root = f"out_{stage}" + (f"_{speed_tag(speed)}" if speed is not None else '') + \
           (f"_v{view}" if a.views > 1 else '')
    name = f"{sc['name']}-{level}-{method}-{lit}-{rate}-{vm}-{subdiv}-{m['ttype']}-{m['axis']}"
    return dict(stage=stage, root=root, seq=name, scene=sc['name'], scene_idx=sc['idx'], level=level,
                method=method, lit=lit, rate=rate, vmethod=vm, subdiv=subdiv, motion=mo,
                trans_type=m['ttype'], trans_axis=m['axis'], is_ref=int(ref),
                speed_px='' if speed is None else speed, step=step, step_unit='deg' if m['kind'] == 'rot' else 'unit',
                frames=a.frames, view=view, views=a.views, res=a.res, start=start, inc=inc,
                start_op=m['cmd'].format(fmt(start)), inc_op=m['cmd'].format(fmt(inc)),
                method_index=method_index(method, vm, lit))


def ref_methods(sc, mode):
    if sc['analytic']:
        return ['analytic']
    return ['lin'] if mode == 'lin' else METHODS


def build_plan(a, stage, scenes, calib):
    lights = {'unlit': ['n'], 'lit': ['l'], 'both': ['n', 'l']}[a.lighting]
    cfgs = CFGSETS[a.cfgset]
    plan = []
    for sc in scenes:
        mos = [m for m in a.motions if not (MOTIONS[m].get('persp_only') and sc['ortho'])]
        for lit in lights:
            if a.refs != 'none':
                for rm in ref_methods(sc, a.refs):
                    for mo in mos:
                        for sp in a.speeds:
                            for v in range(a.views):
                                plan.append(make_seq(a, stage, sc, lit, 0, rm, 8, 'n', 0, mo, sp, v, True, calib))
            for level in a.levels:
                for (m, rate, vm, sd) in cfgs:
                    for mo in mos:
                        for sp in a.speeds:
                            for v in range(a.views):
                                plan.append(make_seq(a, stage, sc, lit, level, m, rate, vm, sd, mo, sp, v, False, calib))
    if a.dup:   # determinism check: the same lin@1 sequence rendered twice, second copy into its own root
        for i, s in enumerate(plan):
            if (not s['is_ref'] and s['method'] == 'lin' and s['rate'] == 1 and s['vmethod'] == 'n'
                    and s['speed_px'] == a.dup_speed and s['view'] == 0):
                d = dict(s, root=s['root'] + '_dup')
                plan.insert(i + 1, d)
                break
    return plan


def emit(plan, a, stage, scenes):
    by = {s['name']: s for s in scenes}
    out = [f"# generated by make_temporal_gsc.py  stage={stage}  sequences={len(plan)}",
           "setinteraction 0", f"logfile {stage}.log", "clearlog", f"resize {a.res} {a.res}"]
    if a.fps:
        out.append("setfpswindow 5.0")
    cur_scene, cur_group, last = None, None, {}
    for sq in plan:
        sc = by[sq['scene']]
        if sq['scene'] != cur_scene:
            cur_scene = sq['scene']
            out += [f"# ===== scene {sc['name']} =====", f"setvolume {sc['volume']}"]
            if sc['bg'] is not None:
                out.append("setbackground " + ' '.join(fmt(x) for x in sc['bg']))
            cur_group = None
        if (sq['scene'], sq['lit']) != cur_group:
            cur_group = (sq['scene'], sq['lit'])
            last = {}
            out.append(f"# --- {sc['name']} {'lit' if sq['lit'] == 'l' else 'unlit'} ---")
            out.append(sc['tf_lit'] if sq['lit'] == 'l' else sc['tf_unlit'])
            if sq['lit'] == 'l':
                out.append("setspecularcolor 0.8 0.8 0.8")
        for key, cmd in (('level', f"setLevel {sq['level']}"), ('rate', f"setrate {sq['rate']}"),
                         ('subdiv', f"setsubdiv {sq['subdiv'] if sq['vmethod'] != 'n' else 1}"),
                         ('method', f"setmethod {sq['method_index']}")):
            if last.get(key) != cmd:
                out.append(cmd)
                last[key] = cmd
        out += [f"setdir {sq['root']}", "settranslation 0 0 0", "resetrotation"] + sc['orient']
        out.append(f"setuseortho {'true' if sc['ortho'] else 'false'}")
        if float(fmt(sq['start'])) != 0:
            out.append(sq['start_op'])
        out += [f"setdir {sq['root']}/{sq['seq']}", f"log {sq['root']}/{sq['seq']}",
                f"repeat {sq['frames']} as r", f"    screenshot {sq['seq']}-$r.png", f"    {sq['inc_op']}", "endrepeat"]
    out += ["setdir " + plan[0]['root'], "log all done", "quit"]
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=list(STAGES))
    ap.add_argument('--calib', help='calibration.json from calibrate.py (px per nominal unit)')
    ap.add_argument('--scenes', nargs='+')
    ap.add_argument('--motions', nargs='+', choices=list(MOTIONS))
    ap.add_argument('--lighting', choices=['unlit', 'lit', 'both'])
    ap.add_argument('--levels', nargs='+', type=int)
    ap.add_argument('--cfgset', choices=list(CFGSETS))
    ap.add_argument('--refs', choices=['none', 'lin', 'per_method'])
    ap.add_argument('--speed', type=float, help='single target speed in px/frame (overrides --speeds)')
    ap.add_argument('--speeds', nargs='+', type=float)
    ap.add_argument('--frames', type=int)
    ap.add_argument('--views', type=int, default=1, help='rotation start angles spread over 360 deg')
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--cal-rot', type=float, default=0.9, help='stage0 rotation step in degrees')
    ap.add_argument('--cal-trans', type=float, default=0.0025, help='stage0 translation step in units')
    ap.add_argument('--dup-speed', type=float, default=0.5)
    ap.add_argument('--fps', action='store_true', help='emit setfpswindow 5.0 as quality.gsc does')
    ap.add_argument('--out', help='output prefix (default: stage name)')
    a = ap.parse_args()

    d = dict(STAGES[a.stage])
    for k in ('scenes', 'motions', 'lighting', 'levels', 'cfgset', 'refs', 'frames'):
        v = getattr(a, k)
        if v is not None:
            d[k] = v
    if a.speed is not None:
        d['speeds'] = [a.speed]
    elif a.speeds is not None:
        d['speeds'] = a.speeds
    for k, v in d.items():
        setattr(a, k, v)
    a.dup = d['dup']
    if a.scenes != 'all' and a.scenes == ['all']:
        a.scenes = 'all'
    calib = json.load(open(a.calib)) if a.calib else {}
    if a.stage != 'stage0' and not a.calib:
        sys.exit("--calib calibration.json is required for this stage")

    scenes = load_scenes(a.scenes)
    plan = build_plan(a, a.stage, scenes, calib)
    prefix = a.out or a.stage
    Path(f"{prefix}.gsc").write_text(emit(plan, a, a.stage, scenes))
    cols = list(plan[0])
    with open(f"{prefix}_manifest.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(plan)
    roots = {}
    for s in plan:
        roots[s['root']] = roots.get(s['root'], 0) + 1
    nref = sum(s['is_ref'] for s in plan)
    print(f"{prefix}.gsc: {len(plan)} sequences ({nref} reference, {len(plan) - nref} test), "
          f"{sum(s['frames'] for s in plan)} frames at {a.res}x{a.res}")
    for r, n in roots.items():
        print(f"  {r}: {n}")


if __name__ == '__main__':
    main()
