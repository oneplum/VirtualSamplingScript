#!/usr/bin/env python3
"""
Independent checker for generated .gsc files. It executes the script in a tiny simulator (set* commands update a
state dictionary, repeat blocks are unrolled) and, for every screenshot sequence, verifies the renderer state against
the manifest: volume, background, transfer function, lighting, level, rate, subdiv, method number, projection,
camera start pose, per-frame increment, frame count, file names and output directory. It catches exactly the kind of
bug temporal.gsc had (state leaking from one loop iteration into the next).

  python check_gsc.py stage2.gsc stage2_manifest.csv
"""
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# method numbers exactly as listed in quality.gsc (written out, NOT computed, so this check is independent)
Q_UNLIT = {'lin': 0, 'quadB': 13, 'quadBf': 26, 'cubicB': 39, 'cubicBf': 52, 'cr': 65}
Q_LIT = {'lin': 7, 'quadB': 20, 'quadBf': 33, 'cubicB': 46, 'cubicBf': 59, 'cr': 72}
Q_VS_UNLIT = {'lin': [3, 4, 5, 6], 'quadB': [16, 17, 18, 19], 'quadBf': [29, 30, 31, 32],
              'cubicB': [42, 43, 44, 45], 'cubicBf': [55, 56, 57, 58], 'cr': [68, 69, 70, 71]}
Q_VS_LIT = {'lin': [9, 10, 11, 12], 'quadB': [22, 23, 24, 25], 'quadBf': [35, 36, 37, 38],
            'cubicB': [48, 49, 50, 51], 'cubicBf': [61, 62, 63, 64], 'cr': [74, 75, 76, 77]}
VS_ORDER = ['linvs', 'crvs', 'hermvs', 'monhermvs']


def expected_method(method, vm, lit):
    if method == 'analytic':
        return 79 if lit == 'l' else 78
    if vm == 'n':
        return (Q_LIT if lit == 'l' else Q_UNLIT)[method]
    return (Q_VS_LIT if lit == 'l' else Q_VS_UNLIT)[method][VS_ORDER.index(vm)]


def simulate(text):
    lines = [l.strip() for l in text.split('\n')]
    lines = [l for l in lines if l and not l.startswith('#')]
    st = dict(volume=None, bg=None, tf=None, spec=None, level=None, rate=None, subdiv=None, method=None,
              ortho=None, dir=None, ops=[])
    seqs, cur = [], None

    def run(line):
        nonlocal cur
        w = line.split()
        c = w[0]
        if c == 'setvolume': st['volume'] = int(w[1])
        elif c == 'setbackground': st['bg'] = [float(x) for x in w[1:]]
        elif c in ('settfparams', 'settfcode'): st['tf'] = line
        elif c == 'setspecularcolor': st['spec'] = [float(x) for x in w[1:]]
        elif c == 'setLevel': st['level'] = int(w[1])
        elif c == 'setrate': st['rate'] = int(w[1])
        elif c == 'setsubdiv': st['subdiv'] = int(w[1])
        elif c == 'setmethod': st['method'] = int(w[1])
        elif c == 'setuseortho': st['ortho'] = (w[1] == 'true')
        elif c == 'setdir': st['dir'] = w[1]
        elif c == 'settranslation': pass
        elif c == 'resetrotation': st['ops'] = []
        elif c.startswith('addrotation') or c.startswith('addtranslation') or c == 'settransformparams':
            st['ops'].append(line)
        elif c == 'screenshot':
            if cur is None or cur['dir'] != st['dir']:
                cur = dict(dir=st['dir'], state={k: (list(v) if isinstance(v, list) else v) for k, v in st.items()},
                           ops_at=[], files=[])
                seqs.append(cur)
            cur['files'].append(w[1])
            cur['ops_at'].append(list(st['ops']))
        elif c in ('setinteraction', 'logfile', 'clearlog', 'resize', 'setfpswindow', 'log', 'quit'): pass
        else:
            raise ValueError(f"unknown command: {line}")

    i = 0
    while i < len(lines):
        l = lines[i]
        m = re.match(r'repeat (\d+) as (\w+)$', l)
        if m:
            n, var = int(m.group(1)), m.group(2)
            j = i + 1
            body = []
            while lines[j] != 'endrepeat':
                if lines[j].startswith('repeat'):
                    raise ValueError("nested repeat not supported by the checker")
                body.append(lines[j]); j += 1
            for r in range(n):
                for b in body:
                    run(b.replace('$' + var, str(r)))
            i = j + 1
        else:
            run(l); i += 1
    return seqs


def num(op, mo):
    """value of an add* operation; for translations the value sits in the slot of its axis."""
    w = op.split()
    if mo.startswith('trans'):
        return float(w[{'transx': 1, 'transy': 2, 'transz': 3}[mo]])
    return float(w[-1])


def check(gsc, manifest_csv):
    scenes = {s['name']: s for s in json.load(open(HERE / 'scenes.json'))}
    rows = list(csv.DictReader(open(manifest_csv)))
    seqs = simulate(open(gsc).read())
    errs = []
    by_dir = {}
    for s in seqs:
        if s['dir'] in by_dir:
            errs.append(f"sequence directory rendered twice: {s['dir']}")
        by_dir[s['dir']] = s
    want = {f"{r['root']}/{r['seq']}": r for r in rows}
    for d in set(by_dir) - set(want):
        errs.append(f"script renders a sequence that is not in the manifest: {d}")
    for d in set(want) - set(by_dir):
        errs.append(f"manifest sequence never rendered: {d}")
    for d, r in want.items():
        if d not in by_dir:
            continue
        s, st = by_dir[d], by_dir[d]['state']
        sc = scenes[r['scene']]
        lit = r['lit']
        e = lambda msg: errs.append(f"{d}: {msg}")
        if st['volume'] != sc['volume']: e(f"volume {st['volume']} != {sc['volume']}")
        if st['bg'] != sc['bg']: e(f"background {st['bg']} != {sc['bg']}")
        want_tf = sc['tf_lit'] if lit == 'l' else sc['tf_unlit']
        if st['tf'] != want_tf: e("transfer function does not match the scene/lighting")
        if lit == 'l' and st['spec'] != [0.8, 0.8, 0.8]: e(f"specular {st['spec']} in lit mode")
        if st['level'] != int(r['level']): e(f"level {st['level']} != {r['level']}")
        if st['rate'] != int(r['rate']): e(f"rate {st['rate']} != {r['rate']}")
        sd = int(r['subdiv']) if r['vmethod'] != 'n' else 1
        if st['subdiv'] != sd: e(f"subdiv {st['subdiv']} != {sd}")
        wm = expected_method(r['method'], r['vmethod'], lit)
        if st['method'] != wm: e(f"method {st['method']} != {wm} ({r['method']}/{r['vmethod']}/{lit})")
        if st['ortho'] != sc['ortho']: e(f"ortho {st['ortho']} != {sc['ortho']}")
        F = int(r['frames'])
        if s['files'] != [f"{r['seq']}-{k}.png" for k in range(F)]: e("screenshot file names / count differ")
        if not s['ops_at']:
            continue
        pre = s['ops_at'][0]
        if pre[:len(sc['orient'])] != sc['orient']: e("initial orientation differs from the scene definition")
        rest = pre[len(sc['orient']):]
        mo = r['motion']
        inc_v, step = float(r['inc']), float(r['step'])
        if mo.startswith('rot'):
            exp_inc = step
            exp_start = -(F - 1) / 2 * step + (int(r['view']) * 360.0 / int(r['views']) if int(r['views']) > 1 else 0)
        else:
            exp_inc = {'transx': +1, 'transy': -1, 'transz': -1}[mo] * step
            exp_start = -(F - 1) / 2 * exp_inc
        if len(rest) > 1: e(f"unexpected extra start ops {rest}")
        got_start = num(rest[0], mo) if rest else 0.0
        if abs(got_start - exp_start) > 1e-4 * max(1.0, abs(exp_start)): e(f"start offset {got_start} != {exp_start}")
        for k in range(1, F):
            added = s['ops_at'][k][len(s['ops_at'][k - 1]):]
            if s['ops_at'][k][:len(s['ops_at'][k - 1])] != s['ops_at'][k - 1] or len(added) != 1:
                e(f"frame {k}: operations differ from previous frame + 1 increment"); break
            if abs(num(added[0], mo) - exp_inc) > 1e-4 * max(1.0, abs(exp_inc)):
                e(f"frame {k}: increment {num(added[0], mo)} != {exp_inc}"); break
        if mo == 'transz' and sc['ortho']: e("z translation requested for an orthographic scene")
    return errs, len(seqs), len(rows)


if __name__ == '__main__':
    errs, ns, nr = check(sys.argv[1], sys.argv[2])
    print(f"simulated {ns} sequences, manifest has {nr}")
    if errs:
        print(f"{len(errs)} PROBLEM(S):")
        for x in errs[:40]:
            print("  -", x)
        sys.exit(1)
    print("OK: every sequence's renderer state matches the manifest")
