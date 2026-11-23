#!/usr/bin/env python3
"""One-off: read the scene definitions (volume index, transfer functions, orientation, background) out of an
existing temporal.gsc and write scenes.json. The long transfer-function strings are copied verbatim.

  python extract_scenes.py temporal.gsc scenes.json
"""
import json, re, sys

def main(src, dst):
    L = open(src).read().split('\n')
    idx = lambda pred, start=0: next(i for i in range(start, len(L)) if pred(L[i].strip()))
    i_log = idx(lambda s: s == 'log Volume $volume')
    i_lit = idx(lambda s: s == 'log lighting pass')
    i_x = idx(lambda s: s == 'repeat 6 as x')
    i_tr = idx(lambda s: s == 'repeat 6 as transform')
    i_t0 = idx(lambda s: s.startswith('if $transform == 0'), i_tr)

    def blocks(lines):
        out, cur = {}, None
        for l in lines:
            s = l.strip()
            m = re.match(r'if \$volume == (\d+)$', s)
            if m: cur = int(m.group(1)); out.setdefault(cur, []); continue
            if s == 'endif': cur = None; continue
            if cur is not None and s and not s.startswith('#'): out[cur].append(s)
        return out

    setup, lit, orient = blocks(L[:i_log]), blocks(L[i_lit:i_x]), blocks(L[i_tr:i_t0])
    scenes = []
    for v in sorted(setup):
        s = setup[v]
        get = lambda pre: [x for x in s if x.startswith(pre)]
        name = get('set volname')[0].split()[2]
        analytic = get('set analytic')[0].split()[2] == 'true'
        volume = int(get('setvolume')[0].split()[1])
        tf_unlit = [x for x in s if x.startswith(('settfparams', 'settfcode'))][0]
        tf_lit = ([x for x in lit.get(v, []) if x.startswith(('settfparams', 'settfcode'))] or [tf_unlit])[0]
        bg = get('setbackground')
        bg = [float(t) for t in bg[0].split()[1:]] if bg else None
        ops = orient[v]
        ortho = [x for x in ops if x.startswith('setuseortho')][-1].split()[1] == 'true'
        scenes.append(dict(idx=v, name=name, analytic=analytic, volume=volume, tf_unlit=tf_unlit, tf_lit=tf_lit,
                           bg=bg, ortho=ortho, orient=[x for x in ops if not x.startswith('setuseortho')]))
    json.dump(scenes, open(dst, 'w'), indent=1)
    for sc in scenes:
        print(f"{sc['idx']} {sc['name']:<10} vol={sc['volume']} analytic={sc['analytic']!s:<5} ortho={sc['ortho']!s:<5} "
              f"bg={sc['bg']} tf_differs={sc['tf_unlit'] != sc['tf_lit']} orient={[o[:40] for o in sc['orient']]}")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
