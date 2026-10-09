"""Multi-threshold candidate forest at 1024 -> 2048 node masks, features, per-record targets.

python forest.py build --maps DIR [DIR...] --stems oof|test --out nodes_xxx.pkl [--levels ...]
Each image -> dict(stem, nodes: list of dict(level, area1024, rle2048, parent, feats), iou[r] (n_nodes x n_gt_r))
"""
import sys, os, json, pickle, argparse
sys.path.insert(0, '/home/user/kraggle-3-in-one/cobalt_heron')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
from scipy import ndimage as ndi
from pycocotools import mask as mu
from ch.post import instances
from ch.inst_feats import inst_features, FEATS

W = Path('/home/user/work/filament')
CACHE = W / 'w' / 'c1024'
TEST1024 = W / 'w' / 'test1024'
LEVELS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
A = None


def load_ens(dirs, stem):
    return np.mean([np.load(Path(d) / f'{stem}.npz')['p'].astype(np.float32) / 255 for d in dirs], 0)


def gt2048():
    gp = W / 'w' / 'gt2048.pkl'
    if gp.exists(): return pickle.load(open(gp, 'rb'))
    from ch.data import load_coco, record_rles
    out = {s: [record_rles(a) for _, a in rs] for s, rs in load_coco().items()}
    pickle.dump(out, open(gp, 'wb')); return out


def up_labels(lab, p2, t2):
    """1024 label map -> 2048 label map; fg = p2 > t2 within dilated support, nearest-label assignment."""
    l2 = cv2.resize(lab.astype(np.float32), (2048, 2048), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    lf = l2.astype(np.float32); k3 = np.ones((3, 3), np.uint8)
    d1 = np.where(lf > 0, lf, cv2.dilate(lf, k3)); d2 = np.where(d1 > 0, d1, cv2.dilate(d1, k3))
    return np.where(p2 > t2, d2, 0).astype(np.int32)


def build_one(args):
    stem, dirs, img_path, levels, merge, min_area, kup, gts = args
    p = load_ens(dirs, stem)
    img = np.load(img_path)['img'] if img_path.endswith('.npz') else np.load(img_path)
    p2 = cv2.resize(p[0], (2048, 2048), interpolation=cv2.INTER_LINEAR)
    nodes = []; prev = None  # prev: (lab, node index offset) of the lower level
    for li, t in enumerate(levels):
        lab, _ = instances(p[0], p[1], t=t, te=1.1, merge=merge, min_area=min_area, min_score=0)
        n = int(lab.max())
        if n == 0: break
        X = inst_features(lab, p[0], p[1], img)
        t2 = t * kup if kup < 2 else min(t + (kup - 2), 0.95)
        l2 = up_labels(lab, p2, t2)
        objs = ndi.find_objects(lab); objs2 = ndi.find_objects(l2, max_label=n)
        off = len(nodes)
        for k in range(1, n + 1):
            par = -1
            if prev is not None:
                plab, poff = prev
                sl = objs[k - 1]; vals = plab[sl][lab[sl] == k]; vals = vals[vals > 0]
                if len(vals): par = poff + int(np.bincount(vals).argmax()) - 1
            m2 = np.zeros((2048, 2048), np.uint8, order='F'); s2 = objs2[k - 1]
            if s2 is not None: m2[s2] = (l2[s2] == k)
            a2 = int(m2[s2].sum()) if s2 is not None else 0
            rle = mu.encode(m2)
            nodes.append(dict(level=t, li=li, parent=par, feats=X[k - 1], a2=a2, rle=rle))
        prev = (lab, off)
    # children counts / structural feats
    nch = np.zeros(len(nodes))
    for i, nd in enumerate(nodes):
        if nd['parent'] >= 0: nch[nd['parent']] += 1
    F = []
    for i, nd in enumerate(nodes):
        pa = nodes[nd['parent']]['feats'][0] if nd['parent'] >= 0 else 0
        F.append(np.concatenate([nd['feats'], [nd['level'], nch[i], nd['feats'][0] / pa if pa else 0, nd['a2']]]))
    F = np.array(F, np.float32) if F else np.zeros((0, len(FEATS) + 4), np.float32)
    rles = [nd['rle'] for nd in nodes]
    ious = []
    if gts is not None:
        for g in gts:
            if len(g) and len(rles): ious.append(np.asarray(mu.iou(rles, g, [0] * len(g)), np.float32).reshape(len(rles), len(g)))
            else: ious.append(np.zeros((len(rles), len(g)), np.float32))
    return dict(stem=stem, level=np.array([nd['level'] for nd in nodes]), parent=np.array([nd['parent'] for nd in nodes], int),
                F=F, a2=np.array([nd['a2'] for nd in nodes]), rles=rles, ious=ious)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--maps', nargs='+'); ap.add_argument('--test', action='store_true')
    ap.add_argument('--out'); ap.add_argument('--levels', default=','.join(map(str, LEVELS))); ap.add_argument('--merge', type=int, default=4)
    ap.add_argument('--min-area', type=int, default=20); ap.add_argument('--kup', type=float, default=1.2); ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--every', type=int, default=1)
    a = ap.parse_args(); levels = [float(x) for x in a.levels.split(',')]
    jobs = []
    if a.test:
        for f in sorted(Path(a.maps[0]).glob('*.npz')):
            jobs.append((f.stem, a.maps, str(TEST1024 / f'{f.stem}.npy'), levels, a.merge, a.min_area, a.kup, None))
    else:
        G = gt2048()
        # maps may be "fold-dirs": a list of dir groups separated by '+' per fold: r18-f1+r34-f1 ...
        for grp in a.maps:
            dirs = grp.split('+')
            for f in sorted(Path(dirs[0]).glob('*.npz')):
                jobs.append((f.stem, dirs, str(CACHE / f'{f.stem}.npz'), levels, a.merge, a.min_area, a.kup, G[f.stem]))
    jobs = jobs[::a.every]
    out = []
    with ProcessPoolExecutor(a.procs) as ex:
        for i, r in enumerate(ex.map(build_one, jobs, chunksize=2)):
            out.append(r)
            if i % 50 == 0: print(i, len(jobs), flush=True)
    pickle.dump(out, open(a.out, 'wb')); print('saved', a.out, len(out), sum(len(r['rles']) for r in out))


if __name__ == '__main__':
    main()
