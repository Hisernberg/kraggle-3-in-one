"""Zoomed aponeurosis-edge crops: pipeline inner edges vs edges implied by the reference MT (blind A/B)."""
import sys, json, random
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/kraggle-3-in-one/umud')
from umud.geometry import find_aponeuroses
from umud import seg as S
W = Path('/home/user/work'); U = Path('/home/user/kraggle-3-in-one/umud')
c = pd.read_csv(W / 'seg_ens/crops.csv').set_index('image_id')
f = pd.read_csv(U / 'daily/inputs/features_ens.csv').set_index('image_id')
v = pd.read_csv(U / 'daily/inputs/ref_vera_public.csv').set_index('image_id')

def render(iid, out, rng, zoom=3, half_w=60):
    cr = c.loc[iid]; ppm = cr.px_per_mm
    g = S.read_gray(str(W / 'data/test_images_v2/test_set_v2' / iid))[int(cr.t):int(cr.b), int(cr.l):int(cr.r)]
    ap = cv2.imread(str(W / 'seg_ens/probs' / f'{iid}_apo.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    apo = find_aponeuroses(ap, ppm) or find_aponeuroses(ap, ppm, thr=0.3)
    sup = apo['sup']['bot']; deep = apo['deep']['top']
    dpx = (f.loc[iid, 'mt_inner'] - v.loc[iid, 'mt_mm']) * ppm  # pipeline thicker by dpx
    H, Wd = g.shape
    tiles = []
    for fx in (0.25, 0.5, 0.75):
        x0 = int(fx * Wd)
        ys, yd = np.polyval(sup, x0), np.polyval(deep, x0)
        for name, yc in (('SUP', ys), ('DEEP', yd)):
            y0 = int(max(0, yc - 2.5 * ppm)); y1 = int(min(H, yc + 2.5 * ppm))
            xa, xb = max(0, x0 - half_w), min(Wd, x0 + half_w)
            t = cv2.cvtColor(g[y0:y1, xa:xb], cv2.COLOR_GRAY2BGR)
            t = cv2.resize(t, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
            tiles.append((name, fx, t, (yc - y0) * zoom, xa, x0))
    # lines: A/B random: pipeline edge vs reference-implied edge (half the difference on each side)
    pipe_first = rng.random() < 0.5
    panels = []
    for lab in ('A', 'B'):
        is_pipe = (lab == 'A') == pipe_first
        row = []
        for name, fx, t, yz, xa, x0 in tiles:
            tt = t.copy()
            shift = 0 if is_pipe else (dpx / 2) * zoom * (1 if name == 'SUP' else -1)
            y = int(round(yz + shift))
            for xx in range(0, tt.shape[1], 8):
                cv2.line(tt, (xx, y), (min(xx + 4, tt.shape[1] - 1), y), (0, 255, 255), 1)
            cv2.putText(tt, f'{lab} {name} {int(fx*100)}%', (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            row.append(tt)
        hmax = max(r.shape[0] for r in row)
        row = [cv2.copyMakeBorder(r, 0, hmax - r.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in row]
        panels.append(np.hstack(row))
    cleanrow = []
    for name, fx, t, yz, xa, x0 in tiles:
        tt = t.copy(); cv2.putText(tt, f'clean {name}', (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1); cleanrow.append(tt)
    hmax = max(r.shape[0] for r in cleanrow)
    cleanrow = [cv2.copyMakeBorder(r, 0, hmax - r.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in cleanrow]
    img = np.vstack([np.hstack(cleanrow), np.full((6, panels[0].shape[1], 3), 255, np.uint8), panels[0], np.full((6, panels[0].shape[1], 3), 255, np.uint8), panels[1]])
    cv2.imwrite(str(out), img)
    return dict(image_id=iid, pipe='A' if pipe_first else 'B', dpx=float(dpx), dmm=float(dpx / ppm))

if __name__ == '__main__':
    ids = sys.argv[2:]; out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(77); key = {}
    for k, iid in enumerate(ids):
        key[f'e{k:02d}'] = render(iid, out / f'e{k:02d}.png', rng)
    json.dump(key, open(out / 'key.json', 'w')); print('rendered', len(key))
