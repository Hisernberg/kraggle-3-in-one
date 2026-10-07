"""FL v3 research: per-fragment records -> protocol-matched aggregations (least extrapolation, central)."""
import math, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/kraggle-3-in-one/umud')
from umud.geometry import find_aponeuroses, _wmedian, _intersect


def fragments(fp, apo, ppm, thr=0.35, min_len_mm=3.0):
    H, W = fp.shape
    s, d = apo['sup'], apo['deep']
    sup_line = s['bot'] if s['bot'] is not None else s['cen']
    deep_line = d['top'] if d['top'] is not None else d['cen']
    xs = np.arange(W)
    y_sup, y_deep = np.polyval(sup_line, xs), np.polyval(deep_line, xs)
    yy = np.arange(H)[:, None]
    region = (yy > y_sup[None] + 0.5 * ppm) & (yy < y_deep[None] - 0.5 * ppm)
    p = np.where(region, fp, 0).astype(np.float32)
    n, lab, st, _ = cv2.connectedComponentsWithStats((p >= thr).astype(np.uint8), 8)
    deep_ang = math.degrees(math.atan(d['cen'][0]))
    R = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 10:
            continue
        ys, xk = np.where(lab == k)
        pts = np.stack([xk, ys], 1).astype(float)
        w = p[ys, xk]
        mu = (pts * w[:, None]).sum(0) / w.sum()
        q = pts - mu
        ev, vec = np.linalg.eigh((q * w[:, None]).T @ q / w.sum())
        v = vec[:, -1]
        length = 4 * math.sqrt(max(ev[-1], 0))
        if length < min_len_mm * ppm or math.sqrt(max(ev[-1], 1e-9) / max(ev[0], 1e-9)) < 3:
            continue
        ang = math.degrees(math.atan2(v[1], v[0]))
        d_ = abs(ang - deep_ang) % 180
        pa = min(d_, 180 - d_)
        if not 2 <= pa <= 60:
            continue
        slope = v[1] / v[0] if abs(v[0]) > 1e-9 else 1e9
        c0 = mu[1] - slope * mu[0]
        xu, xl = _intersect(slope, c0, sup_line), _intersect(slope, c0, deep_line)
        if xu is None or xl is None:
            continue
        yu, yl = slope * xu + c0, slope * xl + c0
        fl_px = math.hypot(xu - xl, yu - yl)
        rel = ((ang - deep_ang + 90) % 180) - 90
        xmid = 0.5 * (xu + xl)
        R.append(dict(pa=pa, fl=fl_px / ppm, vis=min(length / max(fl_px, 1e-6), 1.0), length=length / ppm,
                      conf=float(w.mean()), cen=abs(mu[0] - W / 2) / W, cen_mid=abs(xmid - W / 2) / W,
                      inside=float(0 <= xu < W and 0 <= xl < W), rel=rel))
    if not R:
        return pd.DataFrame()
    R = pd.DataFrame(R)
    wt = R.length * R.conf
    sign = np.sign(np.average(np.sign(R.rel), weights=wt)) or 1.0
    R = R[np.sign(R.rel) == sign]
    return R[(R.fl > 10) & (R.fl < 300)]


def aggregates(R):
    out = {}
    if len(R) == 0:
        return out
    w = (R.length * R.conf).values
    out['fl_med'] = float(R.fl.median())
    out['fl_wmed'] = _wmedian(R.fl.values, w)
    for k in (3, 5):
        t = R.sort_values('vis', ascending=False).head(k)
        out[f'fl_vis{k}'] = float(t.fl.mean())
        out[f'fl_vis{k}_med'] = float(t.fl.median())
    out['fl_viswmed'] = _wmedian(R.fl.values, (R.vis ** 2 * R.conf).values)
    c = R[R.cen_mid < 0.3]
    if len(c):
        out['fl_cvis3'] = float(c.sort_values('vis', ascending=False).head(3).fl.mean())
        out['fl_cwmed'] = _wmedian(c.fl.values, (c.length * c.conf).values)
    ins = R[R.inside > 0]
    if len(ins):
        out['fl_in_med'] = float(ins.fl.median())
        out['fl_in_vis3'] = float(ins.sort_values('vis', ascending=False).head(3).fl.mean())
    out['n_frag'] = len(R)
    out['vis_top'] = float(R.vis.max())
    return out


def run(seg_dir, out_csv):
    seg_dir = Path(seg_dir)
    crops = pd.read_csv(seg_dir / 'crops.csv')
    rows = []
    for r in crops.itertuples():
        ap = cv2.imread(str(seg_dir / 'probs' / f'{r.image_id}_apo.png'), cv2.IMREAD_GRAYSCALE)
        fp = cv2.imread(str(seg_dir / 'probs' / f'{r.image_id}_fasc.png'), cv2.IMREAD_GRAYSCALE)
        rec = dict(image_id=r.image_id)
        if ap is not None and fp is not None:
            ap = ap.astype(np.float32) / 255; fp = fp.astype(np.float32) / 255
            apo = find_aponeuroses(ap, r.px_per_mm) or find_aponeuroses(ap, r.px_per_mm, thr=0.3)
            if apo is not None:
                rec.update(aggregates(fragments(fp, apo, r.px_per_mm)))
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print('wrote', out_csv, len(rows))


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
