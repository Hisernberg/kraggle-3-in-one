"""Blind visual forced choice between two FL candidates (human-in-the-loop arbitration, declared manual analysis).

For one image, draws the straight fascicle lines implied by each candidate FL (a straight path between the detected
superficial and deep aponeurosis inner edges whose length at the image centre equals the candidate; all lines share that
angle) in randomly labelled panels A/B under a clean panel. The rater picks the panel whose lines follow the visible
fascicle fragments; the key (which panel is which source) is written to key.json and not shown while rating.

    python scripts/forced_choice.py osf --work /home/user/work --out work/fc_osf      # validation: truth vs truth x0.78/1.28
    python scripts/forced_choice.py neuage --work /home/user/work --out work/fc_neu
    python scripts/forced_choice.py test --work /home/user/work --out work/fc_test --min-diff 15   # pipeline vs reference
    python scripts/forced_choice.py score --out work/fc_osf --answers answers.csv          # validation accuracy

Answers: CSV rows `panel_id,A|B|X,confidence(0-3)`. Results of the 2026-10-06 run: external/visual_fc/.
"""
import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

UMUD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UMUD))
from umud import seg as S  # noqa: E402
from umud.geometry import find_aponeuroses  # noqa: E402


def _apo_lines(apo):
    sup = apo['sup']['bot'] if apo['sup']['bot'] is not None else apo['sup']['cen']
    deep = apo['deep']['top'] if apo['deep']['top'] is not None else apo['deep']['cen']
    return sup, deep


def _chord(sup, deep, xm, ang):
    """Segment through the mid-point between the aponeuroses at x=xm with image angle ang, clipped to both lines."""
    m = np.array([xm, (np.polyval(sup, xm) + np.polyval(deep, xm)) / 2])
    d = np.array([math.cos(ang), math.sin(ang)])
    pts = []
    for ln in (sup, deep):
        den = d[1] - ln[0] * d[0]
        if abs(den) < 1e-9:
            return None
        t = (ln[0] * m[0] + ln[1] - m[1]) / den
        pts.append(m + t * d)
    return pts[1], pts[0]  # deep point, sup point


def angle_for(apo, s, L_px):
    """Image angle (rad) of the line through the image centre whose aponeurosis-to-aponeurosis length is L_px."""
    sup, deep = _apo_lines(apo)
    phi = math.atan(deep[0])
    xm = apo['W'] / 2
    lo, hi = math.radians(1), math.radians(85)
    for _ in range(60):
        mid = (lo + hi) / 2
        r = _chord(sup, deep, xm, phi + s * mid)
        ln = np.inf if r is None else float(np.hypot(*(r[1] - r[0])))
        if ln > L_px:
            lo = mid
        else:
            hi = mid
    return phi + s * (lo + hi) / 2, math.degrees((lo + hi) / 2)


def lines_for(apo, s, L_px, xs_mid):
    sup, deep = _apo_lines(apo)
    ang, th = angle_for(apo, s, L_px)
    segs = []
    for xm in xs_mid:
        r = _chord(sup, deep, xm, ang)
        if r is not None:
            segs.append((r[0], r[1], th))
    return segs


def _dashed(img, p0, p1, col, on=9, off=7):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = float(np.hypot(*(p1 - p0)))
    if not np.isfinite(n) or n < 1:
        return
    u = (p1 - p0) / n
    t = 0.0
    while t < n:
        a, b = p0 + u * t, p0 + u * min(t + on, n)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 1, cv2.LINE_AA)
        t += on + off


def roi(apo, H, W):
    xs = []
    for k in ('sup', 'deep'):
        for e in ('top_raw', 'bot_raw'):
            v = apo[k].get(e)
            if v is not None:
                xs.append(np.isfinite(v))
    ok = np.logical_or.reduce(xs) if xs else np.ones(W, bool)
    xi = np.flatnonzero(ok)
    x0, x1 = (int(xi.min()), int(xi.max())) if len(xi) > 10 else (0, W - 1)
    sup, deep = _apo_lines(apo)
    xx = np.arange(x0, x1 + 1)
    ys, yd = np.polyval(sup, xx), np.polyval(deep, xx)
    mt = float(np.median(yd - ys))
    y0 = int(max(0, ys.min() - 0.3 * mt))
    y1 = int(min(H - 1, yd.max() + 0.3 * mt))
    return x0, y0, x1, y1


def render(gray, apo, s, ppm, cands, out, nlines=5, width=1000):
    H, W = gray.shape
    x0, y0, x1, y1 = roi(apo, H, W)
    sc = width / (x1 - x0 + 1)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    crop = cv2.resize(g[y0:y1 + 1, x0:x1 + 1], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    base = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    panels = [base.copy()]
    cv2.putText(panels[0], 'clean', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    xs_mid = x0 + np.linspace(0.12, 0.88, nlines) * (x1 - x0)
    for i, L in enumerate(cands):
        ov = base.copy()
        for p0, p1, th in lines_for(apo, s, L * ppm, xs_mid):
            q0 = ((p0[0] - x0) * sc, (p0[1] - y0) * sc)
            q1 = ((p1[0] - x0) * sc, (p1[1] - y0) * sc)
            _dashed(ov, q0, q1, (0, 255, 255))
        vis = cv2.addWeighted(ov, 0.8, base, 0.2, 0)
        cv2.putText(vis, 'ABCDE'[i], (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        panels.append(vis)
    if (y1 - y0) / (x1 - x0 + 1) > 0.9:  # portrait crop: side by side, re-scaled to a common height
        hh = 900
        panels = [cv2.resize(p, (int(p.shape[1] * hh / p.shape[0]), hh), interpolation=cv2.INTER_AREA) for p in panels]
        sep = np.full((hh, 6, 3), 255, np.uint8)
        img = panels[0]
        for p in panels[1:]:
            img = np.hstack([img, sep, p])
    else:
        sep = np.full((6, panels[0].shape[1], 3), 255, np.uint8)
        img = panels[0]
        for p in panels[1:]:
            img = np.vstack([img, sep, p])
    cv2.imwrite(str(out), img)


def load_apo(prob_path, ppm):
    ap = cv2.imread(str(prob_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    return find_aponeuroses(ap, ppm) or find_aponeuroses(ap, ppm, thr=0.3)


def _ext(work, kind):
    c = pd.read_csv(work / "ext_ens/crops.csv"); f = pd.read_csv(work / "ext_ens/features.csv")
    L = pd.read_csv(work / "ext_ens/labels.csv")
    f["image_id"] = c.image_id.values
    d = c.merge(f[["image_id", "fasc_sign"]], on="image_id").merge(L, on="image_id")
    return d[(d.family == kind) & d.fasc_sign.notna() & d.fl_mm.notna()]


def validation(a, kind):
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = _ext(work, kind)
    spec = importlib.util.spec_from_file_location("eb", UMUD / "scripts/external_bench.py")
    eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
    its = {i: (p, box) for i, s, p, _, box in eb.items(work / "osf", work / "osf/neuage_expert.csv") if s == kind}
    rng = random.Random(11 if kind == "osf" else 5)
    ids = list(d.image_id) if kind == "osf" else rng.sample(list(d.image_id), 32)
    key, pre = {}, "q" if kind == "osf" else "n"
    for k, iid in enumerate(ids):
        r = d[d.image_id == iid].iloc[0]
        p, (l, t, rr, b) = its[iid]
        g = S.read_gray(str(p))[t:b, l:rr]
        apo = load_apo(work / "ext_ens/probs" / f"{iid}_apo.png", r.px_per_mm)
        if apo is None or g.shape != (apo["H"], apo["W"]):
            continue
        fac = rng.choice([0.78, 1.28])
        cands, truth = [r.fl_mm, r.fl_mm * fac], "A"
        if rng.random() < 0.5:
            cands, truth = cands[::-1], "B"
        render(g, apo, r.fasc_sign, r.px_per_mm, cands, out / f"{pre}{k:02d}.png", nlines=5 if kind == "osf" else 4)
        key[f"{pre}{k:02d}"] = dict(image_id=iid, truth=truth, fac=fac, cands=[float(x) for x in cands])
    json.dump(key, open(out / "key.json", "w"))
    print("rendered", len(key))


def test(a):
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v = pd.read_csv(UMUD / "daily/inputs/ref_vera_public.csv"); p = pd.read_csv(UMUD / a.pipeline)
    f = pd.read_csv(UMUD / a.features); g = np.load(UMUD / "daily/inputs/test_groups.npy", allow_pickle=True)
    c = pd.read_csv(work / "seg/crops.csv").set_index("image_id")
    d = pd.DataFrame(dict(image_id=v.image_id, grp=g, flv=v.fl_mm, flp=p.fl_mm, ok=f.ok, sign=f.fasc_sign))
    rng = random.Random(a.seed)
    units = []
    for gid, m in d.groupby("grp"):
        flv, flp = m.flv.median(), m.flp.median()
        if (m.ok != 1).any() or m.sign.isna().all() or abs(flp - flv) <= a.min_diff:
            continue
        units.append(dict(grp=int(gid), image_id=m.iloc[len(m) // 2].image_id, members=list(m.image_id),
                          flv=float(flv), flp=float(flp), sign=float(np.sign(m.sign.median()))))
    rng.shuffle(units)
    key = {}
    for k, u in enumerate(units):
        iid = u["image_id"]; cr = c.loc[iid]
        gray = S.read_gray(str(work / "data/test_images_v2/test_set_v2" / iid))[int(cr.t):int(cr.b), int(cr.l):int(cr.r)]
        apo = load_apo(work / "seg/probs" / f"{iid}_apo.png", cr.px_per_mm)
        if apo is None:
            continue
        pipe_first = rng.random() < 0.5
        cands = [u["flp"], u["flv"]] if pipe_first else [u["flv"], u["flp"]]
        render(gray, apo, u["sign"], cr.px_per_mm, cands, out / f"t{k:02d}.png")
        key[f"t{k:02d}"] = dict(u, pipe="A" if pipe_first else "B")
    json.dump(key, open(out / "key.json", "w"))
    print("units rendered", len(key))


def score(a):
    key = json.load(open(Path(a.out) / "key.json"))
    rows = [r for r in csv.reader(open(a.answers)) if r and r[1] != "X"]
    if "truth" in next(iter(key.values())):
        ok = pd.DataFrame([(q, ans == key[q]["truth"], int(c)) for q, ans, c in rows], columns=["q", "ok", "conf"])
        print(f"accuracy {ok.ok.mean():.3f} (n={len(ok)})"); print(ok.groupby("conf").ok.agg(["size", "mean"]))
    else:
        for q, ans, c in rows:
            u = key[q]
            print(q, u["image_id"], f"pipe {u['flp']:.1f} ref {u['flv']:.1f}", "pick",
                  "pipe" if ans == u["pipe"] else "ref", "conf", c)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["osf", "neuage", "test", "score"])
    ap.add_argument("--work", default="/home/user/work")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pipeline", default="daily/inputs/pipeline_ens.csv")
    ap.add_argument("--features", default="daily/inputs/features_ens.csv")
    ap.add_argument("--min-diff", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--answers", default=None)
    a = ap.parse_args()
    {"osf": lambda: validation(a, "osf"), "neuage": lambda: validation(a, "neuage"), "test": lambda: test(a),
     "score": lambda: score(a)}[a.cmd]()
