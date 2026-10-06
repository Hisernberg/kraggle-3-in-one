"""Render test images with the pipeline's aponeurosis / fascicle detections for visual QA.

    python scripts/overlay.py --data <competition dir> --seg <seg out dir> --ids IMG_00003.tif IMG_00120.tif \
        --sub submissions/d7_S2_fl031.csv --ref daily/inputs/ref_vera_public.csv --out work/overlay

Draws, on the full-resolution image: the B-mode crop box, superficial (green) and deep (red) aponeurosis inner-edge
fits, every kept fascicle fragment line (yellow), a 50 px grid (grey, every 100 px labelled) and the predictions.
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umud.geometry import find_aponeuroses  # noqa: E402


def fascicle_segments(fp, apo, ppm, thr=0.35):
    H, W = fp.shape
    s, d = apo["sup"], apo["deep"]
    xs = np.arange(W)
    y_sup = np.polyval(s["bot"] if s["bot"] is not None else s["cen"], xs)
    y_deep = np.polyval(d["top"] if d["top"] is not None else d["cen"], xs)
    yy = np.arange(H)[:, None]
    region = (yy > y_sup[None] + 0.5 * ppm) & (yy < y_deep[None] - 0.5 * ppm)
    p = np.where(region, fp, 0).astype(np.float32)
    n, lab, st, _ = cv2.connectedComponentsWithStats((p >= thr).astype(np.uint8), 8)
    segs = []
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
        half = 2 * math.sqrt(max(ev[-1], 0))
        if 2 * half < 3 * ppm:
            continue
        segs.append((mu - half * v, mu + half * v))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--sub", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="resize factor of the written PNG")
    a = ap.parse_args()
    seg = Path(a.seg)
    crops = pd.read_csv(seg / "crops.csv").set_index("image_id")
    sub = pd.read_csv(a.sub).set_index("image_id") if a.sub else None
    ref = pd.read_csv(a.ref).set_index("image_id") if a.ref else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tdir = Path(a.data) / "test_images_v2" / "test_set_v2"
    for iid in a.ids:
        img = cv2.imread(str(tdir / iid), cv2.IMREAD_COLOR)
        c = crops.loc[iid]
        l, t = int(c.l), int(c.t)
        ap_ = cv2.imread(str(seg / "probs" / f"{iid}_apo.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        fp = cv2.imread(str(seg / "probs" / f"{iid}_fasc.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        vis = img.copy()
        H, W = vis.shape[:2]
        for x in range(0, W, 50):
            cv2.line(vis, (x, 0), (x, H), (90, 90, 90) if x % 100 else (140, 140, 140), 1)
        for y in range(0, H, 50):
            cv2.line(vis, (0, y), (W, y), (90, 90, 90) if y % 100 else (140, 140, 140), 1)
        for x in range(0, W, 100):
            cv2.putText(vis, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        for y in range(100, H, 100):
            cv2.putText(vis, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.rectangle(vis, (l, t), (int(c.r), int(c.b)), (255, 0, 255), 1)
        apo = find_aponeuroses(ap_, c.px_per_mm) or find_aponeuroses(ap_, c.px_per_mm, thr=0.3)
        if apo is not None:
            w = apo["W"]
            xs = np.arange(0, w, 4)
            for key, line, col in (("sup", "bot", (0, 255, 0)), ("deep", "top", (0, 0, 255))):
                cf = apo[key][line] if apo[key][line] is not None else apo[key]["cen"]
                pts = np.stack([xs + l, np.polyval(cf, xs) + t], 1).astype(np.int32)
                cv2.polylines(vis, [pts], False, col, 1)
            for p0, p1 in fascicle_segments(fp, apo, c.px_per_mm):
                cv2.line(vis, (int(p0[0] + l), int(p0[1] + t)), (int(p1[0] + l), int(p1[1] + t)), (0, 255, 255), 1)
        txt = [f"{iid} {c.family} {c.px_per_mm:.2f}px/mm"]
        if sub is not None:
            s = sub.loc[iid]
            txt.append(f"best PA {s.pa_deg:.1f} FL {s.fl_mm:.1f} MT {s.mt_mm:.1f}")
        if ref is not None:
            s = ref.loc[iid]
            txt.append(f"ref  PA {s.pa_deg:.1f} FL {s.fl_mm:.1f} MT {s.mt_mm:.1f}")
        for i, s in enumerate(txt):
            cv2.putText(vis, s, (60, H - 50 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if a.scale != 1.0:
            vis = cv2.resize(vis, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"{Path(iid).stem}.png"), vis)


if __name__ == "__main__":
    main()
