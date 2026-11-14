"""Build the SLM bridge kernel (umud-seg-slm0 v4): winner -> test masks -> PA/FL/MT member CSV.

    python umud/kaggle/build_slm0_bridge.py --out umud/build/slm0
    kaggle kernels push -p umud/build/slm0

P3 bridge of FM_PLAN.md: (1) pair GT train masks; (2) train resnet34 + mit_b1 under identical
protocol (85/15 seeded holdout if no device tags; LODV otherwise); (3) pick winner by val
fascicle IoU; (4) predict apo+fascicle masks on all IMG_*.tif test images with hflip TTA;
(5) geometry layer: aponeurosis line fits, fascicle PCA, MT = aponeurosis separation,
mm conversion via the umud-test-scales dataset (calib_px_per_10mm from bar detection);
(6) if a train annotation CSV with pa/fl/mt exists, run the same geometry on held-out GT
images and report end-to-end MAE (px). Outputs slm_member.csv + run_record.json.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KERNEL = r'''
# umud-seg-slm0 v4 — SLM bridge: comparison -> winner -> test-image PA/FL/MT member
import os, sys, glob, json, random, time, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=False)
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 2401
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IN, OUT = "/kaggle/input", "/kaggle/working"
IMG = 384

def L(m): print(m, flush=True)

def looks_like_mask(p):
    return any(k in p.lower() for k in ("mask", "label", "annot", "gt"))

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
imgs, masks = [], []
for p in glob.glob(f"{IN}/**/*", recursive=True):
    if not os.path.isfile(p) or not p.lower().endswith(IMG_EXT):
        continue
    if "/notebooks/" in p and any(k in p.lower() for k in ("apo", "fasc", "seg", "slm")):
        continue
    (masks if looks_like_mask(p) else imgs).append(p)
stem = lambda p: os.path.splitext(os.path.basename(p))[0].lower()
by = {}
for p in imgs: by.setdefault(stem(p), []).append(p)
pairs = []
for mp in masks:
    s = stem(mp)
    for suf in ("_mask", "_label", "-mask", "_apo", "_fascicle", "_gt", "_seg"):
        if s.endswith(suf): s = s[: -len(suf)]; break
    c = by.get(s) or next((v for k, v in by.items() if k in s or s in k), [])
    pairs += [(x, mp) for x in c]
L(f"GT pairs: {len(pairs)}; image-like files: {len(imgs)}")
if len(pairs) < 50:
    json.dump({"status": "NO_GT_PAIRS", "n_pairs": len(pairs)}, open(f"{OUT}/run_record.json", "w"), indent=1)
    raise SystemExit(0)

def decode_y(mk):
    """Instance/level-coded masks (observed: muscle body at one mid gray, fascicles at distinct
    levels, background 0/1) -> two geometric channels: apo envelope band + interior fascicles.
    v7: muscle body = the component with the largest VERTICAL SPAN (not max area — area picked
    thin sub-structures in v6 and collapsed MT)."""
    st = (mk > 10).astype(np.uint8)
    if st.sum() < 100:
        return np.zeros((2,) + mk.shape, np.float32), 0.0
    k = np.ones((25, 25), np.uint8)
    closed = cv2.morphologyEx(st, cv2.MORPH_CLOSE, k)
    n, lab = cv2.connectedComponents(closed)
    if n > 1:
        best, span = 0, -1
        for i in range(1, n):
            ys = np.where((lab == i).any(axis=1))[0]
            s = (ys.max() - ys.min()) if len(ys) else -1
            if s > span:
                span, best = s, i
        fill = (lab == best).astype(np.uint8)
    else:
        fill = closed
    fill = cv2.morphologyEx(fill, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    band = cv2.dilate(fill, np.ones((7, 7), np.uint8)) - cv2.erode(fill, np.ones((7, 7), np.uint8))
    fasc = (st & ~cv2.dilate(band, np.ones((9, 9), np.uint8))).astype(np.float32)
    y = np.stack([band.astype(np.float32), fasc])
    return y, float(fasc.sum())

def roi_bbox(mk):
    ys, xs = np.where(mk > 10)
    if len(ys) == 0:
        return None
    h, w = mk.shape
    y0, y1 = max(0, ys.min() - 24), min(h, ys.max() + 24)
    x0, x1 = max(0, xs.min() - 24), min(w, xs.max() + 24)
    if (y1 - y0) < 64 or (x1 - x0) < 64:
        return None
    return y0, y1, x0, x1

class SegDS(Dataset):
    def __init__(self, pairs, train): self.pairs, self.train = pairs, train
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        im = cv2.imread(ip, cv2.IMREAD_GRAYSCALE); mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if im is None or mk is None: return torch.zeros(1, IMG, IMG), torch.zeros(2, IMG, IMG)
        h, w = im.shape[:2]
        if mk.shape[:2] != (h, w): mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)
        bb = roi_bbox(mk)
        if bb is not None:
            y0, y1, x0, x1 = bb
            im, mk = im[y0:y1, x0:x1], mk[y0:y1, x0:x1]
        im = cv2.resize(im, (IMG, IMG)).astype(np.float32) / 255.0
        mk = cv2.resize(mk, (IMG, IMG), interpolation=cv2.INTER_NEAREST)
        y, fasc_px = decode_y(mk)
        if self.train:
            op = random.choice(["hflip", "rot", "bright", "contrast", "gamma", "identity"])
            if op == "hflip" and random.random() < 0.5:
                im, y = im[..., ::-1].copy(), y[..., ::-1].copy()
            elif op == "rot":
                k = random.randint(0, 3)
                im, y = np.rot90(im, k, axes=(-2, -1)).copy(), np.rot90(y, k, axes=(-2, -1)).copy()
            elif op == "bright": im = np.clip(im * random.uniform(0.7, 1.3), 0, 1)
            elif op == "contrast":
                m = im.mean(); im = np.clip((im - m) * random.uniform(0.7, 1.4) + m, 0, 1)
            elif op == "gamma": im = np.clip(im ** random.uniform(0.7, 1.4), 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(im.astype(np.float32)))[None],
                torch.from_numpy(np.ascontiguousarray(y.astype(np.float32))))

def iou(g, p, eps=1e-7):
    g, p = np.asarray(g).astype(bool), np.asarray(p).astype(bool)
    if g.sum() == 0 and p.sum() == 0:
        return 1.0
    return ((g & p).sum() + eps) / ((g | p).sum() + eps)

def evaluate(model, va):
    model.eval(); rows = []
    with torch.no_grad():
        for bi in range(0, len(va), 8):
            chunk = va[bi: bi + 8]
            ims = torch.stack([SegDS([c], False)[0][0] for c in chunk]).to(DEV)
            pr = (torch.sigmoid(model(ims)) > 0.5).cpu().numpy().astype(bool)
            for b, (ip, mp) in enumerate(chunk):
                _, y = SegDS([(ip, mp)], False)[0]
                rows.append([iou(y[c].numpy(), pr[b, c]) for c in range(2)])
    return np.array(rows)

def train(model, tr, va, tag, epochs=20, bs=8):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    dsc = smp.losses.DiceLoss(mode="multilabel"); bce = nn.BCEWithLogitsLoss()
    scl = torch.cuda.amp.GradScaler()
    tl = DataLoader(SegDS(tr, True), batch_size=bs, shuffle=True, num_workers=0)
    hist = []
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for im, y in tl:
            im, y = im.to(DEV), y.to(DEV)
            with torch.cuda.amp.autocast():
                loss = 0.5 * dsc(model(im), y) + 0.5 * bce(model(im), y)
            opt.zero_grad(); scl.scale(loss).backward(); scl.step(opt); scl.update()
        arr = evaluate(model, va)
        hist.append([float(loss.item()), float(arr[:, 0].mean()), float(arr[:, 1].mean()), time.time() - t0])
        L(f"[{tag}] ep{ep+1}/{epochs} loss {hist[-1][0]:.3f} IoU apo {hist[-1][1]:.4f} fasc {hist[-1][2]:.4f} ({hist[-1][3]:.0f}s)")
    torch.save(model.state_dict(), f"{OUT}/slm0_{tag}.pt")
    return hist[-1]

import segmentation_models_pytorch as smp

idx = list(range(len(pairs))); random.Random(2401).shuffle(idx)
nva = max(50, len(idx) // 7)
tr = [pairs[i] for i in idx[nva:]]; va = [pairs[i] for i in idx[:nva]]
L(f"holdout split: train {len(tr)} / val {len(va)}")
scores, models = {}, {}
for enc, tag in [("resnet34", "resnet34"), ("mit_b1", "mit_b1")]:
    try:
        m = smp.Unet(enc, encoder_weights="imagenet", in_channels=1, classes=2).to(DEV)
        h = train(m, tr, va, tag, epochs=20)
        scores[tag] = {"apo_iou": h[1], "fasc_iou": h[2], "loss": h[0]}
        models[tag] = m
    except Exception as e:
        scores[tag] = {"error": f"{type(e).__name__}: {e}"}
        L(f"arm {tag} FAILED: {e}")
ok = {k: v for k, v in scores.items() if "fasc_iou" in v}
if not ok:
    json.dump({"status": "ALL_ARMS_FAILED", "scores": scores}, open(f"{OUT}/run_record.json", "w"), indent=1)
    raise SystemExit(0)
winner = max(ok, key=lambda k: (ok[k]["fasc_iou"], ok[k]["apo_iou"]))
L(f"WINNER: {winner} ({ok[winner]})")
model = models[winner]

# ---------- geometry ----------
def apo_lines(mk):
    """Fit superficial (top) and deep (bottom) aponeurosis lines from the apo (gray) band."""
    cols, tops, bots = [], [], []
    for x in range(0, mk.shape[1], 4):
        ys = np.where(mk[:, x])[0]
        if len(ys) >= 3:
            cols.append(x); tops.append(ys.min()); bots.append(ys.max())
    if len(cols) < 20:
        return None, None
    tops, bots, cols = map(np.array, (tops, bots, cols))
    kt, bt = np.polyfit(cols, tops, 1)
    kb, bb = np.polyfit(cols, bots, 1)
    return (kt, bt), (kb, bb)

def fasc_stats(mk, apo_line):
    """Per fascicle component: angle vs aponeurosis line and principal-axis length (px)."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mk.astype(np.uint8), 8)
    ka, ba = apo_line
    th_apo = np.arctan(ka)
    pas, fls = [], []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 60:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 60:
            continue
        xc, yc = xs - xs.mean(), ys - ys.mean()
        cov = np.cov(np.stack([xc, yc]))
        w, v = np.linalg.eigh(cov)
        major = v[:, np.argmax(w)]
        th_f = np.arctan2(major[1], major[0])
        pa = abs(th_f - th_apo)
        pa = min(pa, np.pi - pa)  # fascicles cross the apo at an acute angle
        pas.append(np.degrees(pa))
        fls.append(4.0 * np.sqrt(max(w.max(), 1e-6)))  # principal-axis full length approx
    return (float(np.median(pas)) if pas else None,
            float(np.median(fls)) if fls else None,
            len(pas))

def img_bbox(im):
    """ROI for unlabeled frames: the ultrasound fan has intensity, background is near-black."""
    h, w = im.shape
    col_ok = np.where(im.mean(axis=0) > 0.08 * im.max())[0]
    row_ok = np.where(im.mean(axis=1) > 0.08 * im.max())[0]
    if len(col_ok) < 64 or len(row_ok) < 64:
        return None
    x0, x1 = max(0, col_ok.min() - 24), min(w, col_ok.max() + 24)
    y0, y1 = max(0, row_ok.min() - 24), min(h, row_ok.max() + 24)
    if (y1 - y0) < 64 or (x1 - x0) < 64:
        return None
    return y0, y1, x0, x1

def predict_mask(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None, None
    H, W = im.shape
    bb = img_bbox(im)
    if bb is None:
        crop = cv2.resize(im, (IMG, IMG)).astype(np.float32) / 255.0
        off = (0, 0)
        cw, ch = W, H
    else:
        y0, y1, x0, x1 = bb
        off = (x0, y0)
        cw, ch = x1 - x0, y1 - y0
        crop = cv2.resize(im[y0:y1, x0:x1], (IMG, IMG)).astype(np.float32) / 255.0
    t = torch.from_numpy(crop[None, None]).to(DEV)
    with torch.no_grad():
        p = torch.sigmoid(model(t))[0].cpu().numpy()
        p += torch.sigmoid(model(torch.flip(t, dims=[-1])))[0].cpu().numpy()[..., ::-1]  # hflip TTA
        p /= 2
    canvas = np.zeros((2, H, W), np.float32)
    for c in range(2):
        canvas[c, off[1]:off[1] + ch, off[0]:off[0] + cw] = cv2.resize(p[c], (cw, ch))
    band = (canvas[0] > 0.5).astype(np.uint8)
    fasc = (canvas[1] > 0.5).astype(np.uint8)
    return im, band, fasc

def geometry_from(band, fasc):
    """Per-frame geometry from (band, fasc) masks: returns dict(pa_deg, fl_px, mt_px)."""
    sup, deep = apo_lines(band)
    out = {"pa_deg": np.nan, "fl_px": np.nan, "mt_px": np.nan}
    if deep is None:
        return out
    pa, fl_px, _ = fasc_stats(fasc, deep)
    if pa is not None:
        out["pa_deg"], out["fl_px"] = pa, fl_px
    if sup is not None:
        xs = np.arange(int(band.shape[1] * .2), int(band.shape[1] * .8), 8)
        sep = np.abs((sup[0] * xs + sup[1]) - (deep[0] * xs + deep[1]))
        out["mt_px"] = float(np.median(sep))
    return out

# end-to-end gate on the holdout: geometry from GT mask vs geometry from predicted mask
gate_rows = []
rng_gate = random.Random(77)
for ip, mp in rng_gate.sample(va, min(120, len(va))):
    mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if mk is None:
        continue
    gt = decode_y(mk)[0] > 0.5
    g_gt = geometry_from(gt[0].astype(np.uint8), gt[1].astype(np.uint8))
    out = predict_mask(ip)
    if out[0] is None:
        continue
    _, band_p, fasc_p = out
    g_p = geometry_from(band_p, fasc_p)
    gate_rows.append({k: g_p[k] - g_gt[k] for k in g_gt})
gate_arr = pd.DataFrame(gate_rows)
gate = {f"{c}_mae": float(gate_arr[c].abs().mean()) for c in gate_arr.columns} if len(gate_arr) else {}
gate["n"] = len(gate_arr)
L(f"GATE (pred-geometry vs GT-mask-geometry, holdout): {gate}")

# scales for test images
scales = {}
for c in glob.glob(f"{IN}/**/umud_test_scales.csv", recursive=True):
    s = pd.read_csv(c)
    scales = dict(zip(s.image_id, s.calib_px_per_10mm))
    L(f"scales loaded: {len(scales)} from {c}")
med_scale = float(np.median(list(scales.values()))) if scales else 125.0

member, geoms = {}, []
test_imgs = [p for p in imgs if os.path.basename(p).upper().startswith("IMG_")]
L(f"test images (IMG_*): {len(test_imgs)}")
t0 = time.time()
for j, ip in enumerate(sorted(test_imgs)):
    out = predict_mask(ip)
    if out[0] is None:
        continue
    im, band, fasc = out
    g = geometry_from(band, fasc)
    cal = float(scales.get(os.path.basename(ip), med_scale))
    mm_per_px = 10.0 / cal
    row = {"image_id": os.path.basename(ip), "pa_deg": g["pa_deg"],
           "fl_mm": g["fl_px"] * mm_per_px if np.isfinite(g["fl_px"]) else np.nan,
           "mt_mm": g["mt_px"] * mm_per_px if np.isfinite(g["mt_px"]) else np.nan}
    geoms.append({"image_id": row["image_id"], "pa": row["pa_deg"], "fl_mm": row["fl_mm"],
                  "mt_mm": row["mt_mm"]})
    member[row["image_id"]] = row
    if (j + 1) % 50 == 0:
        L(f"predicted {j+1}/{len(test_imgs)} ({time.time()-t0:.0f}s)")
mdf = pd.DataFrame(list(member.values()))
mdf.to_csv(f"{OUT}/slm_member.csv", index=False)
pd.DataFrame(geoms).to_csv(f"{OUT}/geometry_detail.csv", index=False)
L(f"member CSV: {len(mdf)} rows; PA non-null {mdf.pa_deg.notna().sum()}, FL {mdf.fl_mm.notna().sum()}, MT {mdf.mt_mm.notna().sum()}")
L(mdf[["pa_deg", "fl_mm", "mt_mm"]].describe().to_string())

json.dump({"status": "OK", "winner": winner, "scores": scores, "n_pairs": len(pairs),
           "n_test_predicted": len(mdf), "gate": gate,
           "scales_source": "umud-test-scales (bar calib)",
           "note": "v7: column-span fill + ROI crop + AMP + 20 epochs + GT-geometry gate"},
          open(f"{OUT}/run_record.json", "w"), indent=1)
L("== slm0 v4 bridge done ==")
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "slm0"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(KERNEL)
    meta = {
        "id": "kragglenote2forwork/umud-seg-slm0",
        "title": "umud-seg-slm0",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "competition_sources": ["umud-challenge-muscle-architecture-in-ultrasound-data"],
        "dataset_sources": ["kragglenote2forwork/umud-test-scales"],
        "kernel_sources": ["kragglenote2forwork/umud-seg-train"],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"bridge kernel written to {out} — push with: kaggle kernels push -p {out}")

if __name__ == "__main__":
    main()
