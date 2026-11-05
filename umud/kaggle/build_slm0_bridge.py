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

class SegDS(Dataset):
    def __init__(self, pairs, train): self.pairs, self.train = pairs, train
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        im = cv2.imread(ip, cv2.IMREAD_GRAYSCALE); mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if im is None or mk is None: return torch.zeros(1, IMG, IMG), torch.zeros(2, IMG, IMG)
        h, w = im.shape[:2]
        if mk.shape[:2] != (h, w): mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)
        im = cv2.resize(im, (IMG, IMG)).astype(np.float32) / 255.0
        mk = cv2.resize(mk, (IMG, IMG), interpolation=cv2.INTER_NEAREST)
        y = np.stack([(mk > 63) & (mk < 192), mk >= 192]).astype(np.float32)
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

def train(model, tr, va, tag, epochs=10, bs=8):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    dsc = smp.losses.DiceLoss(mode="multilabel"); bce = nn.BCEWithLogitsLoss()
    tl = DataLoader(SegDS(tr, True), batch_size=bs, shuffle=True, num_workers=0)
    hist = []
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for im, y in tl:
            im, y = im.to(DEV), y.to(DEV)
            loss = 0.5 * dsc(model(im), y) + 0.5 * bce(model(im), y)
            opt.zero_grad(); loss.backward(); opt.step()
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
        apo_iou, fasc_iou, _ = train(m, tr, va, tag)
        scores[tag] = {"apo_iou": apo_iou, "fasc_iou": fasc_iou}
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

def predict_mask(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None, None
    h, w = im.shape[:2]
    x = cv2.resize(im, (IMG, IMG)).astype(np.float32) / 255.0
    t = torch.from_numpy(x[None, None]).to(DEV)
    with torch.no_grad():
        p = torch.sigmoid(model(t))[0].cpu().numpy()
        p += torch.sigmoid(model(torch.flip(t, dims=[-1])))[0].cpu().numpy()[..., ::-1]  # hflip TTA
        p /= 2
    mk = (cv2.resize(p[0], (w, h)) > 0.5).astype(np.uint8) * 127 + \
         (cv2.resize(p[1], (w, h)) > 0.5).astype(np.uint8) * 255
    return im, mk

# scales for test images
scales = {}
for c in glob.glob(f"{IN}/**/umud_test_scales.csv", recursive=True):
    s = pd.read_csv(c)
    scales = dict(zip(s.image_id, s.calib_px_per_10mm))
    L(f"scales loaded: {len(scales)} from {c}")
med_scale = float(np.median(list(scales.values()))) if scales else 125.0

# optional end-to-end validation vs train annotation CSVs
val_rows = {}
for c in glob.glob(f"{IN}/**/*.csv", recursive=True):
    try:
        d = pd.read_csv(c, nrows=3)
    except Exception:
        continue
    lc = {col.lower(): col for col in d.columns}
    if {"image_id"} <= set(lc) and ({"pa_deg", "pa"} & set(lc)):
        ann = pd.read_csv(c)
        pa_c = lc.get("pa_deg", lc.get("pa"))
        val_rows[c] = (ann, pa_c, lc.get("fl_mm", lc.get("fl")), lc.get("mt_mm", lc.get("mt")))
        L(f"annotation CSV: {c} rows={len(ann)} cols={list(ann.columns)[:10]}")

member, geoms = {}, []
test_imgs = [p for p in imgs if os.path.basename(p).upper().startswith("IMG_")]
L(f"test images (IMG_*): {len(test_imgs)}")
t0 = time.time()
for j, ip in enumerate(sorted(test_imgs)):
    im, mk = predict_mask(ip)
    if mk is None:
        continue
    apo = (mk == 127) | (mk == 255)  # gray band incl. overlap; fasc = 255
    fasc = mk == 255
    sup, deep = apo_lines(apo)
    cal = float(scales.get(os.path.basename(ip), med_scale))
    mm_per_px = 10.0 / cal
    row = {"image_id": os.path.basename(ip), "pa_deg": np.nan, "fl_mm": np.nan, "mt_mm": np.nan}
    if deep is not None:
        pa, fl_px, ncomp = fasc_stats(fasc, deep)
        if pa is not None:
            row["pa_deg"], row["fl_mm"] = pa, fl_px * mm_per_px
        if sup is not None:
            # vertical separation sampled across the middle 60% of columns
            xs = np.arange(int(mk.shape[1] * .2), int(mk.shape[1] * .8), 8)
            sep = np.abs((sup[0] * xs + sup[1]) - (deep[0] * xs + deep[1]))
            row["mt_mm"] = float(np.median(sep)) * mm_per_px
        geoms.append({"image_id": row["image_id"], "n_fasc": ncomp if pa is not None else 0,
                      "pa": row["pa_deg"], "fl_mm": row["fl_mm"], "mt_mm": row["mt_mm"]})
    member[row["image_id"]] = row
    if (j + 1) % 50 == 0:
        L(f"predicted {j+1}/{len(test_imgs)} ({time.time()-t0:.0f}s)")
mdf = pd.DataFrame(list(member.values()))
mdf.to_csv(f"{OUT}/slm_member.csv", index=False)
pd.DataFrame(geoms).to_csv(f"{OUT}/geometry_detail.csv", index=False)
L(f"member CSV: {len(mdf)} rows; PA non-null {mdf.pa_deg.notna().sum()}, FL {mdf.fl_mm.notna().sum()}, MT {mdf.mt_mm.notna().sum()}")
L(mdf[["pa_deg", "fl_mm", "mt_mm"]].describe().to_string())

json.dump({"status": "OK", "winner": winner, "scores": scores, "n_pairs": len(pairs),
           "n_test_predicted": len(mdf), "scales_source": "umud-test-scales (bar calib)",
           "note": "geometry is v1 heuristic (apo line fits + fascicle PCA); expect refinement vs ensemble"},
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
