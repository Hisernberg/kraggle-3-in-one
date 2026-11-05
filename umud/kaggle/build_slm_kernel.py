"""Build the SLM P0 gate-baseline Kaggle kernel (umud-seg-slm0).

    python umud/kaggle/build_slm_kernel.py --out umud/build/slm0
    kaggle kernels push -p umud/build/slm0

P0 of FM_PLAN.md: the nnU-Net-style gate baseline every later stage must beat.
2D U-Net (segmentation_models_pytorch, ResNet-34 ImageNet init), 2 classes (aponeurosis,
fascicle), TrivialAugment, leave-one-device-out validation. Discovery-based: auto-finds
competition images and any mask dirs mounted from the umud-seg kernel lineage; logs a full
inventory and exits cleanly if no usable pairs exist (fix the mount, re-push).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KERNEL = r'''
# umud-seg-slm0 — P0 gate baseline (FM_PLAN.md): U-Net ResNet-34, TrivialAugment, LODV
import os, glob, json, random, time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 2401
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IN = "/kaggle/input"
OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)
log = []

def L(msg):
    print(msg, flush=True); log.append(msg)

# ---------- 1. inventory: find competition images and mask dirs ----------
L("== inventory ==")
comp_imgs = sorted(glob.glob(f"{IN}/*/**/*.png", recursive=True) + glob.glob(f"{IN}/*/**/*.jpg", recursive=True)
                   + glob.glob(f"{IN}/*/**/*.bmp", recursive=True))
def looks_like_mask(p):
    return any(k in p.lower() for k in ("mask", "label", "annot", "apo", "fasc", "gt"))
masks = [p for p in comp_imgs if looks_like_mask(p)]
images = [p for p in comp_imgs if not looks_like_mask(p)]
L(f"total pngs/jpgs: {len(comp_imgs)} | mask-like: {len(masks)} | image-like: {len(images)}")
for d in sorted(glob.glob(f"{IN}/*")):
    n = sum(len(f) for _, _, f in os.walk(d))
    L(f"  mount: {os.path.basename(d)} ({n} files)")

stem = lambda p: os.path.splitext(os.path.basename(p))[0]
img_by_stem = {}
for p in images:
    img_by_stem.setdefault(stem(p), []).append(p)
pairs = []
for mp in masks:
    s = stem(mp)
    for suf in ("_mask", "_label", "-mask", "_apo", "_fascicle", "_gt"):
        if s.endswith(suf):
            s = s[: -len(suf)]; break
    for c in img_by_stem.get(s, []):
        pairs.append((c, mp))
L(f"image-mask pairs found: {len(pairs)}")
if len(pairs) < 50:
    json.dump({"status": "NO_DATA", "n_images": len(images), "n_masks": len(masks),
               "n_pairs": len(pairs)}, open(f"{OUT}/run_record.json", "w"), indent=1)
    raise SystemExit("slm0: fewer than 50 pairs — inventory written to run_record.json; fix mount and re-push")

# ---------- 2. device families (filename heuristics + competition features if present) ----------
import re
def device_of(path):
    s = os.path.basename(path).lower()
    for k in ("telemed", "juniper", "lumify", "gray853", "h512", "left_ticks", "lefttick"):
        if k in s:
            return k
    return "unknown"
feats = None
for c in glob.glob(f"{IN}/*/**/*.csv", recursive=True):
    try:
        d = pd.read_csv(c, nrows=2)
        if "image_id" in d.columns and "family" in d.columns:
            feats = pd.read_csv(c).set_index("image_id"); L(f"features: {c}"); break
    except Exception:
        pass
def device_of_id(image_id):
    if feats is not None and image_id in feats.index:
        return str(feats.loc[image_id, "family"])
    return device_of(image_id)

# ---------- 3. dataset ----------
IMG = 384
class SegDS(Dataset):
    def __init__(self, pairs, train):
        self.pairs, self.train = pairs, train
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        im = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if im is None or mk is None:
            im = np.zeros((IMG, IMG), np.uint8); mk = np.zeros((IMG, IMG), np.uint8)
        h, w = im.shape[:2]
        mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST) if mk.shape[:2] != (h, w) else mk
        im = cv2.resize(im, (IMG, IMG)).astype(np.float32) / 255.0
        mk = cv2.resize(mk, (IMG, IMG), interpolation=cv2.INTER_NEAREST)
        y = np.stack([(mk > 63) & (mk < 192), mk >= 192]).astype(np.float32)  # ch0 apo (mid gray), ch1 fascicle
        if self.train:  # TrivialAugment-style: one op, random magnitude (Tupper & Gagne, TMLR 2025)
            op = random.choice(["hflip", "rot", "bright", "contrast", "gamma", "identity"])
            if op == "hflip" and random.random() < 0.5:
                im, y = im[:, ::-1], y[:, ::-1]
            elif op == "rot":
                k = random.randint(0, 3)
                im, y = np.rot90(im, k), np.rot90(y, k)
            elif op == "bright":
                im = np.clip(im * random.uniform(0.7, 1.3), 0, 1)
            elif op == "contrast":
                m = im.mean(); im = np.clip((im - m) * random.uniform(0.7, 1.4) + m, 0, 1)
            elif op == "gamma":
                g = random.uniform(0.7, 1.4); im = np.clip(im ** g, 0, 1)
        im = np.ascontiguousarray(im)[None]
        return torch.from_numpy(im), torch.from_numpy(np.ascontiguousarray(y))

import segmentation_models_pytorch as smp

def make_model(encoder="resnet34", weights="imagenet"):
    return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=2)

def iou_score(gt, pr, eps=1e-7):
    inter = (gt & pr).sum(); union = (gt | pr).sum()
    return (inter + eps) / (union + eps)

def train_one(tr_pairs, va_pairs, encoder="resnet34", weights="imagenet", epochs=12, bs=8, tag=""):
    model = make_model(encoder, weights).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    dsc = smp.losses.DiceLoss(mode="multilabel")
    bce = nn.BCEWithLogitsLoss()
    tl = DataLoader(SegDS(tr_pairs, True), batch_size=bs, shuffle=True, num_workers=2)
    vl = DataLoader(SegDS(va_pairs, False), batch_size=bs, num_workers=2)
    best = -1
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for im, y in tl:
            im, y = im.to(DEV), y.to(DEV)
            loss = 0.5 * dsc(model(im), y) + 0.5 * bce(model(im), y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); ious = []
        with torch.no_grad():
            for im, y in vl:
                p = (torch.sigmoid(model(im.to(DEV))) > 0.5).cpu().numpy().astype(bool)
                for b in range(len(y)):
                    ious.append([iou_score(y[b, c].numpy(), p[b, c]) for c in range(2)])
        ious = np.array(ious)
        L(f"[{tag}] ep{ep+1}/{epochs} loss {loss.item():.3f} IoU apo {ious[:,0].mean():.4f} fasc {ious[:,1].mean():.4f} ({time.time()-t0:.0f}s)")
        if ious[:, 1].mean() > best:
            best = ious[:, 1].mean()
            torch.save(model.state_dict(), f"{OUT}/slm0_{tag}.pt")
    return best

# ---------- 4. leave-one-device-out ----------
devs = sorted({device_of_id(stem(ip)) for ip, _ in pairs})
L(f"devices: {devs}")
results = {}
for d in devs:
    tr = [(c, m) for c, m in pairs if device_of_id(stem(c)) != d]
    va = [(c, m) for c, m in pairs if device_of_id(stem(c)) == d]
    if len(va) < 5 or len(tr) < 30:
        L(f"LODV skip {d}: train {len(tr)} / val {len(va)}"); continue
    results[d] = train_one(tr, va, tag=d.replace("/", "_"))
    L(f"LODV {d}: best fasc IoU {results[d]:.4f}")

json.dump({"status": "OK", "n_pairs": len(pairs), "devices": devs, "lodv_fasc_iou": results,
           "encoder": "resnet34/imagenet (P0 gate baseline — FM must beat this)"},
          open(f"{OUT}/run_record.json", "w"), indent=1)
L("== slm0 done — gate baseline recorded ==")
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
        "kernel_sources": ["kragglenote2forwork/umud-seg-train"],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"slm0 kernel written to {out} — push with: kaggle kernels push -p {out}")

if __name__ == "__main__":
    main()
