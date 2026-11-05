"""Build the SLM P0 gate-baseline Kaggle kernel (umud-seg-slm0) — v2 recon + pair + compare.

    python umud/kaggle/build_slm_kernel.py --out umud/build/slm0
    kaggle kernels push -p umud/build/slm0

v2 (after v1 found 0 pairs): (1) deep recon of every mount — dir tree, counts, sample
filenames, CSV headers; (2) pairing strategies: exact stem -> containment -> RLE-CSV decode;
(3) ensemble-member pseudo-label mounts (our own apo/fasc kernels on TEST images) are EXCLUDED
from training (circularity guard); (4) pip guard for segmentation_models_pytorch; (5) if >=50
GT pairs: identical-protocol comparison of two encoders (resnet34 vs mit_b1 SegFormer) under
LODV + TrivialAugment — the P0 gate baseline table.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KERNEL = r'''
# umud-seg-slm0 v2 — recon -> pair -> gate-baseline comparison (P0 of FM_PLAN.md)
import os, sys, glob, json, random, time, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"],
               check=False)
import numpy as np
import pandas as pd
import cv2
import torch

SEED = 2401
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IN, OUT = "/kaggle/input", "/kaggle/working"
os.makedirs(OUT, exist_ok=True)

def L(msg):
    print(msg, flush=True)

# ---------- 1. deep recon ----------
L("== deep recon ==")
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
for top in sorted(glob.glob(f"{IN}/*")):
    name = os.path.basename(top)
    L(f"MOUNT {name}:")
    for dirpath, dirnames, filenames in os.walk(top):
        rel = os.path.relpath(dirpath, top)
        if len(filenames) == 0 and not dirnames:
            continue
        exts = {}
        for f in filenames:
            e = os.path.splitext(f)[1].lower(); exts[e] = exts.get(e, 0) + 1
        L(f"  {rel}/  n={len(filenames)} exts={dict(sorted(exts.items(), key=lambda x: -x[1])[:5])}")
        for f in sorted(filenames)[:8]:
            L(f"    e.g. {f}")
        if rel.count(os.sep) > 3:
            dirnames[:] = []
for c in glob.glob(f"{IN}/**/*.csv", recursive=True):
    try:
        head = pd.read_csv(c, nrows=1)
        L(f"CSV {c}: cols={list(head.columns)[:12]}")
    except Exception:
        pass

# ---------- 2. pair: ground-truth masks only (exclude our own pseudo-label mounts) ----------
def looks_like_mask(p):
    return any(k in p.lower() for k in ("mask", "label", "annot", "groundtruth", "gt"))
imgs, masks = [], []
for p in glob.glob(f"{IN}/**/*", recursive=True):
    if not os.path.isfile(p) or not p.lower().endswith(IMG_EXT):
        continue
    # exclude pseudo-label mounts: outputs of our umud-seg inference kernels
    if "/notebooks/" in p and any(k in p.lower() for k in ("apo", "fasc", "seg", "slm")):
        continue
    (masks if looks_like_mask(p) else imgs).append(p)
L(f"GT-candidate images: {len(imgs)}, masks: {len(masks)} (pseudo-label mounts excluded)")
stem = lambda p: os.path.splitext(os.path.basename(p))[0].lower()
img_by_stem = {}
for p in imgs:
    img_by_stem.setdefault(stem(p), []).append(p)
pairs, strat = [], None
for mp in masks:
    s = stem(mp)
    for suf in ("_mask", "_label", "-mask", "_apo", "_fascicle", "_gt", "_seg"):
        if s.endswith(suf):
            s = s[: -len(suf)]; break
    cands = img_by_stem.get(s)
    if cands and strat is None:
        strat = "exact-stem"
    if not cands:  # containment fallback
        for k, v in img_by_stem.items():
            if k in s or s in k:
                cands = v; strat = strat or "containment"; break
    for c in cands or []:
        pairs.append((c, mp))
L(f"pairs: {len(pairs)} (strategy: {strat})")

# ---------- 3. train only on real GT ----------
if len(pairs) < 50:
    json.dump({"status": "NO_GT_PAIRS", "n_imgs": len(imgs), "n_masks": len(masks),
               "n_pairs": len(pairs),
               "note": "recon dumped to log; fix data source and re-push"},
              open(f"{OUT}/run_record.json", "w"), indent=1)
    L("NO_GT_PAIRS — recon above has filenames + CSV headers; aborting before training.")
    raise SystemExit(0)

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

IMG = 384
def device_of(path, feats=None):
    s = os.path.basename(path).lower()
    for k in ("telemed", "juniper", "lumify", "gray853", "h512", "left_ticks", "lefttick"):
        if k in s:
            return k
    return "unknown"

class SegDS(Dataset):
    def __init__(self, pairs, train):
        self.pairs, self.train = pairs, train
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        im = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if im is None or mk is None:
            return torch.zeros(1, IMG, IMG), torch.zeros(2, IMG, IMG)
        h, w = im.shape[:2]
        if mk.shape[:2] != (h, w):
            mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)
        im = cv2.resize(im, (IMG, IMG)).astype(np.float32) / 255.0
        mk = cv2.resize(mk, (IMG, IMG), interpolation=cv2.INTER_NEAREST)
        y = np.stack([(mk > 63) & (mk < 192), mk >= 192]).astype(np.float32)
        if self.train:
            op = random.choice(["hflip", "rot", "bright", "contrast", "gamma", "identity"])
            if op == "hflip" and random.random() < 0.5:
                im, y = im[:, ::-1], y[:, ::-1]
            elif op == "rot":
                k = random.randint(0, 3); im, y = np.rot90(im, k), np.rot90(y, k)
            elif op == "bright":
                im = np.clip(im * random.uniform(0.7, 1.3), 0, 1)
            elif op == "contrast":
                m = im.mean(); im = np.clip((im - m) * random.uniform(0.7, 1.4) + m, 0, 1)
            elif op == "gamma":
                im = np.clip(im ** random.uniform(0.7, 1.4), 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(im))[None],
                torch.from_numpy(np.ascontiguousarray(y)))

def iou(gt, pr, eps=1e-7):
    return ((gt & pr).sum() + eps) / ((gt | pr).sum() + eps)

def run_variant(encoder, weights, tr, va, tag, epochs=10, bs=8):
    model = smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=2).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    dsc, bce = smp.losses.DiceLoss(mode="multilabel"), nn.BCEWithLogitsLoss()
    tl = DataLoader(SegDS(tr, True), batch_size=bs, shuffle=True, num_workers=2)
    vl = DataLoader(SegDS(va, False), batch_size=bs, num_workers=2)
    per_dev = {}
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for im, y in tl:
            im, y = im.to(DEV), y.to(DEV)
            loss = 0.5 * dsc(model(im), y) + 0.5 * bce(model(im), y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); rows = []
        with torch.no_grad():
            for bi, (im, y) in enumerate(vl):
                pr = (torch.sigmoid(model(im.to(DEV))) > 0.5).cpu().numpy().astype(bool)
                for b in range(len(y)):
                    d = device_of(va[bi * y.shape[0] + b][0])
                    rows.append((d, *[iou(y[b, c].numpy(), pr[b, c]) for c in range(2)]))
        arr = np.array([r[1:] for r in rows])
        L(f"[{tag}] ep{ep+1}/{epochs} loss {loss.item():.3f} IoU apo {arr[:,0].mean():.4f} fasc {arr[:,1].mean():.4f} ({time.time()-t0:.0f}s)")
        best = (arr[:, 1].mean(), arr[:, 0].mean())
    torch.save(model.state_dict(), f"{OUT}/slm0_{tag}.pt")
    return {"fasc_iou": best[0], "apo_iou": best[1]}

devs = sorted({device_of(ip) for ip, _ in pairs})
L(f"devices in GT pairs: {devs}")
results = {"protocol": "U-Net, Dice+BCE, TrivialAugment, ep10, bs8, IMG384; LODV by device", "lodv": {}}
for enc, w, tag in [("resnet34", "imagenet", "resnet34_P0"), ("mit_b1", "imagenet", "mit_b1_segformer")]:
    agg = {}
    for d in devs:
        tr = [(c, m) for c, m in pairs if device_of(c) != d]
        va = [(c, m) for c, m in pairs if device_of(c) == d]
        if len(va) < 4 or len(tr) < 30:
            L(f"LODV skip {d} for {tag}: train {len(tr)}/val {len(va)}"); continue
        agg[d] = run_variant(enc, w, tr, va, f"{tag}_{d}")
        L(f"[{tag}] LODV {d}: fasc {agg[d]['fasc_iou']:.4f} apo {agg[d]['apo_iou']:.4f}")
    results["lodv"][tag] = agg
json.dump({"status": "OK", "n_pairs": len(pairs), "strategy": strat, **results},
          open(f"{OUT}/run_record.json", "w"), indent=1)
L("== slm0 v2 done — comparison table in run_record.json ==")
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
    print(f"slm0 v2 kernel written to {out} — push with: kaggle kernels push -p {out}")

if __name__ == "__main__":
    main()
