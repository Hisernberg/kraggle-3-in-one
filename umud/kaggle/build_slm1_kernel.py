"""Build the SLM P1 comparison kernel (umud-seg-slm1).

    python umud/kaggle/build_slm1_kernel.py --out umud/build/slm1
    kaggle kernels push -p umud/build/slm1

P1 of FM_PLAN.md: same GT pairs, same splits, same augmentation as slm0 v2 — three encoder arms:
  A. resnet34 + ImageNet           (P0 control)
  B. mit_b1 (SegFormer) + ImageNet (transformer arm, natively multi-scale)
  C. US-foundation init: runtime-probe HuggingFace for USFM / UltraSam weights; if loadable,
     ViT + light FPN seg head; else timm vit_small augreg_in21k + same FPN (generic-SSL arm).
Output run_record.json = the same-benchmark comparison table + encoder provenance.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KERNEL = r'''
# umud-seg-slm1 — P1: encoder comparison on identical protocol (FM_PLAN.md gate)
import os, sys, glob, json, random, time, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "segmentation-models-pytorch", "timm", "huggingface_hub"], check=False)
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

def L(m): print(m, flush=True)

# ---------- 1. pair GT (same as slm0 v2, trimmed) ----------
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
imgs, masks = [], []
for p in glob.glob(f"{IN}/**/*", recursive=True):
    if not os.path.isfile(p) or not p.lower().endswith(IMG_EXT):
        continue
    if "/notebooks/" in p and any(k in p.lower() for k in ("apo", "fasc", "seg", "slm")):
        continue
    (masks if any(k in p.lower() for k in ("mask", "label", "annot", "gt")) else imgs).append(p)
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
L(f"GT pairs: {len(pairs)}")
if len(pairs) < 50:
    json.dump({"status": "NO_GT_PAIRS", "n_pairs": len(pairs)},
              open(f"{OUT}/run_record.json", "w"), indent=1); raise SystemExit(0)

def device_of(p):
    s = p.lower()
    return next((k for k in ("telemed", "juniper", "lumify", "gray853", "h512", "left_ticks") if k in s), None)

for mp, _ in random.sample(pairs, min(8, len(pairs))):
    mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if mk is not None:
        u, c = np.unique(mk, return_counts=True)
        L(f"mask {os.path.basename(mp)} top values {sorted(zip(u.tolist(), c.tolist()), key=lambda x: -x[1])[:4]}")
if len(pairs) > 4000:
    pairs = random.sample(pairs, 4000)
    L("pairs capped to 4000 for wall-time")

IMG = 384
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
            if op == "hflip" and random.random() < 0.5: im, y = im[:, ::-1], y[:, ::-1]
            elif op == "rot":
                k = random.randint(0, 3); im, y = np.rot90(im, k), np.rot90(y, k)
            elif op == "bright": im = np.clip(im * random.uniform(0.7, 1.3), 0, 1)
            elif op == "contrast":
                m = im.mean(); im = np.clip((im - m) * random.uniform(0.7, 1.4) + m, 0, 1)
            elif op == "gamma": im = np.clip(im ** random.uniform(0.7, 1.4), 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(im))[None],
                torch.from_numpy(np.ascontiguousarray(y)))

def iou(g, p, eps=1e-7): return ((g & p).sum() + eps) / ((g | p).sum() + eps)

def evaluate(model, va):
    model.eval(); rows = []
    with torch.no_grad():
        for ip, mp in va:
            im, _ = SegDS([(ip, mp)], False)[0]
            pr = (torch.sigmoid(model(im[None].to(DEV)))[0] > 0.5).cpu().numpy().astype(bool)
            _, y = SegDS([(ip, mp)], False)[0]
            rows.append((device_of(ip), *[iou(y[c].numpy(), pr[c]) for c in range(2)]))
    return rows

def train(model, tr, va, tag, epochs=8, bs=8):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    dsc = smp.losses.DiceLoss(mode="multilabel"); bce = nn.BCEWithLogitsLoss()
    tl = DataLoader(SegDS(tr, True), batch_size=bs, shuffle=True, num_workers=2)
    best = None
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for im, y in tl:
            im, y = im.to(DEV), y.to(DEV)
            loss = 0.5 * dsc(model(im), y) + 0.5 * bce(model(im), y)
            opt.zero_grad(); loss.backward(); opt.step()
        rows = evaluate(model, va)
        arr = np.array([r[1:] for r in rows])
        L(f"[{tag}] ep{ep+1}/{epochs} loss {loss.item():.3f} IoU apo {arr[:,0].mean():.4f} fasc {arr[:,1].mean():.4f} ({time.time()-t0:.0f}s)")
        best = arr
    torch.save(model.state_dict(), f"{OUT}/slm1_{tag}.pt")
    return rows

import segmentation_models_pytorch as smp

# ---------- 2. encoder arm C: US-foundation init (runtime HF probe) ----------
USFM_NOTES = []
def probe_us_foundation():
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        cands = []
        for q in ("USFM ultrasound", "ultrasound foundation model", "UltraSam", "ultrasam"):
            try: cands += [m.id for m in api.list_models(search=q, limit=5)]
            except Exception as e: USFM_NOTES.append(f"search '{q}': {e}")
        L(f"HF candidates: {cands}")
        for rid in cands:
            try:
                files = [s.rfilename for s in api.model_info(rid).siblings]
                ck = [f for f in files if f.endswith((".pth", ".pt", ".bin", ".safetensors"))]
                if not ck: continue
                p = hf_hub_download(rid, ck[0])
                sd = torch.load(p, map_location="cpu", weights_only=False)
                sd = sd.get("state_dict", sd.get("model", sd))
                k = list(sd.keys())[:3]
                USFM_NOTES.append(f"loaded {rid} keys {k}")
                return rid, sd
            except Exception as e:
                USFM_NOTES.append(f"{rid}: {type(e).__name__} {e}")
    except Exception as e:
        USFM_NOTES.append(f"hub probe failed: {e}")
    return None, None

class ViTSeg(nn.Module):
    """ViT encoder (frozen patch embedding, fine-tuned blocks) + light FPN head for 2-class seg."""
    def __init__(self, vit_state=None, arch="vit_small_patch16_224"):
        super().__init__()
        import timm
        self.vit = timm.create_model(arch, pretrained=(vit_state is None), in_chans=1,
                                     img_size=224, num_classes=0, global_pool="")
        if vit_state is not None:
            matched, missing = self.vit.load_state_dict(vit_state, strict=False)
            L(f"ViT init: matched={len(matched) if isinstance(matched,list) else matched} "
              f"missing={len(missing) if isinstance(missing,list) else missing}")
        for p in self.vit.parameters(): p.requires_grad = True
        d = self.vit.embed_dim
        self.head = nn.Sequential(
            nn.Conv2d(d, 128, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=6), nn.Conv2d(64, 2, 1))
    def forward(self, x):
        x = nn.functional.interpolate(x, size=224, mode="bilinear")
        t = self.vit.forward_features(x)  # B, 1+N, D
        t = t[:, 1:, :].transpose(1, 2)
        m = t.reshape(-1, self.vit.embed_dim, 14, 14)
        return self.head(m)

def make_arm(name):
    if name == "resnet34_P0":
        return smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=2)
    if name == "mit_b1_segformer":
        return smp.Unet("mit_b1", encoder_weights="imagenet", in_channels=1, classes=2)
    rid, sd = probe_us_foundation()
    arch = "vit_small_patch16_224"
    if rid is not None:
        USFM_NOTES.append(f"arm-C init from {rid}")
    else:
        USFM_NOTES.append("no US-foundation weights loadable; arm-C = timm vit_small augreg_in21k (generic SSL)")
        try:
            arch = "vit_small_patch16_224.augreg_in21k"
        except Exception:
            arch = "vit_small_patch16_224"
    return ViTSeg(sd if rid else None, arch)

devs = sorted({d for d in (device_of(ip) for ip, _ in pairs) if d})
folds = [(f"lodv_{d}", [(c, m) for c, m in pairs if device_of(c) != d],
          [(c, m) for c, m in pairs if device_of(c) == d]) for d in devs]
if not devs:
    idx = list(range(len(pairs))); random.Random(2401).shuffle(idx)
    nva = max(50, len(idx) // 7)
    folds = [("random_holdout", [pairs[i] for i in idx[nva:]], [pairs[i] for i in idx[:nva]])]
L(f"folds: {[f[0] for f in folds]} sizes {[len(f[2]) for f in folds]}")
results = {"protocol": "identical: U-Net/head, Dice+BCE, TrivialAugment, ep8, bs8, IMG384, LODV (or seeded 85/15 holdout)",
           "arms": {}, "encoder_provenance": USFM_NOTES}
for arm in ("resnet34_P0", "mit_b1_segformer", "usfm_vit"):
    agg = {}
    try:
        for fname, tr, va in folds:
            if len(va) < 4 or len(tr) < 30:
                L(f"skip {fname} for {arm}"); continue
            model = make_arm(arm).to(DEV)
            rows = train(model, tr, va, f"{arm}_{fname}")
            arr = np.array([r[1:] for r in rows])
            agg[fname] = {"fasc_iou": float(arr[:, 1].mean()), "apo_iou": float(arr[:, 0].mean())}
            L(f"[{arm}] {fname}: fasc {agg[fname]['fasc_iou']:.4f} apo {agg[fname]['apo_iou']:.4f}")
        results["arms"][arm] = agg
    except Exception as e:
        results["arms"][arm] = {"error": f"{type(e).__name__}: {e}"}
        L(f"arm {arm} FAILED: {e}")
json.dump({"status": "OK", "n_pairs": len(pairs), **results},
          open(f"{OUT}/run_record.json", "w"), indent=1)
L("== slm1 done — same-benchmark encoder comparison in run_record.json ==")
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "slm1"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(KERNEL)
    meta = {
        "id": "kragglenote2forwork/umud-seg-slm1",
        "title": "umud-seg-slm1",
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
    print(f"slm1 kernel written to {out} — push with: kaggle kernels push -p {out}")

if __name__ == "__main__":
    main()
