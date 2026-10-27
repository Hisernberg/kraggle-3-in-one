"""Local (DGX) probability-map inference for trained seg models: test set and the held-out train split.

    python scripts/maps.py infer --weights <dir with apo.pt/fasc.pt[/*.encoder]> --size 640x960 --kinds fasc \
        --set test|val_fasc|val_apo --out work/maps/<name> [--iso]
    python scripts/maps.py avg --dirs work/maps/a work/maps/b ... --out work/maps/ens [--kinds fasc] [--apo DIR]

Layout of every output dir = seg.py / external_bench.py layout (probs/<id>_<kind>.png + crops.csv), so
`umud.predict.extract_features` and scripts/avg_probs.py work unchanged.

Sets:
  test      the 309 competition test images, B-mode crop from umud.scale.detect_scale (as in seg.infer_test)
  val_fasc  the fascicle validation split that EVERY fascicle kernel held out (dedupe + RandomState(42) permutation,
            first len/12; see seg.train_kind) - unseen by all fascicle models -> fair map-quality harness
  val_apo   the same for the aponeurosis split (unseen by the apo model)
Train frames with a detected scale (Juniper / h512 screenshots) are cropped like the test; the rest use the full frame
(fascicle crops in the train set have no scale bar) with a nominal 14.82 px/mm (the detect_scale fallback).

--iso: isotropic inference for the non-square-pixel devices (Juniper R 1.12, Lumify R 0.87 per the d16 physics):
the crop is resampled to square true-geometry pixels before the network and the map is resampled back.
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = Path("/home/urad/data/umud/comp")
ISO_R = {"juniper_right": 1.12, "lumify_png": 0.87}  # horizontal / vertical px per mm (true-geometry factor)


def val_split(kind: str, cache: Path) -> list:
    """(image path, mask path) of the held-out split, identical to seg.train_kind (SPLIT_SEED 42, len // 12)."""
    from umud import seg
    f = cache / f"split_{kind}.json"
    if f.exists():
        return json.loads(f.read_text())
    pairs = seg.list_pairs(DATA, kind)
    idx = np.random.RandomState(seg.SPLIT_SEED).permutation(len(pairs))
    nval = max(20, len(pairs) // 12)
    va = [list(pairs[i]) for i in idx[:nval]]
    cache.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(va))
    return va


def items(set_name: str, cache: Path):
    """(id, image path, mask path or None, px_per_mm, (l, t, r, b), family)."""
    from PIL import Image
    from umud.scale import detect_scale
    out = []
    if set_name == "test":
        for f in sorted(glob.glob(str(DATA / "test_images_v2" / "*" / "IMG_*"))):
            name = Path(f).name
            with Image.open(f) as im:
                sc = detect_scale(np.asarray(im), name.rsplit(".", 1)[-1])
            out.append((name, f, None, sc.px_per_mm, (sc.l, sc.t, sc.r, sc.b), sc.family))
        return out
    kind = set_name.split("_")[1]
    for ip, mp in val_split(kind, cache):
        with Image.open(ip) as im:
            a = np.asarray(im)
        sc = detect_scale(a, "tif")
        h, w = a.shape[:2]
        if sc.family == "unknown":
            box, fam = (0, 0, w, h), f"train_{w}x{h}"
        else:
            box, fam = (sc.l, sc.t, sc.r, sc.b), sc.family
        iid = f"{kind}_{Path(ip).stem}"
        out.append((iid, ip, mp, sc.px_per_mm, box, fam))
    return out


def iso_predict(seg, model, g, dev, r):
    """Resample the crop to square pixels (width / r), predict, resample the map back."""
    h, w = g.shape
    w2 = max(8, int(round(w / r)))
    gi = cv2.resize(g, (w2, h), interpolation=cv2.INTER_AREA if w2 < w else cv2.INTER_LINEAR)
    p = seg.predict(model, gi, dev)
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)


def infer(a):
    import torch
    from umud import seg
    torch.set_num_threads(4)
    if a.mem_frac:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    seg.IN_H, seg.IN_W = (int(v) for v in a.size.split("x"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wdir = Path(a.weights)
    models = {}
    for k in a.kinds.split(","):
        if not (wdir / f"{k}.encoder").exists():  # seg-train wrote no .encoder files: resnet34
            seg.ENCODERS.setdefault(k, "resnet34")
        models[k] = seg.load_model(wdir, k, dev)
        seg.ENCODERS.pop(k, None)
    out = Path(a.out)
    (out / "probs").mkdir(parents=True, exist_ok=True)
    rows = []
    its = items(a.set, Path(a.cache))
    gt = out / "gt"
    for i, (iid, ip, mp, ppm, (l, t, r, b), fam) in enumerate(its):
        rows.append((iid, ppm, l, t, r, b, fam))
        g = seg.read_gray(ip)[t:b, l:r]
        if mp is not None and a.write_gt:  # ground-truth mask on the same crop, stored as a 0/255 "probability"
            gt.mkdir(exist_ok=True)
            m = seg.read_gray(mp)
            full = seg.read_gray(ip)
            m = cv2.resize(m, (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST)
            if (m > 127).mean() > 0.5:
                m = 255 - m
            cv2.imwrite(str(gt / f"{iid}.png"), ((m[t:b, l:r] > 127) * 255).astype(np.uint8))
        rr = next((v for k, v in ISO_R.items() if fam.startswith(k)), None) if a.iso else None
        for k, m in models.items():
            p = iso_predict(seg, m, g, dev, rr) if rr else seg.predict(m, g, dev)
            cv2.imwrite(str(out / "probs" / f"{iid}_{k}.png"), np.clip(p * 255, 0, 255).astype(np.uint8))
        if i % 50 == 0:
            print(f"{i}/{len(its)} {iid}", flush=True)
    with open(out / "crops.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "px_per_mm", "l", "t", "r", "b", "family"])
        w.writerows(rows)
    print(f"{a.set}: {len(rows)} images x {list(models)} -> {out}", flush=True)


def avg(a):
    """Mean of the fascicle (and/or apo) maps of several dirs; apo taken from --apo (default: first dir)."""
    import pandas as pd
    dirs = [Path(d) for d in a.dirs]
    w = a.weights or [1.0] * len(dirs)
    out = Path(a.out)
    (out / "probs").mkdir(parents=True, exist_ok=True)
    apo_dir = Path(a.apo) if a.apo else dirs[0]
    crops = pd.read_csv(apo_dir / "crops.csv")
    crops.to_csv(out / "crops.csv", index=False)
    kinds = a.kinds.split(",")
    for iid in crops.image_id:
        for k in ("apo", "fasc"):
            src = [apo_dir] if (k == "apo" and k not in kinds) else dirs
            ws = [1.0] if (k == "apo" and k not in kinds) else w
            acc, tw = None, 0.0
            for d, wi in zip(src, ws):
                p = d / "probs" / f"{iid}_{k}.png"
                if not p.exists():
                    continue
                x = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                if acc is not None and x.shape != acc.shape:
                    x = cv2.resize(x, (acc.shape[1], acc.shape[0]), interpolation=cv2.INTER_LINEAR)
                acc = wi * x if acc is None else acc + wi * x
                tw += wi
            if acc is not None:
                cv2.imwrite(str(out / "probs" / f"{iid}_{k}.png"), np.clip(acc / tw, 0, 255).astype(np.uint8))
    print(f"averaged {len(crops)} ids ({kinds}) from {len(dirs)} dirs -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("infer")
    a1.add_argument("--weights", required=True)
    a1.add_argument("--size", default="512x768")
    a1.add_argument("--kinds", default="apo,fasc")
    a1.add_argument("--set", default="test", choices=["test", "val_fasc", "val_apo"])
    a1.add_argument("--out", required=True)
    a1.add_argument("--cache", default="/home/urad/data/umud/work")
    a1.add_argument("--iso", action="store_true")
    a1.add_argument("--write-gt", action="store_true")
    a1.add_argument("--mem-frac", type=float, default=None)
    a2 = sub.add_parser("avg")
    a2.add_argument("--dirs", nargs="+", required=True)
    a2.add_argument("--weights", type=float, nargs="+", default=None)
    a2.add_argument("--kinds", default="fasc")
    a2.add_argument("--apo", default=None)
    a2.add_argument("--out", required=True)
    a = ap.parse_args()
    infer(a) if a.cmd == "infer" else avg(a)


if __name__ == "__main__":
    main()
