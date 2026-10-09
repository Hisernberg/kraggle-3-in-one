"""Expert-labelled external validation sets (UMUD OSF repository, osf.io/xbawc), declared external data.

    python scripts/external_bench.py infer --osf <dir with the unzipped OSF sets> --weights <dir with apo.pt/fasc.pt> \
        --out work/ext            # segmentation probabilities for every labelled image (CPU is fine)
    python scripts/external_bench.py labels --osf <dir> --neuage-csv <neuage_expert.csv> --out work/ext

Sets (all public, CC-BY-4.0):
  osf     "Expert Analysed Benchmark" architecture set: 35 images (GM / SOL / VL; Aloka, Telemed, Philips HD11),
          up to 7 raters, FIJI protocol identical to the competition labels (3 MT lines, 3 fascicles, 3 angles).
  neuage  "Example Annotated Images": 249 VL images (Esaote MyLab70), one expert, fascicles drawn as segmented lines
          extrapolated straight to the aponeuroses, PA = intersection angle with the deep aponeurosis. Expert values
          are parsed from the red annotation lines (table from the public CC0 "umud-code" Kaggle dataset).
  gm      "GM dynamic" calf-raise video (Telemed): 167 frames, 3 raters, PA in degrees and FL in pixels; raters
          analysed every second frame (odd frames are interpolations), so only even frames are used.

`infer` writes probs/<id>_apo.png, probs/<id>_fasc.png and crops.csv (id, px_per_mm, l, t, r, b, family) in the
same layout as the test-set segmentation output, so `umud.predict.extract_features` works unchanged.
`labels` writes labels.csv (id, set, pa_deg, fl_mm, mt_mm, fl_px) with the expert reference values.
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OSF_DIR = "benchmark_arch/benchmark_dataset_architecture_v0.1.0"
GM_DIR = "benchmark_gm_dynamic/benchmark_dataset_architecture_GM_dynamic_v0.1.0"
GM_PX_PER_MM = 7.75  # Telemed depth ruler: 10 mm ticks every 77.6 px (y = 43, 119.5, ..., 507.5)
GM_CROP = (97, 45, 560, 556)  # l, t, r, b of the B-mode area in the 660x556 frames
NEUAGE_TOP = 128  # first B-mode row of the Esaote frames


def items(osf: Path, neuage_csv: Path | None):
    """(id, set, image path, px_per_mm, crop box) for every labelled image."""
    out = []
    x = pd.read_excel(glob.glob(str(osf / OSF_DIR / "Results_*.xlsx"))[0], sheet_name="Manual_architecture")
    for r in x.itertuples():
        p = osf / OSF_DIR / f"{r.ImageID}.tif"
        h, w = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).shape
        out.append((r.ImageID, "osf", p, r.Scale_pixel_per_cm / 10.0, (0, 0, w, h)))
    if neuage_csv is not None and neuage_csv.exists():
        n = pd.read_csv(neuage_csv)
        for r in n.itertuples():
            if isinstance(r.flags, str) and "parse_incomplete" in r.flags:
                continue
            sub = "young_annot" if r.cohort == "young" else "old_annot"
            p = osf / sub / r.raw_member
            if not p.exists():
                continue
            h = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).shape[0]
            out.append((r.image_id, "neuage", p, r.px_per_mm, (int(r.field_x0), NEUAGE_TOP, int(r.field_x1), h)))
    for p in sorted(glob.glob(str(osf / GM_DIR / "frames" / "*.png"))):
        k = int(Path(p).stem.split("_")[-1])
        if k % 2 == 0:
            out.append((f"gm_{k:03d}", "gm", Path(p), GM_PX_PER_MM, GM_CROP))
    return out


def infer(a):
    import torch
    from umud import seg
    torch.set_num_threads(4)
    if a.size:  # network input HxW; match the size the weights were trained at
        seg.IN_H, seg.IN_W = (int(v) for v in a.size.split("x"))
    models = {k: seg.load_model(Path(a.weights), k) for k in a.kinds.split(",")}
    out = Path(a.out)
    (out / "probs").mkdir(parents=True, exist_ok=True)
    rows = []
    its = items(Path(a.osf), Path(a.neuage_csv) if a.neuage_csv else None) if a.osf != "none" else []
    if a.test_data:
        from PIL import Image
        from umud.scale import detect_scale
        for f in sorted(glob.glob(str(Path(a.test_data) / "test_images_v2" / "*" / "IMG_*"))):
            name = Path(f).name
            with Image.open(f) as im:
                sc = detect_scale(np.asarray(im), name.rsplit(".", 1)[-1])
            its.append((name, "test", Path(f), sc.px_per_mm, (sc.l, sc.t, sc.r, sc.b)))
    for i, (iid, s, p, ppm, (l, t, r, b)) in enumerate(its):
        rows.append((iid, ppm, l, t, r, b, s))
        if (out / "probs" / f"{iid}_fasc.png").exists():
            continue
        g = seg.read_gray(str(p))[t:b, l:r]
        for k, m in models.items():
            pr = seg.predict_ms(m, g, "cpu") if a.ms else seg.predict(m, g, "cpu")
            cv2.imwrite(str(out / "probs" / f"{iid}_{k}.png"), np.clip(pr * 255, 0, 255).astype(np.uint8))
        if i % 25 == 0:
            print(f"{i}/{len(its)} {iid}", flush=True)
    with open(out / "crops.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "px_per_mm", "l", "t", "r", "b", "family"])
        w.writerows(rows)


def labels(a):
    osf = Path(a.osf)
    rows = []
    x = pd.read_excel(glob.glob(str(osf / OSF_DIR / "Results_*.xlsx"))[0], sheet_name="Manual_architecture")
    for t in ("MT", "FL", "PA"):
        r = x[[f"R{i}_{t}" for i in range(1, 8)]].astype(float)
        med = r.median(axis=1)
        r = r.where((r - med.values[:, None]).abs() < 0.5 * med.values[:, None])  # drop entry errors (80 mm MT)
        x[f"GT_{t}"] = r.mean(axis=1)
    for r in x.itertuples():
        rows.append(dict(image_id=r.ImageID, set="osf", pa_deg=r.GT_PA, fl_mm=r.GT_FL, mt_mm=r.GT_MT, fl_px=np.nan))
    if a.neuage_csv:
        n = pd.read_csv(a.neuage_csv)
        for r in n.itertuples():
            if isinstance(r.flags, str) and "parse_incomplete" in r.flags:
                continue
            rows.append(dict(image_id=r.image_id, set="neuage", pa_deg=r.pa_deg, fl_mm=r.fl_mm, mt_mm=r.mt_mm,
                             fl_px=np.nan))
    xl = glob.glob(str(osf / GM_DIR / "*.xlsx"))[0]
    fl, pa = pd.read_excel(xl, sheet_name="FL"), pd.read_excel(xl, sheet_name="PA")
    for f, p in zip(fl.itertuples(), pa.itertuples()):
        k = int(f.frame)
        if k % 2 == 0:
            rows.append(dict(image_id=f"gm_{k:03d}", set="gm", pa_deg=p.Mean, fl_mm=f.Mean / GM_PX_PER_MM,
                             mt_mm=np.nan, fl_px=f.Mean))
    pd.DataFrame(rows).to_csv(Path(a.out) / "labels.csv", index=False)
    print(pd.DataFrame(rows).groupby("set").describe().T.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["infer", "labels"])
    ap.add_argument("--osf", required=True)
    ap.add_argument("--neuage-csv", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ms", action="store_true", help="multi-scale TTA (0.875 / 1 / 1.125)")
    ap.add_argument("--size", default=None, help="network input HxW (default: umud.seg.IN_H x IN_W)")
    ap.add_argument("--kinds", default="apo,fasc", help="models to run, e.g. fasc only")
    ap.add_argument("--test-data", default=None, help="also infer the competition test images (B-mode crop) into out")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    infer(a) if a.cmd == "infer" else labels(a)


if __name__ == "__main__":
    main()
