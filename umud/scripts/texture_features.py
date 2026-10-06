"""Raw-texture PA/FL features (geometry.raw_texture_pa) for a seg-output directory.

    python scripts/texture_features.py --seg <dir with probs/ + crops.csv> --images test --data <competition dir> \
        --out work/tex_test.csv
    python scripts/texture_features.py --seg <ext dir> --images ext --osf <osf dir> --neuage-csv <csv> --out work/tex_ext.csv

The aponeuroses come from the segmentation probabilities (same as the main pipeline); the fascicle direction comes
from the raw image intensity only, so its errors are largely independent of the fascicle segmentation.
"""
import argparse
import glob
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umud import seg  # noqa: E402
from umud.geometry import find_aponeuroses, raw_texture_pa  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seg", required=True)
ap.add_argument("--images", choices=["test", "ext"], required=True)
ap.add_argument("--data", default=None)
ap.add_argument("--osf", default=None)
ap.add_argument("--neuage-csv", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
segd = Path(a.seg)
crops = pd.read_csv(segd / "crops.csv")
if a.images == "test":
    paths = {Path(p).name: p for p in glob.glob(str(Path(a.data) / "test_images_v2" / "*" / "IMG_*"))}
else:
    spec = importlib.util.spec_from_file_location("eb", Path(__file__).with_name("external_bench.py"))
    eb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eb)
    paths = {iid: str(p) for iid, _, p, _, _ in eb.items(Path(a.osf), Path(a.neuage_csv) if a.neuage_csv else None)}
rows = []
for r in crops.itertuples():
    if r.image_id not in paths:
        continue
    apo_p = cv2.imread(str(segd / "probs" / f"{r.image_id}_apo.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    apo = find_aponeuroses(apo_p, r.px_per_mm) or find_aponeuroses(apo_p, r.px_per_mm, thr=0.3)
    rec = {"image_id": r.image_id}
    if apo is not None:
        g = seg.read_gray(paths[r.image_id])[int(r.t):int(r.b), int(r.l):int(r.r)]
        if g.shape[:2] == apo_p.shape[:2]:
            rec.update(raw_texture_pa(g, apo, r.px_per_mm))
    rows.append(rec)
pd.DataFrame(rows).to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(rows)} rows")
