"""Average segmentation probability maps of several runs (model / TTA ensembles) into one seg-output directory.

    python scripts/avg_probs.py --dirs work/seg work/seg_fasc2 --kinds fasc --out work/seg_ens [--weights 0.5 0.5]

Every dir has the layout written by seg.py / external_bench.py (probs/<id>_<kind>.png + crops.csv). Kinds not listed
in --kinds are copied from the first dir. crops.csv is taken from the first dir; ids missing in another dir are
averaged over the dirs that have them.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--kinds", default="apo,fasc")
ap.add_argument("--weights", type=float, nargs="+", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dirs = [Path(d) for d in a.dirs]
w = a.weights or [1.0] * len(dirs)
out = Path(a.out)
(out / "probs").mkdir(parents=True, exist_ok=True)
crops = pd.read_csv(dirs[0] / "crops.csv")
crops.to_csv(out / "crops.csv", index=False)
kinds = a.kinds.split(",")
for iid in crops.image_id:
    for k in ("apo", "fasc"):
        name = f"{iid}_{k}.png"
        if k not in kinds:
            shutil.copy(dirs[0] / "probs" / name, out / "probs" / name)
            continue
        acc, tw = None, 0.0
        for d, wi in zip(dirs, w):
            p = d / "probs" / name
            if not p.exists():
                continue
            x = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            if acc is not None and x.shape != acc.shape:
                x = cv2.resize(x, (acc.shape[1], acc.shape[0]), interpolation=cv2.INTER_LINEAR)
            acc = wi * x if acc is None else acc + wi * x
            tw += wi
        cv2.imwrite(str(out / "probs" / name), np.clip(acc / tw, 0, 255).astype(np.uint8))
print(f"averaged {len(crops)} ids x {kinds} from {len(dirs)} dirs -> {out}")
