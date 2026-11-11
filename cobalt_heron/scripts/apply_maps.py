"""Apply a fixed post-processing cfg to (averaged) uint8 test map sets -> submission.csv (no OOF needed).

    PYTHONPATH=. python scripts/apply_maps.py --test DIR [DIR ...] --cfg '{"t": 0.35, "hyst": 0.99, ...}' --out sub.csv
Areas in cfg are at 1024 px (the maps are averaged at 2048 and resized to 1024, as in tune_maps.py).
"""
import argparse, json
from pathlib import Path
import pandas as pd
from ch.post import instances
from ch.submit import to2048, rows_for
from tune_maps import load

ap = argparse.ArgumentParser()
ap.add_argument("--test", nargs="+", required=True); ap.add_argument("--cfg", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
cfg = json.loads(a.cfg); rows = []
stems = sorted(set.intersection(*[{p.stem for p in Path(d).glob("*.png")} for d in a.test]))
for s in stems:
    p = load(a.test, s)
    lab, _ = instances(p, None, **cfg)
    rows += rows_for(s, to2048(lab, p, cfg["t"]))
pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(a.out, index=False)
print("wrote", a.out, len(stems), "test images", len(rows), "instances")
