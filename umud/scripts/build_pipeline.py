"""Probability maps -> features -> pipeline CSV, with PA and FL allowed to come from different fascicle-map ensembles.

    python scripts/build_pipeline.py --apo M/test/r34 --pa-maps M/test/r34 M/test/fasc3 M/test/fasc5 \
        [--fl-maps ...] [--config configs/a5_pa_avg.json] --work M/test/_ens --out daily/inputs/pipeline_X.csv \
        [--features-out daily/inputs/features_X.csv] [--compare daily/inputs/pipeline_three.csv]

Each map set is averaged (scripts/maps.py avg), features are extracted (umud.predict.extract_features) and the
submission columns are built with umud.predict.make_submission (config + cine groups daily/inputs/test_groups.npy).
PA comes from the --pa-maps build, FL from the --fl-maps build (default: the same maps), MT from the apo map.
--compare prints max / mean |diff| per target against an existing pipeline CSV.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from umud.predict import extract_features, make_submission  # noqa: E402


def ens(apo, maps, work: Path, tag: str) -> pd.DataFrame:
    d = work / tag
    fcsv = d / "features.csv"
    if not fcsv.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "maps.py"), "avg", "--dirs", *maps, "--apo", apo,
                        "--kinds", "fasc", "--out", str(d)], check=True)
        extract_features(d).to_csv(fcsv, index=False)
    return pd.read_csv(fcsv)


def tag_of(maps):
    return "+".join(Path(m).name for m in maps)


def compare(new: pd.DataFrame, old_csv: str):
    o = pd.read_csv(old_csv).set_index("image_id")
    n = new.set_index("image_id").loc[o.index]
    for t in ("pa_deg", "fl_mm", "mt_mm"):
        d = (n[t] - o[t]).abs()
        print(f"  {t}: max |diff| {d.max():.4f}  mean {d.mean():.4f}  rows > 0.01: {(d > 0.01).sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apo", required=True)
    ap.add_argument("--pa-maps", nargs="+", required=True)
    ap.add_argument("--fl-maps", nargs="+", default=None)
    ap.add_argument("--config", default=str(ROOT / "configs" / "a5_pa_avg.json"))
    ap.add_argument("--groups", default=str(ROOT / "daily" / "inputs" / "test_groups.npy"))
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--features-out", default=None)
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    work = Path(a.work)
    cfg = json.loads(Path(a.config).read_text())
    g = np.load(a.groups)
    fpa = ens(a.apo, a.pa_maps, work, tag_of(a.pa_maps))
    sub = make_submission(fpa, cfg, g)
    if a.fl_maps and tag_of(a.fl_maps) != tag_of(a.pa_maps):
        ffl = ens(a.apo, a.fl_maps, work, tag_of(a.fl_maps))
        sub["fl_mm"] = make_submission(ffl, cfg, g)["fl_mm"].values
    sub.to_csv(a.out, index=False)
    if a.features_out:
        fpa.to_csv(a.features_out, index=False)
    print(f"wrote {a.out} (PA {tag_of(a.pa_maps)}, FL {tag_of(a.fl_maps or a.pa_maps)})")
    if a.compare:
        compare(sub, a.compare)


if __name__ == "__main__":
    main()
