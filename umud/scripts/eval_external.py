"""Score every geometry estimator against the expert-labelled external sets (see external_bench.py).

    python scripts/eval_external.py --ext work/ext [--features work/ext/features.csv]

Prints, per set (osf / neuage / gm) and target, MAE, bias and MAE after removing the set's median bias, all in
competition units (PA deg, FL mm, MT mm) plus the normalised MAE (PA/6, FL/12, MT/3). For the GM video, FL is
compared in pixels converted with the frame scale, so only relative FL errors are meaningful there.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umud.predict import extract_features  # noqa: E402

TAU = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", required=True)
    ap.add_argument("--features", default=None)
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    ext = Path(a.ext)
    fpath = Path(a.features) if a.features else ext / "features.csv"
    if a.recompute or not fpath.exists():
        extract_features(ext).to_csv(fpath, index=False)
    f = pd.read_csv(fpath).set_index("image_id")
    lab = pd.read_csv(ext / "labels.csv").set_index("image_id")
    df = f.join(lab, how="inner")
    cands = {
        "pa_deg": [c for c in df.columns if c.startswith("pa_") or c.startswith("v2_pa_")],
        "fl_mm": [c for c in df.columns if c.startswith("fl_") and c not in ("fl_mm", "fl_px")]
        + [c for c in df.columns if c.startswith("v2_fl_")],
        "mt_mm": [c for c in df.columns if (c.startswith("mt_") or c.startswith("v2_mt_")) and c != "mt_mm"],
    }
    cands["pa_deg"] = [c for c in cands["pa_deg"] if c not in ("pa_deg", "v2_pa_slope")]
    rows = []
    for s, g in df.groupby("set"):
        for t, cs in cands.items():
            y = g[t]
            if y.notna().sum() < 5:
                continue
            for c in cs:
                e = (g[c] - y)[g.ok == 1].dropna()
                if len(e) < 5:
                    continue
                rows.append(dict(set=s, target=t, est=c, n=len(e), mae=e.abs().mean(), bias=e.mean(),
                                 mae_debiased=(e - e.median()).abs().mean(), norm=e.abs().mean() / TAU[t]))
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for (s, t), g in r.groupby(["set", "target"]):
        print(f"\n== {s} {t} (n={g.n.max()})")
        print(g.sort_values("mae_debiased").drop(columns=["set", "target"]).round(3).to_string(index=False))
    print("\nfailures:", (df.ok != 1).groupby(df.set).sum().to_dict())


if __name__ == "__main__":
    main()
