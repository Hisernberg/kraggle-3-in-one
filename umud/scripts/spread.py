"""Scale each family's deviations from its own mean for one target (family means kept exactly).

    python scripts/spread.py --base submissions/X.csv --target pa_deg --s 0.90 --out daily/out/Y.csv
Families: the device family from daily/inputs/features_ens.csv (Juniper / gray853 sub-presets pooled).
Values are clipped to the host's physiological ranges afterwards.
"""
import argparse
import pandas as pd

RANGE = {"pa_deg": (5, 45), "fl_mm": (30, 200), "mt_mm": (10, 50)}

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--target", required=True, choices=list(RANGE))
ap.add_argument("--s", type=float, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
b = pd.read_csv(a.base)
fam = b.image_id.map(pd.read_csv("daily/inputs/features_ens.csv").set_index("image_id").family.astype(str))
fam = fam.where(~fam.str.startswith("juniper"), "juniper").where(~fam.str.startswith("gray"), "gray853")
m = b.groupby(fam)[a.target].transform("mean")
b[a.target] = (m + a.s * (b[a.target] - m)).clip(*RANGE[a.target]).round(4)
b.to_csv(a.out, index=False)
print(a.out, "mean |d|", round(float((b[a.target] - pd.read_csv(a.base)[a.target]).abs().mean()), 4))
