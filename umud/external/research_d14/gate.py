"""Day-14 gate: new fascicle (resnet34 @640x960) and aponeurosis (resnext50 @640x960) models vs the live pipeline.

Runs are seg-output dirs with features.csv (scripts/eval_external.py --ext <dir> --recompute). The first run is the
baseline. Errors are debiased per set (the LB blend tunes offsets/scales) and compared row-paired on identical images;
95 % CI from a 4000-draw bootstrap of the per-row |error| difference (negative = candidate better).

    python external/research_d14/gate.py base=/home/user/work/ext_pair cand=/home/user/work/ext_x ...
"""
import sys
import numpy as np
import pandas as pd

W = "/home/user/work"
runs = dict(a.split("=", 1) for a in sys.argv[1:])
lab = pd.read_csv(f"{W}/ext_bench/labels.csv").set_index("image_id")


def est(f):
    return pd.DataFrame({"pa_deg": (f.pa_wmed + f.v2_pa_lowhalf.fillna(f.pa_wmed)) / 2,
                         "fl_mm": (f.fl_med + f.v2_fl_chord_mid.fillna(f.fl_med)) / 2,
                         "mt_mm": f.mt_inner})


E = {k: est(pd.read_csv(f"{v}/features.csv").set_index("image_id")) for k, v in runs.items()}
names = list(E)
base = names[0]
rng = np.random.default_rng(0)
for t in ("pa_deg", "fl_mm", "mt_mm"):
    print(f"\n=== {t}: debiased MAE (paired, identical rows); CI of mean(|e_cand| - |e_{base}|)")
    for s in ("osf", "neuage", "gm"):
        L = lab[lab.set == s][t].dropna()
        ok = L.index
        for k in names:
            ok = ok.intersection(E[k][t].dropna().index)
        if len(ok) < 10:
            continue
        err = {k: (E[k][t][ok] - L[ok]) for k in names}
        err = {k: (e - e.median()).abs() for k, e in err.items()}
        line = f"  {s:6s} n={len(ok):3d} " + " ".join(f"{k}={err[k].mean():.3f}" for k in names)
        for k in names[1:]:
            d = (err[k] - err[base]).values
            bs = np.array([rng.choice(d, len(d)).mean() for _ in range(4000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            flag = "BETTER" if hi < 0 else ("WORSE" if lo > 0 else "ns")
            line += f" | {k} {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] {flag}"
        print(line)
