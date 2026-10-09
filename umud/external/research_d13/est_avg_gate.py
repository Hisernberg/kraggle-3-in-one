"""Estimate-level vs probability-level model averaging on the expert sets (paired, debiased, identical rows).

Candidates are built from per-model features.csv files (production estimators), then compared with the live
two-model probability-averaged ensemble (ext_ens). Same bootstrap as ens3_gate.py.
"""
import numpy as np
import pandas as pd

W = "/home/user/work"
lab = pd.read_csv(f"{W}/ext_bench/labels.csv").set_index("image_id")


def est(path):
    f = pd.read_csv(f"{path}/features.csv").set_index("image_id")
    return pd.DataFrame({"pa_deg": (f.pa_wmed + f.v2_pa_lowhalf.fillna(f.pa_wmed)) / 2,
                         "fl_mm": (f.fl_med + f.v2_fl_chord_mid.fillna(f.fl_med)) / 2})


M = {k: est(f"{W}/{d}") for k, d in dict(ens2="ext_ens", r34="ext_bench", eff="ext_eff", x50="ext_x50",
                                          r34x50="ext_r34x50", ens3="ext_ens3x").items()}
C = {"ens2": M["ens2"],
     "avg3_est": pd.concat([M["r34"], M["eff"], M["x50"]]).groupby(level=0).mean(),
     "avg_r34x50_est": pd.concat([M["r34"], M["x50"]]).groupby(level=0).mean(),
     "r34x50_prob": M["r34x50"],
     "ens2+x50_est": pd.concat([M["ens2"], M["x50"]]).groupby(level=0).mean()}
rng = np.random.default_rng(0)
for t in ("pa_deg", "fl_mm"):
    print(f"\n=== {t}")
    for s in ("osf", "neuage", "gm"):
        L = lab[lab.set == s][t].dropna()
        ok = L.index
        for v in C.values():
            ok = ok.intersection(v[t].dropna().index)
        err = {k: (v[t][ok] - L[ok]) for k, v in C.items()}
        err = {k: (e - e.median()).abs() for k, e in err.items()}
        line = f"  {s:6s} n={len(ok):3d} ens2={err['ens2'].mean():.3f}"
        for k in list(C)[1:]:
            d = (err[k] - err["ens2"]).values
            bs = np.array([rng.choice(d, len(d)).mean() for _ in range(4000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            line += f" | {k} {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]" + ("*" if hi < 0 or lo > 0 else "")
        print(line)
