"""Day-13 gate: does a third fascicle model (resnext50_32x4d, 640x960) improve the production PA / FL estimators on
the expert sets when it joins the two-model ensemble (resnet34 + efficientnet-b3)?

Runs are seg-output dirs with features.csv (python scripts/eval_external.py --ext <dir> --recompute). Errors are
debiased per set (the LB blend tunes offsets/scales), compared row-paired on identical images; 95 % CI from a
4000-draw bootstrap of the per-row |error| difference (negative = candidate better).
Also: OSF blend simulation against DLTrack (stand-in for the Vera reference) at the live weights PA 0.62 / FL 0.31,
and the blend weight that minimises it, to see whether a less noisy pipeline would move the optimal weight.

    python external/research_d13/ens3_gate.py base=/home/user/work/ext_ens cand=/home/user/work/ext_ens3x ...
"""
import sys
import numpy as np
import pandas as pd

W = "/home/user/work"
runs = dict(a.split("=", 1) for a in sys.argv[1:])
lab = pd.read_csv(f"{W}/ext_bench/labels.csv").set_index("image_id")
xl = pd.read_excel(f"{W}/osf/benchmark_arch/benchmark_dataset_architecture_v0.1.0/"
                   "Results_benchmark_architecture_v0.1.0.xlsx", sheet_name="Manual_architecture").set_index("ImageID")


def est(f):
    pa = (f.pa_wmed + f.v2_pa_lowhalf.fillna(f.pa_wmed)) / 2
    fl = (f.fl_med + f.v2_fl_chord_mid.fillna(f.fl_med)) / 2
    return pd.DataFrame({"pa_deg": pa, "fl_mm": fl, "mt_mm": f.mt_inner})


E = {k: est(pd.read_csv(f"{v}/features.csv").set_index("image_id")) for k, v in runs.items()}
names = list(E)
base = names[0]
rng = np.random.default_rng(0)
for t in ("pa_deg", "fl_mm"):
    print(f"\n=== {t}: debiased MAE per set (paired, identical rows); CI of mean(|e_cand| - |e_{base}|)")
    for s in ("osf", "neuage", "gm"):
        L = lab[lab.set == s][t].dropna()
        ok = L.index
        for k in names:
            ok = ok.intersection(E[k][t].dropna().index)
        err = {k: (E[k][t][ok] - L[ok]) for k in names}
        err = {k: (e - e.median()).abs() for k, e in err.items()}
        line = f"  {s:6s} n={len(ok):3d} " + " ".join(f"{k}={err[k].mean():.3f}" for k in names)
        for k in names[1:]:
            d = (err[k] - err[base]).values
            bs = np.array([rng.choice(d, len(d)).mean() for _ in range(4000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            flag = "BETTER" if hi < 0 else ("WORSE" if lo > 0 else "ns")
            line += f" | {k}-{base} {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] {flag}"
        print(line)

print("\n=== OSF blend sim vs DLTrack (both debiased), score units (PA/6, FL/12)")
o = lab[lab.set == "osf"]
for k in names:
    P = E[k].loc[o.index]
    for t, dcol, w0, div in (("pa_deg", "DLTrack_PA", 0.62, 6), ("fl_mm", "DLTrack_FL", 0.31, 12)):
        m = pd.DataFrame({"p": P[t], "d": xl[dcol].reindex(o.index), "y": o[t]}).dropna()
        p = m.p - (m.p - m.y).median()
        d = m.d - (m.d - m.y).median()
        ws = np.round(np.arange(0, 1.001, 0.02), 2)
        sc = {w: (d + w * (p - d) - m.y).abs().mean() / div for w in ws}
        wb = min(sc, key=sc.get)
        print(f"  {k:6s} {t}: n={len(m)} at live w={w0}: {sc[w0]:.4f}  best w={wb:.2f}: {sc[wb]:.4f}  "
              f"pipe alone {sc[1.0]:.4f}  err corr {np.corrcoef(p - m.y, d - m.y)[0, 1]:+.2f}")
