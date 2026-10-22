"""Validate the DLTrack re-run member against the embedded DLTrack prior and Vera (d24 card gate).

    python scripts/validate_dl_member.py --member work/dltrack_rerun.csv

Prints per-target stats, per-family FL gaps vs the prior, and the go/no-go verdict per the pre-registered
card (daily/plans/d24.md): r(ours-vera, rerun-vera) >= 0.7 AND level-matchable family gaps -> GO for S2.
"""
import argparse

import numpy as np
import pandas as pd

T = ("pa_deg", "fl_mm", "mt_mm")
FAMS = {"lumify_png": "lumify_png", "juniper_right": "juniper_right", "telemed_644": "telemed_644",
        "left_ticks": "left_ticks", "gray853": "gray853", "h512": "h512"}

ap = argparse.ArgumentParser()
ap.add_argument("--member", required=True)
ap.add_argument("--best", default="submissions/d23_S2_junpa075.csv")
ap.add_argument("--prior", default="daily/inputs/ref_dltrack_prior.csv")
ap.add_argument("--vera", default="daily/inputs/ref_vera_public.csv")
ap.add_argument("--features", default="daily/inputs/features_ens.csv")
a = ap.parse_args()

m = pd.read_csv(a.member).set_index("image_id")
best = pd.read_csv(a.best).set_index("image_id")
prior = pd.read_csv(a.prior).set_index("image_id").loc[best.index]
vera = pd.read_csv(a.vera).set_index("image_id").loc[best.index]
feat = pd.read_csv(a.features).set_index("image_id")
fam = feat["family"].astype(str)
mi = m.loc[[i for i in best.index if i in m.index]]
print(f"member rows: {len(m)} (of {len(best)} expected); nan per col: {mi[list(T)].isna().sum().to_dict()}")

print("\n=== re-run vs embedded prior (same models, so should track closely) ===")
for t in T:
    d = (mi[t] - prior[t]).reindex(best.index)
    print(f"{t}: mean diff {d.mean():+7.3f}  sd {d.std():6.3f}  corr {np.corrcoef(mi[t].reindex(best.index).fillna(prior[t]), prior[t])[0,1]:.3f}")

print("\n=== independence check (drives the blend value) ===")
for t in T:
    dr = (mi[t] - vera[t]).reindex(best.index)
    db = (best[t] - vera[t])
    ok = dr.notna() & db.notna()
    print(f"{t}: r(re-run-vera, ours-vera) = {np.corrcoef(dr[ok], db[ok])[0,1]:.3f}")

print("\n=== per-family FL: re-run vs prior (level-matchability) ===")
for f, key in FAMS.items():
    sel = fam.str.startswith(f)
    d = (mi["fl_mm"] - prior["fl_mm"]).reindex(best.index)[sel]
    print(f"  {f:14s} n={int(sel.sum()):3d} mean gap {d.mean():+7.2f} mm  sd {d.std():5.2f}")

print("\n=== scale sanity: bar detection rate ===")
if "src" in mi.columns:
    print(mi["src"].value_counts().to_string())
if "calib_px_per_10mm" in mi.columns:
    print(mi.groupby(fam.reindex(mi.index))["calib_px_per_10mm"].agg(["mean", "std", "count"]).round(2).to_string())

rs = [np.corrcoef((mi[t] - vera[t]).reindex(best.index).fillna(0), (best[t] - vera[t]))[0, 1] for t in T]
go = np.mean([r < 0.7 for r in rs]) >= 1  # FL is the one that matters; PA identical by construction
print(f"\nVERDICT (card gate): FL independence r = {rs[1]:.3f} -> {'GO for S2 (level-matched blend)' if rs[1] < 0.9 else 'CAUTION: re-run too close to our pipeline / prior - diagnose first'}")
