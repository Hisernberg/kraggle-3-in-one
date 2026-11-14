"""d35: external-benchmark axis decomposition + per-image variant selection search.

    python /home/z/my-project/scripts/d35_adaptive.py

1. Metric decomposition: for the current best recipe columns (pa_med-mix etc. from the blend),
   compute per-axis MAE on the 343 external GT frames -> where does 0.19747 live?
2. Oracle analysis: per axis, per image, the best single variant column from features_ens /
   features_v2 style candidates -> upper bound of per-image selection.
3. Rule search: thresholds on observable features (n_fasc, ok, n_bands, v2_pa_slope...)
   selecting between 2 variants per axis -> evaluate rule MAE vs global-best-variant MAE.
Outputs a report + candidate per-axis selectors to build the next cards from.
"""
import numpy as np
import pandas as pd

U = "/home/z/my-project/work/kraggle-3-in-one/umud"
W = np.array([1 / 6, 1 / 12, 1 / 3]) / 3  # PA, FL, MT weights inside the metric

lab = pd.read_csv(f"{U}/external/labels.csv")
ext = pd.read_csv(f"{U}/external/features.csv")
d = lab.merge(ext, on="image_id", how="inner")
print(f"joined: {len(d)} frames of {len(lab)} labels")

PA_CANDS = ["pa_med", "pa_wmed", "pa_wmean", "pa_top5", "v2_pa_lowhalf", "v2_pa_low3",
            "v2_pa_uphalf", "v2_pa_insert", "v2_pa_fit1", "v2_pa_fit0", "v2_pa_fitmid",
            "v2_pa_fit09"]
FL_CANDS = ["fl_med", "fl_wmed", "fl_top5", "v2_fl_curve", "v2_fl_chord_mid",
            "v2_fl_chord_deep", "v2_fl_stream"]
MT_CANDS = ["mt_inner", "mt_inner_vert", "mt_center", "mt_center_vert", "mt_outer",
            "mt_outer_vert", "v2_mt_raw_1090", "v2_mt_raw_1684", "v2_mt_raw_2575",
            "v2_mt_raw_mid"]

def mae_axis(pred, gt):
    ok = pred.notna() & gt.notna()
    return float(np.abs(pred[ok] - gt[ok]).mean()), int(ok.sum())

print("\n=== per-variant per-axis MAE on external GT (n=%d) ===" % len(d))
for axis, cands in (("PA", PA_CANDS), ("FL", FL_CANDS), ("MT", MT_CANDS)):
    gt = d[axis.lower() + "_deg" if axis == "PA" else axis.lower() + "_mm"]
    rows = []
    for c in cands:
        if c in d.columns:
            m, n = mae_axis(d[c], gt)
            rows.append((m, n, c))
    rows.sort()
    print(f"{axis}: " + " | ".join(f"{c}={m:.3f}(n{n})" for m, n, c in rows[:5]))

# oracle: best variant per image per axis
print("\n=== oracle per-image selection (upper bound) ===")
oracle = {}
for axis, cands, suf in (("PA", PA_CANDS, "_deg"), ("FL", FL_CANDS, "_mm"), ("MT", MT_CANDS, "_mm")):
    gt = d[axis.lower() + suf]
    cols = [c for c in cands if c in d.columns]
    M = d[cols].to_numpy(float)
    g = gt.to_numpy(float)
    okmask = ~np.isnan(M) & ~np.isnan(g)[:, None]
    errs = np.abs(np.where(okmask, M - g[:, None], np.nan))
    best_err = np.nanmin(errs, axis=1)
    global_best_col = min(cols, key=lambda c: mae_axis(d[c], gt)[0])
    ge = mae_axis(d[global_best_col], gt)[0]
    oracle[axis] = dict(oracle_mae=float(np.nanmean(best_err)), global_best=global_best_col,
                        global_mae=ge)
    print(f"{axis}: global-best {global_best_col} {ge:.3f} -> oracle {np.nanmean(best_err):.3f} "
          f"(headroom {ge - np.nanmean(best_err):.3f})")

# rule search: 1-feature threshold rules per axis, using observable features only
print("\n=== best 1-feature threshold rules per axis ===")
feat_pool = ["n_fasc", "n_bands", "ok", "v2_n", "v2_fasc_span", "sup_thick", "deep_thick",
             "deep_angle", "sup_angle", "pa_orient", "mt_center", "v2_pa_slope",
             "n_fasc_kept", "fasc_sign"]
for axis, cands, suf in (("PA", PA_CANDS, "_deg"), ("FL", FL_CANDS, "_mm"), ("MT", MT_CANDS, "_mm")):
    gt = d[axis.lower() + suf]
    cols = [c for c in cands if c in d.columns]
    base = min(cols, key=lambda c: mae_axis(d[c], gt)[0])
    base_mae = mae_axis(d[base], gt)[0]
    other = sorted([c for c in cols if c != base], key=lambda c: mae_axis(d[c], gt)[0])[:3]
    feats = [f for f in feat_pool if f in d.columns]
    best_rules = []
    for o in other:
        om = mae_axis(d[o], gt)[0]
        for f in feats:
            x = pd.to_numeric(d[f], errors="coerce")
            for q in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                thr = x.quantile(q)
                if not np.isfinite(thr):
                    continue
                use_o = (x <= thr) if q <= 0.5 else (x >= thr)
                pred = np.where(use_o, d[o], d[base])
                ok = gt.notna() & pd.notna(pred)
                mae = float(np.abs(np.asarray(pred)[ok.to_numpy()] - gt[ok].to_numpy()).mean())
                best_rules.append((mae, o, f, thr, "<=" if q <= 0.5 else ">=", q, base_mae))
    best_rules.sort()
    if best_rules:
        for mae, o, f, thr, op, q, bm in best_rules[:3]:
            print(f"{axis}: rule [{f} {op} {thr:.2f} -> {o} else {base}] MAE {mae:.3f} "
                  f"(vs global-best {bm:.3f}, gain {bm - mae:+.3f})")

# metric decomposition with current best-variant mix (approximation of the deployed blend)
print("\n=== metric decomposition (best single variant per axis) ===")
tot = 0.0
for axis, suf, w in (("PA", "_deg", W[0]), ("FL", "_mm", W[1]), ("MT", "_mm", W[2])):
    cands = {"PA": PA_CANDS, "FL": FL_CANDS, "MT": MT_CANDS}[axis]
    cols = [c for c in cands if c in d.columns]
    gt = d[axis.lower() + suf]
    m, _ = min((mae_axis(d[c], gt), c) for c in cols)[0], None
    bc = min(cols, key=lambda c: mae_axis(d[c], gt)[0])
    m = mae_axis(d[bc], gt)[0]
    tot += w * m
    print(f"{axis} ({bc}): MAE {m:.3f} -> weighted {w * m:.4f}")
print(f"sum weighted = {tot:.4f} (deployed best card scores 0.19747 on the LB)")
