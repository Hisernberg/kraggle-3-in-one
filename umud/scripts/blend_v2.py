"""Robust per-target blend of the pipeline with a reference submission (v2).

    python scripts/blend_v2.py --pipeline A.csv --ref vera.csv --features features.csv --groups groups.npy \
        --w 0.56 0.27 0.75 --pa-offset 1.6 [--clip-pa 6 --clip-fl 15 --clip-mt 3] --out out.csv

For each target t: out = ref + w_t * clip(pipe - ref, -c_t, c_t)  (no clip = plain linear blend), then
cine-loop smoothing (0.6 toward the 5-frame median) and the two public anchors pinned. Rows where the
pipeline failed (features ok == 0) use the reference only. The PA offset is added to the pipeline PA first.
Fixes relative to the day-1..3 blends: failures no longer blend prior constants; optional outlier clipping.
"""
import argparse

import numpy as np
import pandas as pd

T = ("pa_deg", "fl_mm", "mt_mm")
RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}
ANCHORS = {"IMG_00001.tif": (17.334, 79.423, 21.778), "IMG_00002.tif": (12.876, 69.424, 15.478)}

ap = argparse.ArgumentParser()
ap.add_argument("--pipeline", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--features", required=True)
ap.add_argument("--groups", required=True, help=".npy of video-group ids in image_id order")
ap.add_argument("--w", type=float, nargs=3, required=True, metavar=("PA", "FL", "MT"))
ap.add_argument("--pa-offset", type=float, default=1.6)
ap.add_argument("--clip-pa", type=float, default=None)
ap.add_argument("--clip-fl", type=float, default=None)
ap.add_argument("--clip-mt", type=float, default=None)
ap.add_argument("--fl-tail", type=float, nargs=2, default=None, metavar=("THRESH", "W"),
                help="piecewise FL blend: weight w_FL for |pipe-ref| <= THRESH mm, weight W beyond it")
ap.add_argument("--mt-offset", type=float, default=0.0, help="mm added to the final MT (before smoothing)")
ap.add_argument("--mt-scale", type=float, default=1.0, help="global factor on the final MT")
ap.add_argument("--fl-scale", type=float, default=1.0, help="global factor on the final FL")
ap.add_argument("--w-clip", type=float, nargs=3, default=None, metavar=("PA", "FL", "MT"),
                help="weights for rows in 5-frame cine clips (clip median averages pipeline noise away)")
ap.add_argument("--alpha", type=float, default=0.6, help="cine-loop smoothing strength")
ap.add_argument("--fam-w", nargs=4, action="append", default=[], metavar=("FAMILY", "PA", "FL", "MT"),
                help="per-device weights for rows whose features.family starts with FAMILY (repeatable)")
ap.add_argument("--fam-pa-offset", nargs=2, action="append", default=[], metavar=("FAMILY", "DEG"),
                help="extra pipeline PA offset for one device family (repeatable)")
ap.add_argument("--famcal", nargs="+", default=[], choices=["pa", "fl", "mt"],
                help="per-device calibration of the pipeline to the reference: PA/MT shifted, FL scaled so that each "
                     "device family's median matches the reference's (family = features.family without the depth suffix)")
ap.add_argument("--fail-fill", choices=["ref", "clip"], default="ref",
                help="pipeline failures: 'ref' uses the reference; 'clip' uses the median pipeline values of the "
                     "frame's cine-clip mates (reference only when the whole clip failed)")
ap.add_argument("--sanity", action="store_true",
                help="also treat rows as pipeline failures when MT is outside the host's 10-50 mm range or no fascicle "
                     "angle was measured (pipeline values there are prior constants or from a wrong aponeurosis)")
ap.add_argument("--row-w", default=None,
                help="CSV (image_id, w_pa, w_fl, w_mt; blank = keep) overriding the blend weights of single rows, "
                     "e.g. visual forced-choice arbitration between the pipeline and the reference")
ap.add_argument("--no-anchors", action="store_true",
                help="do not pin IMG_00001/2 to the sample_submission values (host, topic 743111: they are made up)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

p = pd.read_csv(a.pipeline).set_index("image_id")
r = pd.read_csv(a.ref).set_index("image_id").loc[p.index]
feat = pd.read_csv(a.features).set_index("image_id").loc[p.index]
g = pd.Series(np.load(a.groups), index=p.index)
p = p.copy()
p["pa_deg"] += a.pa_offset
fam = feat["family"].astype(str) if "family" in feat else pd.Series("", index=p.index)
for fp, off in a.fam_pa_offset:
    p.loc[fam.str.startswith(fp), "pa_deg"] += float(off)
failed = feat["ok"] != 1
if a.sanity:
    failed |= ~feat["mt_inner"].between(10.0, 50.0) | feat["pa_wmed"].isna()
if a.fail_fill == "clip":
    okp = p[list(T)].where(~failed)
    clip_med = okp.groupby(g).transform("median")  # NaN when every frame of the clip failed
    fill = clip_med.where(clip_med.notna(), r[list(T)])
    p.loc[failed, list(T)] = fill.loc[failed]
else:
    p.loc[failed, list(T)] = r.loc[failed, list(T)]
if a.famcal:
    fam0 = fam.str.replace(r"_\d+$", "", regex=True)
    okr = ~failed
    for t in a.famcal:
        col = {"pa": "pa_deg", "fl": "fl_mm", "mt": "mt_mm"}[t]
        for fv in fam0[okr].unique():
            m = okr & (fam0 == fv)
            if t == "fl":
                p.loc[fam0 == fv, col] *= float((r[col][m] / p[col][m]).median())
            else:
                p.loc[fam0 == fv, col] += float((r[col][m] - p[col][m]).median())
clips = {"pa_deg": a.clip_pa, "fl_mm": a.clip_fl, "mt_mm": a.clip_mt}
out = pd.DataFrame(index=p.index)
in_clip = g.map(g.value_counts()) > 1
for i, (t, w) in enumerate(zip(T, a.w)):
    if a.w_clip is not None:
        w = pd.Series(np.where(in_clip, a.w_clip[i], w), index=p.index)
    if a.fam_w:
        w = pd.Series(w, index=p.index, dtype=float) if np.isscalar(w) else w.astype(float)
        for fw in a.fam_w:
            w[fam.str.startswith(fw[0])] = float(fw[1 + i])
    if a.row_w:
        rw = pd.read_csv(a.row_w).set_index("image_id")
        col = {"pa_deg": "w_pa", "fl_mm": "w_fl", "mt_mm": "w_mt"}[t]
        if col in rw:
            w = pd.Series(w, index=p.index, dtype=float) if np.isscalar(w) else w.astype(float)
            ov = rw[col].dropna()
            ov = ov[ov.index.isin(p.index) & ~failed.reindex(ov.index, fill_value=True)]
            w[ov.index] = ov.values
    d = p[t] - r[t]
    if clips[t] is not None:
        d = d.clip(-clips[t], clips[t])
    if t == "fl_mm" and a.fl_tail is not None:
        th, w2 = a.fl_tail
        core = d.clip(-th, th)
        out[t] = r[t] + w * core + w2 * (d - core)  # continuous; slope w inside, w2 in the tail
        continue
    out[t] = r[t] + w * d
out["mt_mm"] = out["mt_mm"] * a.mt_scale + a.mt_offset
out["fl_mm"] *= a.fl_scale
for t in T:
    out[t] = (1 - a.alpha) * out[t] + a.alpha * out.groupby(g)[t].transform("median")
if not a.no_anchors:
    for k, v in ANCHORS.items():
        out.loc[k] = v
for t, (lo, hi) in RANGES.items():
    out[t] = out[t].clip(lo, hi)
out.round(3).reset_index().to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(out)} rows, {int(failed.sum())} pipeline failures -> ref")
