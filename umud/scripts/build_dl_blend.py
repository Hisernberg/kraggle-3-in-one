"""Build a level-matched anchor-swapped blend from a re-runnable member CSV (Track B).

    python scripts/build_dl_blend.py --member work/dltrack_rerun.csv --agg median \
        --out daily/out/d24_S2_dlrr.csv [--w-scale 1.0]

Takes the current best args (daily/state.json), swaps --ref to the member CSV, optionally rescales the FL
pipeline weights by --w-scale (single factor), and re-centres every family mean (FL scale, MT offset) so that
per-family means match the current best submission. Two build passes: pass 1 measures the family-mean shifts,
pass 2 applies the corrections. Writes <out> and <out>.args.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

UMUD = Path(__file__).resolve().parents[1]
INPUTS = ["--ref", "daily/inputs/ref_vera_public.csv", "--groups", "daily/inputs/test_groups.npy",
          "--pipeline", "daily/inputs/pipeline_three.csv", "--features", "daily/inputs/features_ens.csv"]
FAMS = ["lumify_png", "juniper_right", "telemed_644", "left_ticks_800", "gray853", "h512"]
REF_NAME = {"lumify_png": "lumify_png", "juniper_right": "juniper_right", "telemed_644": "telemed_644",
            "left_ticks_800": "left_ticks", "gray853": "gray853", "h512": "h512"}


def set_flag(tokens, flag, fam, vals):
    out, i, found = [], 0, False
    while i < len(tokens):
        if tokens[i] == flag and tokens[i + 1] == fam:
            out += [flag, fam] + vals
            found = True
            i += 2 + len(vals)
        else:
            out.append(tokens[i])
            i += 1
    if not found:
        out += [flag, fam] + vals
    return out


def get_scale(tokens, fam):
    for i, a in enumerate(tokens):
        if a == "--fam-fl-scale" and tokens[i + 1] == fam:
            return float(tokens[i + 2])
    return 1.0


def build(args, out_csv):
    r = subprocess.run([sys.executable, "scripts/blend_v2.py", *INPUTS, *args, "--out", str(out_csv)],
                       capture_output=True, text=True, cwd=UMUD)
    if r.returncode != 0:
        sys.exit(r.stderr)
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True, help="re-runnable member CSV (image_id,pa_deg,fl_mm,mt_mm)")
    ap.add_argument("--agg", choices=["median", "mean"], default="median",
                    help="use the member's fl_mm (median) or fl_mean_mm column")
    ap.add_argument("--w-scale", type=float, default=1.0, help="multiply all FL pipeline weights by this factor")
    ap.add_argument("--ref-col", default=None, help="optional: use these columns as-is (default: aggregate)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    state = json.load(open(UMUD / "daily/state.json"))
    best_csv = UMUD / "submissions" / f"{state['best_name']}.csv"
    best = pd.read_csv(best_csv).set_index("image_id")
    feat = pd.read_csv(UMUD / "daily/inputs/features_ens.csv").set_index("image_id")
    fam = feat["family"].astype(str)

    # 1. normalise the member to a reference-format CSV (median or mean FL aggregation)
    m = pd.read_csv(a.member)
    if "fl_mean_mm" in m and a.agg == "mean":
        m["fl_mm"] = m["fl_mean_mm"]
    m = m[["image_id", "pa_deg", "fl_mm", "mt_mm"]].dropna(subset=["pa_deg", "fl_mm", "mt_mm"], how="any")
    m["fam"] = m.image_id.map(feat["family"]).astype(str)
    missing = sorted(set(best.index) - set(m.image_id))
    if missing:
        print(f"WARNING: {len(missing)} rows missing from the member; filled with per-family medians")
        fills = m.groupby("fam")[["pa_deg", "fl_mm", "mt_mm"]].median().to_dict("index")
        famof = feat["family"].astype(str)
        for x in missing:
            f = famof[x]
            fill = fills.get(f) or next((v for k, v in fills.items() if f.startswith(k)), {c: np.nan for c in ["pa_deg", "fl_mm", "mt_mm"]})
            m.loc[len(m)] = {"image_id": x, **fill, "fam": f}
    ref_path = UMUD / "daily/inputs/ref_member_agg.csv"
    m.drop_duplicates("image_id").set_index("image_id").loc[best.index].round(4).to_csv(ref_path)
    print(f"member rows: {len(m)} -> {ref_path.name}")

    args = list(state["best_args"])
    if a.w_scale != 1.0:  # scale every FL pipeline weight (global --w FL + each --fam-w FL)
        for i, t in enumerate(args):
            if t == "--w":
                args[i + 2] = f"{float(args[i + 2]) * a.w_scale:.5f}"
        out, i = [], 0
        while i < len(args):
            if args[i] == "--fam-w":
                famname, vals = args[i + 1], args[i + 2:i + 5]
                out += ["--fam-w", famname, vals[0], f"{float(vals[1]) * a.w_scale:.5f}", vals[2]]
                i += 5
            else:
                out.append(args[i])
                i += 1
        args = out

    # 2. pass 1: measure family-mean shifts
    tmp1 = UMUD / "daily/out/_dl_pass1.csv"
    build(args + ["--ref", "daily/inputs/ref_member_agg.csv", "--out", tmp1.name], tmp1)
    p1 = pd.read_csv(tmp1).set_index("image_id")
    for f in FAMS:
        sel = fam.str.startswith(REF_NAME[f])
        old = get_scale(args, f)
        ratio = float(best.loc[sel, "fl_mm"].mean() / p1.loc[sel, "fl_mm"].mean())
        args = set_flag(args, "--fam-fl-scale", f, [f"{old * ratio:.5f}"])
        dmt = float(best.loc[sel, "mt_mm"].mean() - p1.loc[sel, "mt_mm"].mean())
        cur = 0.0
        for i, t in enumerate(args):
            if t == "--fam-mt-offset" and args[i + 1] == f:
                cur = float(args[i + 2])
        args = set_flag(args, "--fam-mt-offset", f, [f"{cur + dmt:.5f}"]) if abs(dmt) > 1e-4 else args

    # 3. pass 2: final build + verification
    build(args + ["--ref", "daily/inputs/ref_member_agg.csv", "--out", a.out], a.out)
    (UMUD / "daily/out" / (Path(a.out).stem + ".args")).write_text(" ".join(args))
    fin = pd.read_csv(a.out).set_index("image_id")
    print("family means vs best (should be ~0):")
    for f in FAMS:
        sel = fam.str.startswith(REF_NAME[f])
        print(f"  {f:14s} dFL={float((fin.loc[sel,'fl_mm']-best.loc[sel,'fl_mm']).mean()):+7.3f} "
              f"dMT={float((fin.loc[sel,'mt_mm']-best.loc[sel,'mt_mm']).mean()):+7.3f} "
              f"dPA={float((fin.loc[sel,'pa_deg']-best.loc[sel,'pa_deg']).mean()):+7.3f}")
    print("row-level FL sd vs best:", round(float((fin['fl_mm'] - best['fl_mm']).std()), 2))


if __name__ == "__main__":
    main()
