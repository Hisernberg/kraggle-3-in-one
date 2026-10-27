"""Local validation harness on the held-out train split (the split every seg kernel held out, see scripts/maps.py).

The competition has no regression labels on train, only masks. The harness therefore scores map-derived geometry
against the SAME geometry computed on the expert masks of unseen images, with the aponeuroses held fixed:

  val_fasc  PA / FL: umud.geometry.analyse(pred apo, candidate fascicle map) vs analyse(pred apo, GT fascicle mask)
            -> isolates fascicle-map quality (the thing 4+ maps / new encoders / iso inference change)
  val_apo   MT: analyse(candidate apo map) vs analyse(GT apo mask) (mt_inner, the submitted MT column)

Estimators are the live ones (configs/a5_pa_avg.json): PA = mean(pa_wmed, v2_pa_lowhalf | pa_wmed),
FL = mean(fl_med, v2_fl_chord_mid) * 0.9912, MT = mt_inner. Errors are reported raw and per-family debiased (the LB
blend level-matches every device, so only within-device row error survives into the score); score =
mean(PA/6, FL/12, MT/3) on debiased MAEs. Candidates are compared row-paired with a bootstrap CI vs the first one.
FL/MT of unscaled train crops use the nominal 14.82 px/mm: compare candidates, not absolute levels.

    python scripts/harness.py --set val_fasc --apo M/val_fasc/r34 --cand three=M/val_fasc/r34,fasc3,fasc5 \
        four=M/val_fasc/r34,fasc3,fasc5,fasc6 [--gt M/val_fasc/r34/gt] [--out table.csv]
A candidate is NAME=DIR[,DIR...]; relative names resolve against the first DIR's parent.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umud.geometry import analyse  # noqa: E402

TAU = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
FL_SCALE = 0.9912


def est(f: pd.DataFrame) -> pd.DataFrame:
    pa = (f.pa_wmed + f.v2_pa_lowhalf.fillna(f.pa_wmed)) / 2 if "v2_pa_lowhalf" in f else f.get("pa_wmed")
    fl = (f.fl_med + f.v2_fl_chord_mid.fillna(f.fl_med)) / 2 * FL_SCALE if "fl_med" in f else np.nan
    ok = f.ok == 1
    return pd.DataFrame({"pa_deg": pa.where(ok), "fl_mm": pd.Series(fl, index=f.index).where(ok),
                         "mt_mm": f.get("mt_inner", pd.Series(np.nan, index=f.index)).where(ok)}, index=f.index)


def _read(p):
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)


def _mean_map(dirs, name):
    acc, n = None, 0
    for d in dirs:
        x = _read(Path(d) / "probs" / name)
        if x is None:
            continue
        x = x.astype(np.float32)
        if acc is not None and x.shape != acc.shape:
            x = cv2.resize(x, (acc.shape[1], acc.shape[0]))
        acc = x if acc is None else acc + x
        n += 1
    return None if acc is None else (acc / n).astype(np.uint8)


def _one(job):
    iid, ppm, apo_src, fasc_src, mode = job
    if mode == "gt_fasc":
        ap, fp = _read(Path(apo_src) / "probs" / f"{iid}_apo.png"), _read(Path(fasc_src) / f"{iid}.png")
    elif mode == "gt_apo":
        ap = _read(Path(apo_src) / f"{iid}.png")
        fp = np.zeros_like(ap)
    else:
        ap = _mean_map(apo_src, f"{iid}_apo.png")
        fp = _mean_map(fasc_src, f"{iid}_fasc.png")
    if ap is None or fp is None:
        return dict(image_id=iid, ok=0)
    if fp.shape != ap.shape:
        fp = cv2.resize(fp, (ap.shape[1], ap.shape[0]))
    try:
        r = analyse(ap, fp, ppm)
    except Exception:  # geometry edge cases on odd train frames
        r = dict(ok=0)
    r["image_id"] = iid
    return r


def features(crops, apo_src, fasc_src, mode, cache: Path | None, workers=8):
    if cache is not None and cache.exists():
        return pd.read_csv(cache).set_index("image_id")
    jobs = [(r.image_id, r.px_per_mm, apo_src, fasc_src, mode) for r in crops.itertuples()]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_one, jobs, chunksize=4))
    f = pd.DataFrame(rows).set_index("image_id")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        f.to_csv(cache)
    return f


def boot_ci(d, rng, n=4000):
    bs = np.array([rng.choice(d, len(d)).mean() for _ in range(n)])
    return np.percentile(bs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["val_fasc", "val_apo"])
    ap.add_argument("--apo", required=True, help="dir whose apo maps are used (val_fasc) / the GT dir root (val_apo)")
    ap.add_argument("--gt", default=None, help="GT mask dir (default <apo>/gt)")
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    apo = Path(a.apo)
    gt = Path(a.gt) if a.gt else apo / "gt"
    crops = pd.read_csv(apo / "crops.csv")
    fam = crops.set_index("image_id").family.str.replace(r"_\d+$", "", regex=True)
    cache = apo.parent / "_feat"
    if a.set == "val_fasc":
        ref = est(features(crops, apo, gt, "gt_fasc", cache / "gt_fasc.csv", a.workers))
        targets = (a.targets or "pa_deg,fl_mm").split(",")
    else:
        ref = est(features(crops, gt, None, "gt_apo", cache / "gt_apo.csv", a.workers))
        targets = (a.targets or "mt_mm").split(",")
    E = {}
    for c in a.cand:
        name, ds = c.split("=", 1)
        dirs = [Path(d) if "/" in d else apo.parent / d for d in ds.split(",")]
        apo_src = [apo] if a.set == "val_fasc" else dirs[:1]
        f = features(crops, apo_src, dirs, "pred", cache / f"{a.set}_{name}.csv", a.workers)
        E[name] = est(f)
    names = list(E)
    rng = np.random.default_rng(0)
    rows = []
    for t in targets:
        ok = ref[t].dropna().index
        for k in names:
            ok = ok.intersection(E[k][t].dropna().index)
        print(f"\n=== {t} (n={len(ok)} rows with GT and every candidate; debiased per family)")
        err = {}
        for k in names:
            e = E[k][t][ok] - ref[t][ok]
            err[k] = (e - e.groupby(fam[ok]).transform("median")).abs()
            for fv in sorted(fam[ok].unique()) + ["ALL"]:
                m = ok if fv == "ALL" else ok[fam[ok] == fv]
                rows.append(dict(cand=k, target=t, family=fv, n=len(m), mae_raw=(E[k][t][m] - ref[t][m]).abs().mean(),
                                 bias=(E[k][t][m] - ref[t][m]).median(), mae=err[k][m].mean(),
                                 norm=err[k][m].mean() / TAU[t]))
        tab = pd.DataFrame([r for r in rows if r["target"] == t])
        print(tab.pivot(index="family", columns="cand", values="mae")[names].round(3).to_string())
        for k in names[1:]:
            d = (err[k] - err[names[0]]).values
            lo, hi = boot_ci(d, rng)
            flag = "BETTER" if hi < 0 else ("WORSE" if lo > 0 else "ns")
            print(f"  {k} vs {names[0]}: {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] {flag}")
    r = pd.DataFrame(rows)
    if a.out:
        r.to_csv(a.out, index=False)
    al = r[r.family == "ALL"].pivot(index="cand", columns="target", values="norm")
    print("\nnormalised debiased MAE (ALL):")
    print(al.loc[names].round(4).to_string())


if __name__ == "__main__":
    main()
