"""d35 rule validation: split-half holdout of the variant-switch rules on external GT.

    python /home/z/my-project/scripts/d35_validate_rules.py

Rules (learned on the full external set, now validated on held-out halves):
- FL switch: use fl_med-implied value when sup_angle <= thr OR v2_fasc_span >= thr2,
  else v2_fl_chord_mid.
- PA switch: use v2_pa_lowhalf when mt_center >= thr, else v2_pa_fitmid.
For each rule: learn thresholds on half A, evaluate metric delta on half B (and vice versa).
Report held-out deltas in metric units (weights PA/6, FL/12, MT/3, /3).
"""
import numpy as np
import pandas as pd

U = "/home/z/my-project/work/kraggle-3-in-one/umud"
lab = pd.read_csv(f"{U}/external/labels.csv")
ext = pd.read_csv(f"{U}/external/features.csv")
d = lab.merge(ext, on="image_id", how="inner").reset_index(drop=True)
rng = np.random.default_rng(2401)
d["half"] = rng.random(len(d)) < 0.5
W = {"pa_deg": 1 / 6 / 3, "fl_mm": 1 / 12 / 3, "mt_mm": 1 / 3 / 3}

def axis_mae(pred, gt, mask=None):
    ok = pred.notna() & gt.notna()
    if mask is not None:
        ok &= mask
    return float(np.abs(pred[ok] - gt[ok]).mean()), int(ok.sum())

def metric_delta(rows, axis, base_col, alt_col, rule_fn, gt_col):
    """Per-image: base value; where rule fires, alt value. Returns MAE delta vs base."""
    base = rows[base_col]
    alt = rows[alt_col]
    fire = rule_fn(rows)
    pred = base.copy()
    pred[fire] = alt[fire]
    m_base = axis_mae(base, rows[gt_col])[0]
    m_rule = axis_mae(pred, rows[gt_col])[0]
    return m_base - m_rule

def learn_fl(rows):
    """Grid thresholds for the FL rule, maximize delta on given rows."""
    best = None
    for sa in np.quantile(rows.sup_angle.dropna(), [0.2, 0.3, 0.4, 0.5, 0.6]):
        for fs in np.quantile(rows.v2_fasc_span.dropna(), [0.4, 0.5, 0.6, 0.7, 0.8]):
            def rule(r, sa=sa, fs=fs):
                return (r.sup_angle <= sa) | (r.v2_fasc_span >= fs)
            delta = metric_delta(rows, "fl", "v2_fl_chord_mid", "fl_med", rule, "fl_mm")
            if best is None or delta > best[0]:
                best = (delta, sa, fs)
    return best

def learn_pa(rows):
    best = None
    for mt in np.quantile(rows.mt_center.dropna(), [0.3, 0.4, 0.5, 0.6, 0.7]):
        def rule(r, mt=mt):
            return r.mt_center >= mt
        delta = metric_delta(rows, "pa", "v2_pa_fitmid", "v2_pa_lowhalf", rule, "pa_deg")
        if best is None or delta > best[0]:
            best = (delta, mt)
    return best

results = {}
for name, learn, mk in (("FL", learn_fl, lambda p: (lambda r: (r.sup_angle <= p[1]) | (r.v2_fasc_span >= p[2]))),
                        ("PA", learn_pa, lambda p: (lambda r: r.mt_center >= p[1]))):
    deltas = []
    for train_half, test_half in ((True, False), (False, True)):
        tr, te = d[d.half == train_half], d[d.half == test_half]
        params = learn(tr)
        rule = mk(params)
        base_mae = axis_mae(te[{"FL": ("v2_fl_chord_mid", "fl_mm"), "PA": ("v2_pa_fitmid", "pa_deg")}[name][0]], te[{"FL": "fl_mm", "PA": "pa_deg"}[name]])[0]
        delta = metric_delta(te, name, {"FL": "v2_fl_chord_mid", "PA": "v2_pa_fitmid"}[name],
                             {"FL": "fl_med", "PA": "v2_pa_lowhalf"}[name], rule,
                             {"FL": "fl_mm", "PA": "pa_deg"}[name])
        deltas.append(delta)
        results[f"{name}_heldout_{test_half}"] = dict(delta=delta, params=list(params),
                                                      base_mae=base_mae)
    w = W["fl_mm" if name == "FL" else "pa_deg"]
    print(f"{name} rule: held-out deltas {deltas[0]:+.3f}/{deltas[1]:+.3f} (raw units) "
          f"-> metric {np.mean(deltas) * w:+.4f} | params {list(params)}")

import json
json.dump(results, open("/tmp/d35_rules.json", "w"), indent=1, default=float)
print("saved /tmp/d35_rules.json")
