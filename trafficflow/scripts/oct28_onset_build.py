"""2026-10-28: onset rows from T08 probabilities (reproduced exactly: 285/285 cells) and our April-adapted models.

    python oct28_onset_build.py BASE.csv OUT.csv VAL_SPEC PRIV_SPEC
SPEC: "keep" (base rows unchanged) or "<members>@<decoder>", members = "T08" and/or model names in t08/
("model_<name>.txt"), averaged; decoder "eiou" or a threshold ("0.55" = T08's rule with top-1 fallback).
The O1 cells of the base (base onset cells that T08 does not select) are kept in every changed window.
Only T+30 onset cells change; everything else stays byte-identical to BASE.
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from trafficflow.t2.models import eiou_topm
from trafficflow.t2.onset_t08 import official, predict, t08_models, decode_t08

W08 = "/home/urad/tfbroot/work/t08"
KEY = "/home/urad/tfbroot/work/tfbw/work__pub__submission_key.parquet"
base_p, out_p, vspec, pspec = sys.argv[1:5]

rows = official(None)
T = predict(rows, t08_models())
T["t08"] = False
for _, g in T.groupby("window_id"):
    T.loc[g.index, "t08"] = decode_t08(g)
X = np.concatenate([r[5] for r in rows])


def probs(members):
    ps = []
    for m in members:
        if m == "T08":
            ps.append(T.p.to_numpy())
        else:
            ps.append(lgb.Booster(model_file=f"{W08}/model_{m}.txt").predict(X, num_threads=4))
    return np.mean(ps, 0)


sub = pd.read_csv(base_p, dtype=str)
key = pd.read_parquet(KEY, columns=["submission_id", "task", "window_id", "timestamp", "link_id"])
key = key[key.task == "queue"].copy(); key["submission_id"] = key.submission_id.astype(str)
wi = pd.concat([pd.read_csv(f) for f in __import__("glob").glob(
    "/home/urad/tfbroot/data/kaggle_public/task2/*/[vp]*/window_index.csv")])
org = dict(zip(wi.window_id, wi.forecast_origin))
key = key[key.window_id.isin(set(T.window_id))]
key["h"] = ((pd.to_datetime(key.timestamp, utc=True) - pd.to_datetime(key.window_id.map(org), utc=True))
            .dt.total_seconds() // 300).astype(int)
pos = pd.Series(np.arange(len(sub)), index=sub.submission_id)
k6 = key[key.h == 6].merge(T[["window_id", "link_id", "split", "t08"]], on=["window_id", "link_id"])
k6["row"] = pos.loc[k6.submission_id].to_numpy()
k6["base"] = sub.queue_pred.iloc[k6.row].astype(int).to_numpy().astype(bool)
assert not key[key.h != 6].submission_id.map(pos).map(lambda i: sub.queue_pred.iat[i] == "1").any()
o1 = k6.base & ~k6.t08
print("base onset cells", int(k6.base.sum()), "| T08", int(k6.t08.sum()), "| O1 extra", int(o1.sum()),
      "| base cells T08 does not have but... missing T08 cells:", int((k6.t08 & ~k6.base).sum()))
new = k6.base.copy()
for split, spec in (("validation", vspec), ("private", pspec)):
    if spec == "keep":
        continue
    mem, dec = spec.split("@")
    p = pd.Series(probs(mem.split("+")), index=pd.MultiIndex.from_frame(T[["window_id", "link_id"]]))
    sel = k6.split == split
    for wid, g in k6[sel].groupby("window_id"):
        q = p.loc[wid].reindex(g.link_id).to_numpy()
        if dec == "eiou":
            idx, _ = eiou_topm(q); a = np.zeros(len(q), bool); a[idx] = True
        else:
            a = q >= float(dec)
            if not a.any():
                a[np.argmax(q)] = True
        new.loc[g.index] = a | o1.loc[g.index].to_numpy()
for split in ("validation", "private"):
    s = k6.split == split
    print(split, "added", int((new & ~k6.base)[s].sum()), "removed", int((~new & k6.base)[s].sum()),
          "windows changed", int(k6[s & (new != k6.base)].window_id.nunique()))
q = sub.queue_pred.copy()
q.iloc[k6.row.to_numpy()] = new.astype(int).astype(str).to_numpy()
sub["queue_pred"] = q
sub.to_csv(out_p, index=False)
print("wrote", out_p)
