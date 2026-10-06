"""lgb_v11 ongoing: the v5 blend with its two main components at capacity p3 (255 leaves, 900 rounds).

v5 ongoing = 0.35 og_v3 + 0.35 og_v3_noloc + 0.15 lgb_v3 + 0.15 rob_noloc_p2w, the first two at p2 (127 leaves,
600 rounds). Capacity paid off on ongoing so far (31 -> 63 -> 127 leaves: 0.852 -> 0.867 -> 0.877 CV), and the v5
capacity gain transferred to March. Here og_v3 and og_v3_noloc are retrained at p3 on all train windows (same
windows, weights and half of the extra candidates as v5); the other two components are the saved v5 ones.
Top-m expected-IoU decoding; only the ongoing rows of BASE are replaced (onset, v8, unchanged).

    T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 \\
        python -m trafficflow.t2.ongoing_v11 /home/user/work/t2/lgb_v8og11.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import tindex
from .core import PANELS8, WORK, official_windows, statics
from .robust_pipeline import blend_probs, components, decode_topm
from .submit import check

RECIPE = ["train:og_v3:p3:w:noop:0.35", "train:og_v3_noloc:p3:w:noop:0.35", "lgb_v3:0.15", "rob_noloc_p2w:0.15"]
BASE = Path("/home/user/work/t2/lgb_v8_seeds9_stack03.csv")


def main(out: str):
    comps = components(RECIPE, "queue_ongoing", "v11")
    preds, allp = {}, []
    for split in ("validation", "private"):
        B = blend_probs(comps, split, "queue_ongoing")
        preds.update(decode_topm(B[["window_id", "panel", "k", "link", "p"]]))
        allp.append(B.assign(split=split))
    pd.concat(allp).to_parquet(WORK / "probs_v11_ongoing.parquet")
    sub = pd.read_csv(BASE, dtype=str)
    Wd = pd.concat([official_windows(p, s) for p in PANELS8 for s in ("validation", "private")]).set_index("window_id")
    og = sub.window_id.map(Wd.condition).to_numpy() == "queue_ongoing"
    k = tindex(sub.timestamp) - sub.window_id.map(Wd["T"]).to_numpy() - 1
    q = sub.queue_pred.astype(int).to_numpy().copy()
    for i in np.flatnonzero(og):
        wid = sub.window_id.iat[i]
        lid = statics(Wd.loc[wid, "panel"])["lid"][sub.link_id.iat[i]]
        q[i] = int(preds[wid][k[i], lid])
    changed = int((q != sub.queue_pred.astype(int).to_numpy()).sum())
    sub["queue_pred"] = q
    sub.to_csv(out, index=False)
    print(json.dumps({"changed_vs_base": changed, **check(Path(out))}))


if __name__ == "__main__":
    main(sys.argv[1])
