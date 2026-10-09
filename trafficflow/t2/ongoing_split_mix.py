"""Ongoing rows from per-split recipes spliced onto a base queue CSV (onset rows byte-identical).

Like ``ongoing_mix.py``, but each split (validation = March / public, private = April) takes its own recipe, and a
split left out of the spec keeps the base file's rows unchanged. This is how the April-only adaptation files are
built: the March-adapted components (``adapt.py``) may only serve April windows (all of March is <= T for every
April window; they must not serve March windows, whose origins precede most of the March pseudo windows).

    T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 \\
        python -m trafficflow.t2.ongoing_split_mix BASE.csv OUT.csv '{"private": [["t2h:v12_og_v3", 0.175], ...]}'

Model names as in ``pseudo._booster``: "<name>" -> $T2_WORK/model_<name>_queue_ongoing.txt, "t2h:<name>" ->
/home/user/work/t2h/..., "adapt:<name>" -> $T2_WORK/adapt/...  Top-m expected-IoU decoding per window.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import tindex
from .core import K, PANELS8, official_windows, statics
from .ongoing_v9 import onset_lines_identical, write_from_base
from .pseudo import _booster
from .robust_pipeline import blend_probs, decode_topm


def base_preds(base: Path) -> dict:
    """window_id -> [K, L] bool field of the base file's ongoing rows."""
    sub = pd.read_csv(base, dtype=str)
    W = pd.concat([official_windows(p, s) for p in PANELS8 for s in ("validation", "private")]).set_index("window_id")
    og = sub.window_id.map(W.condition).to_numpy() == "queue_ongoing"
    k = tindex(sub.timestamp) - sub.window_id.map(W["T"]).to_numpy() - 1
    out = {}
    for i in np.flatnonzero(og):
        wid = sub.window_id.iat[i]
        st = statics(W.loc[wid, "panel"])
        if wid not in out:
            out[wid] = np.zeros((K, st["L"]), bool)
        out[wid][k[i], st["lid"][sub.link_id.iat[i]]] = sub.queue_pred.iat[i] == "1"
    return out


def main(base: Path, out: Path, spec: dict) -> dict:
    preds = base_preds(base)
    for split, recipe in spec.items():
        comps = [(_booster(m, "queue_ongoing"), float(w)) for m, w in recipe]
        B = blend_probs(comps, split, "queue_ongoing")
        preds.update(decode_topm(B[["window_id", "panel", "k", "link", "p"]]))
    res = write_from_base(preds, out, base)
    res["onset_identical"] = onset_lines_identical(base, out)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3]))
