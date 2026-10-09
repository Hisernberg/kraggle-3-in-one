"""Ongoing rows from a weighted mix of saved component models, spliced onto a base queue file.

The recipe is chosen on the Task 2 pseudo-holdout (``pseudo.py``). Components are applied to the same feature tables
as in that evaluation ($T2_FEAT, feat_v3: full-train old-truth profiles), top-m expected-IoU decoding per window; only
the ongoing rows of BASE change (onset rows are byte-identical).

    T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 \\
        python -m trafficflow.t2.ongoing_mix v7v11 /home/user/work/t2/lgb_v8og_v7v11.csv     # H16b queue
        python -m trafficflow.t2.ongoing_mix v12v11 /home/user/work/t2/lgb_v8og_v12v11.csv   # H17 queue

The components are trained by ``og_components.py`` (v7, v11, v12).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from .ongoing_v9 import onset_lines_identical, write_from_base
from .pseudo import _booster
from .robust_pipeline import blend_probs, decode_topm

BASE = Path("/home/user/work/t2/lgb_v8_seeds9_stack03.csv")
RECIPES = {
    "v7v11": [("t2h:v7_og_v3", 0.175), ("t2h:v7_og_v3_noloc", 0.175), ("v11_og_v3", 0.175), ("v11_og_v3_noloc", 0.175),
              ("t2h:v7_og_v2", 0.15), ("t2h:v7_og_v2_noloc", 0.15)],
    # 2026-10-09 (H17): v7's og_v3 / og_v3_noloc replaced by v12 (same variants on hybrid labels at p3)
    "v12v11": [("t2h:v12_og_v3", 0.175), ("t2h:v12_og_v3_noloc", 0.175), ("v11_og_v3", 0.175), ("v11_og_v3_noloc", 0.175),
               ("t2h:v7_og_v2", 0.15), ("t2h:v7_og_v2_noloc", 0.15)],
    "v7": [("t2h:v7_og_v3", 0.35), ("t2h:v7_og_v3_noloc", 0.35), ("t2h:v7_og_v2", 0.15), ("t2h:v7_og_v2_noloc", 0.15)],
}


def main(name: str, out: str, base: Path = BASE) -> dict:
    comps = [(_booster(m, "queue_ongoing"), w) for m, w in RECIPES[name]]
    preds = {}
    for split in ("validation", "private"):
        B = blend_probs(comps, split, "queue_ongoing")
        preds.update(decode_topm(B[["window_id", "panel", "k", "link", "p"]]))
    res = write_from_base(preds, Path(out), base)
    res["onset_identical"] = onset_lines_identical(base, Path(out))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
