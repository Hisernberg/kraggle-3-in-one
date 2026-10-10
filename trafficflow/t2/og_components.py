"""Train the saved ongoing components of the 2026-10-09 mixes (v7, v11, v12) on all train windows.

    # v11: og_v3 / og_v3_noloc at p3 on the old truth (T2_WORK=/home/user/work/t2, T2_FEAT=.../t2/feat_v3)
    T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 python -m trafficflow.t2.og_components v11
    # v7: og_v3 / og_v3_noloc / og_v2 / og_v2_noloc at p2 on the hybrid truth (re-drawn windows, t2h/feat_og)
    T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat_og python -m trafficflow.t2.og_components v7
    # v12: og_v3 / og_v3_noloc at p3 on the hybrid truth (v7's labels with v11's capacity); optional seed
    T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat_og python -m trafficflow.t2.og_components v12 [seed]

Models: $T2_WORK/model_<name>_<variant>[_s<seed>]_queue_ongoing.txt (existing files are kept). Window weights on,
half of the extra candidate windows (robust.CAND_FRAC), 2 threads (models.CFG). About 10-15 min per p2 model and
20-35 min per p3 model on 2 threads; peak RSS about 2.5 GB per job (two jobs next to a pseudo-holdout scorer
overflowed 15 GB on 2026-10-09).
"""
from __future__ import annotations

import sys

from .core import WORK
from .robust_pipeline import train_variant

SETS = {
    "v7": ("p2", ["og_v3", "og_v3_noloc", "og_v2", "og_v2_noloc"]),
    "v11": ("p3", ["og_v3", "og_v3_noloc"]),
    "v12": ("p3", ["og_v3", "og_v3_noloc"]),
}


def main(name: str, seed: int = 0) -> None:
    cfg, variants = SETS[name]
    for v in variants:
        f = WORK / f"model_{name}_{v}{f'_s{seed}' if seed else ''}_queue_ongoing.txt"
        if not f.exists():
            train_variant(v, cfg, True, cond="queue_ongoing", seed=seed).save_model(str(f))
        print(name, v, "ready", f, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
