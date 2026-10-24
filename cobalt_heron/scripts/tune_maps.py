"""Tune post-processing on OOF uint8 prob maps (2048 PNG, e.g. the public dariushafshar U-Net set) with the
exact pooled PQ at 1024, then write a submission from the matching test maps.

    CH_DATA=... CH_WORK=... python scripts/tune_maps.py --oof DIR [DIR ...] --test DIR [DIR ...] --out sub.csv
Several DIRs are averaged (map ensemble). Coordinate search over instances() parameters, honest: the maps are
fold-held-out predictions.
"""
import argparse, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, cv2, pandas as pd
from pycocotools import mask as mu
from ch.metric import PQAccumulator
from ch.post import instances
from ch.submit import to2048, rows_for

A = None


def load(dirs, stem, size=1024):
    m = np.mean([cv2.imread(str(Path(d) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for d in dirs], 0)
    return m if size == 2048 else cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)


def rles(lab):
    return [mu.encode(np.asfortranarray((lab == k).astype(np.uint8))) for k in range(1, lab.max() + 1)]


def eval_one(args):
    stem, cfg = args
    p = load(A.oof, stem)
    lab, _ = instances(p, None, **cfg)
    acc = PQAccumulator()
    for rec in GT[stem]:
        acc.add(rec, rles(lab))
    return acc


def evaluate(cfg, stems, pool):
    tot = PQAccumulator()
    for a in pool.map(eval_one, [(s, cfg) for s in stems], chunksize=8):
        tot.merge(a)
    return tot


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", nargs="+", required=True); ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--cfg", default=None, help="skip tuning, use this json cfg")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start", default=None, help="json start cfg for the search")
    ap.add_argument("--grid", default=None, help="json grid replacing the default")
    A = ap.parse_args()
    from ch.tune import gt1024
    GT = gt1024()
    stems = sorted(p.stem for p in Path(A.oof[0]).glob("*.png") if p.stem in GT)
    print("oof images with GT:", len(stems), flush=True)
    with ProcessPoolExecutor(A.workers) as pool:
        if A.cfg:
            best = json.loads(A.cfg); acc = evaluate(best, stems, pool)
        else:
            best = json.loads(A.start) if A.start else dict(t=0.5, merge=0, min_area=0, min_score=0.0)
            acc = evaluate(best, stems, pool); print("start", best, acc, flush=True)
            grid = {"t": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6], "merge": [0, 4, 8, 16],
                    "min_area": [0, 30, 80, 200], "min_score": [0.0, 0.5, 0.55, 0.6, 0.65]}
            if A.grid:
                grid = json.loads(A.grid)
            for rnd in range(2):
                for k, vals in grid.items():
                    for v in vals:
                        if v == best.get(k):
                            continue
                        c = dict(best, **{k: v}); a = evaluate(c, stems, pool)
                        if a.pq > acc.pq:
                            best, acc = c, a; print("better", best, acc, flush=True)
        print("BEST", json.dumps(best), acc, flush=True)
    rows = []
    tstems = sorted(p.stem for p in Path(A.test[0]).glob("*.png"))
    for s in tstems:
        p = load(A.test, s)
        lab, _ = instances(p, None, **best)
        rows += rows_for(s, to2048(lab, p, best["t"]))
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(A.out, index=False)
    print("wrote", A.out, len(tstems), "test images", len(rows), "instances", flush=True)
