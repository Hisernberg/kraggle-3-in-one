"""Replace the queue_ongoing rows of a full submission with those of a queue CSV (window_id,timestamp,link_id,queue_pred).

    python trafficflow/scripts/splice_ongoing.py BASE.csv QUEUE.csv OUT.csv [--split private]
Onset rows, state, ODME stay byte-identical. Prints the number of changed ongoing cells per split vs BASE.
"""
import argparse
import glob

import pandas as pd

REL = "/home/user/data/kaggle_public"
KEY = "/home/user/work/tfbw/work__pub__submission_key.parquet"

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("queue"); ap.add_argument("out")
ap.add_argument("--split", default=None, help="only splice this split (validation / private)")
a = ap.parse_args()
wi = pd.concat([pd.read_csv(f) for f in glob.glob(f"{REL}/task2/*/[vp]*/window_index.csv")])
cond = dict(zip(wi.window_id.astype(str), wi.condition))
key = pd.read_parquet(KEY, columns=["submission_id", "task", "split", "window_id", "timestamp", "link_id"])
key = key[key.task == "queue"].copy()
key["submission_id"] = key.submission_id.astype(int)
key = key[key.window_id.map(cond) == "queue_ongoing"]
if a.split:
    key = key[key.split == a.split]
q = pd.read_csv(a.queue).merge(key, on=["window_id", "timestamp", "link_id"])
assert len(q) == len(key), (len(q), len(key))
base = pd.read_csv(a.base, dtype={"speed_kmh": str, "flow_vph": str, "path_flow": str})
new = base.queue_pred.copy()
pos = pd.Series(range(len(base)), index=base.submission_id.astype(int))
idx = pos.loc[q.submission_id.values].values
new.iloc[idx] = q.queue_pred.astype(int).values
chg = new.ne(base.queue_pred)
print("changed ongoing cells:", base.loc[chg].submission_id.astype(int).map(key.set_index("submission_id").split).value_counts().to_dict(),
      "| positives before/after:", int(base.queue_pred.iloc[idx].sum()), int(new.iloc[idx].sum()))
base["queue_pred"] = new.astype(int)
base.to_csv(a.out, index=False)
print("wrote", a.out)
