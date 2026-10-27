"""2026-10-27: move the K-line state by the H-line state delta (new recipe - old recipe), keeping the K-line's
public blend weights and its FD congested-branch flow reconciliation.

    python oct27_state_delta.py BASE.csv H_old.csv H_new.csv OUT.csv [ws=0.92] [wf=0.62]

BASE (K2r) state = 0.92 H11P + 0.08 public speed / 0.62 H11P + 0.38 public flow, then FD flow recon at v < 0.6 v_f
(J2: q = v w kj / (v + w), FD totals), then the half-gap common-mode flow factor at free-flow targets (K1).
H_old / H_new are full submissions from the rebuilt members through make_submission (old = H11P recipe, new = the
H13P/H14P recipe). OUT state: v = v_base + ws (v_new - v_old); q = q_base + wf (q_new - q_old), and on rows with
v < 0.6 v_f the FD flow of the new speed (as J2). Queue and ODME rows are byte-identical to BASE.
"""
import glob
import sys

import numpy as np
import pandas as pd

REL = "/home/urad/tfbroot/data/kaggle_public"
KEY = "/home/urad/tfbroot/work/tfbw/work__pub__submission_key.parquet"
base_p, old_p, new_p, out_p = sys.argv[1:5]
ws = float(sys.argv[5]) if len(sys.argv) > 5 else 0.92
wf = float(sys.argv[6]) if len(sys.argv) > 6 else 0.62
GATE = 0.6

fd = []
for f in glob.glob(f"{REL}/corridors/*/network/fd_parameters.csv"):
    d = pd.read_csv(f); d["panel"] = f.split("/")[-3]; fd.append(d)
fd = pd.concat(fd).drop_duplicates(["panel", "link_id"])
fd["w"] = fd.capacity_vph / (fd.k_jam - fd.critical_density)

key = pd.read_parquet(KEY, columns=["submission_id", "task", "panel", "link_id"])
key = key[key.task == "state"].drop(columns="task")
key["submission_id"] = key.submission_id.astype(np.int64)
key = key.merge(fd[["panel", "link_id", "free_speed_kmh", "k_jam", "w"]], on=["panel", "link_id"], how="left")
assert key.w.notna().all()

base = pd.read_csv(base_p, dtype={"queue_pred": str, "path_flow": str, "speed_kmh": str, "flow_vph": str})
st = base.task.eq("state").to_numpy()
ids = base.submission_id.to_numpy(np.int64)[st]
key = key.set_index("submission_id").loc[ids]
vb = base.speed_kmh[st].astype(float).to_numpy(); qb = base.flow_vph[st].astype(float).to_numpy()


def state(p):
    d = pd.read_csv(p, usecols=["submission_id", "task", "speed_kmh", "flow_vph"])
    d = d[d.task == "state"].set_index("submission_id").loc[ids]
    return d.speed_kmh.to_numpy(float), d.flow_vph.to_numpy(float)


vo, qo = state(old_p); vn, qn = state(new_p)
vf = key.free_speed_kmh.to_numpy(); kj = key.k_jam.to_numpy(); w = key.w.to_numpy()
fdq = lambda v, m: v * w[m] * kj[m] / (v + w[m])

g0 = vb < GATE * vf
print(f"base rows below gate {g0.sum()}, FD-recon check: median |q_base - FD(v_base)|/q = "
      f"{np.median(np.abs(qb[g0] - fdq(vb[g0], g0)) / np.maximum(qb[g0], 1)):.4f}")
dv, dq = vn - vo, qn - qo
print(f"delta rows: speed {np.mean(np.abs(dv) > 1e-6):.3f} share, mean |dv| {np.abs(dv).mean():.4f}; "
      f"flow mean |dq| {np.abs(dq).mean():.3f}")
v = np.clip(vb + ws * dv, 1.0, None)
q = np.clip(qb + wf * dq, 0.0, None)
g = v < GATE * vf
q[g] = fdq(v[g], g)
print(f"rows below gate after: {g.sum()} (entered {np.sum(g & ~g0)}, left {np.sum(~g & g0)})")
out = base.copy()
out.loc[st, "speed_kmh"] = np.round(v, 4).astype(str)
out.loc[st, "flow_vph"] = np.round(q, 4).astype(str)
out.to_csv(out_p, index=False)
print("wrote", out_p, "| changed state rows:", int(np.sum((np.abs(v - vb) > 1e-4) | (np.abs(q - qb) > 1e-4))))
