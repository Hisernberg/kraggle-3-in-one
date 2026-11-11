"""Our own re-implementation of the T08 / anubisyy (v20) onset feature rows, written from a reading of the public
notebook `anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517` (queue_features.Panel.features). No downloaded code is
executed; only its released model files (LightGBM text / CatBoost) are loaded as data.

Features read only the 12-slot history [T-12, T) of speed, flow and eligibility plus statics and the origin time,
so the same function serves official windows (window_history) and the pseudo windows (filled masked-view history).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

REL = Path(os.environ.get("TFB_DATA_ROOT", "/home/urad/tfbroot/data/kaggle_public"))
HISTORY, HORIZON = 12, 6


def panel_names():
    cfg = json.loads((REL / "config" / "corridors.json").read_text())["panels"]
    return [e["corridor_id"] for e in cfg if e["corridor_id"] not in {"D12_I405_N", "D12_I405_S"}]


def _avg(x):
    ok = np.isfinite(x); n = ok.sum(0)
    return np.where(n > 0, np.nansum(x, 0) / np.maximum(n, 1), np.nan)


def _minf(x):
    r = np.where(np.isfinite(x), x, np.inf).min(0)
    return np.where(np.isfinite(r), r, np.nan)


class T08Panel:
    def __init__(self, name: str, panel_no: int):
        self.name, self.panel_no = name, panel_no
        nd = REL / "corridors" / name / "network"
        fd = pd.read_csv(nd / "fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id")
        topo = pd.read_csv(nd / "lwr_mainline_topology.csv", dtype=str)
        self.links = fd.link_id.astype(str).tolist()
        self.idx = {l: i for i, l in enumerate(self.links)}
        f0 = sorted((REL / "corridors" / name / "train" / "mainline_states" / "year_month=2030-06").glob("*.parquet"))[0]
        loc = pd.read_parquet(f0, columns=["link_id", "milepost"]).drop_duplicates("link_id")
        loc["link_id"] = loc.link_id.astype(str)
        self.mp = loc.set_index("link_id").reindex(self.links).milepost.to_numpy(np.float32)
        assert np.isfinite(self.mp).all(), name
        self.order = np.argsort(self.mp)
        self.rank = np.empty(len(self.links), np.int32); self.rank[self.order] = np.arange(len(self.links))
        self.lanes = fd.lanes.to_numpy(np.float32); self.vf = fd.free_speed_kmh.to_numpy(np.float32)
        self.cap = fd.capacity_vph.to_numpy(np.float32); self.length = fd.length_km.to_numpy(np.float32)
        self.cut = (fd.v_cut if "v_cut" in fd else 0.6 * fd.free_speed_kmh).to_numpy(np.float32)
        self.inc = self._adj(topo, "incoming_link_ids"); self.out = self._adj(topo, "outgoing_link_ids")

    def _adj(self, topo, col):
        r, c = [], []
        for row in topo.itertuples(index=False):
            a = str(row.link_id)
            if a not in self.idx:
                continue
            for b in str(getattr(row, col)).split(";"):
                if b in self.idx:
                    r.append(self.idx[a]); c.append(self.idx[b])
        m = sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(self.links),) * 2)
        m.data[:] = 1.0
        return m

    def _nb(self, g, v):
        ok = np.isfinite(v).astype(np.float32); n = g @ ok; s = g @ np.nan_to_num(v, nan=0.0)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)

    def _mpn(self, v, off):
        r = self.rank + off; ok = (r >= 0) & (r < len(self.links))
        return np.where(ok, v[self.order[np.clip(r, 0, len(self.links) - 1)]], np.nan)

    def onset_rows(self, speed, flow, elig, origin: pd.Timestamp) -> np.ndarray:
        """[L, 84] rows for horizon step 6 (condition queue_onset)."""
        L = len(self.links)
        assert speed.shape == (HISTORY, L) and flow.shape == speed.shape
        s = speed.astype(np.float32) / self.vf[None]; f = flow.astype(np.float32) / self.cap[None]
        s = np.where(np.isfinite(s), s, np.nan); f = np.where(np.isfinite(f), f, np.nan)
        ok = np.isfinite(s)
        pos = np.where(ok, np.arange(HISTORY)[:, None], -1).max(0)
        cols = np.arange(L)
        ls = np.where(pos >= 0, s[np.maximum(pos, 0), cols], np.nan)
        lf = np.where(pos >= 0, f[np.maximum(pos, 0), cols], np.nan)
        age = np.where(pos >= 0, HISTORY - 1 - pos, 12)
        rs, ps, rf, pf = _avg(s[-3:]), _avg(s[-6:-3]), _avg(f[-3:]), _avg(f[-6:-3])
        qh = np.where(ok, s <= (self.cut / self.vf)[None], False)
        lq = ls <= self.cut / self.vf
        hour = origin.hour + origin.minute / 60
        C = lambda v: np.full(L, v, np.float32)
        F = [C(self.panel_no), np.arange(L, dtype=np.float32), self.vf, self.cap / np.maximum(self.lanes, 1),
             self.lanes, self.length, C(hour), C(np.sin(2 * np.pi * hour / 24)), C(np.cos(2 * np.pi * hour / 24)),
             C(origin.dayofweek), C(0), ls, lf, C(0) + age, rs, ps, rs - ps, rf, pf, rf - pf,
             _avg(s), _minf(s), _avg(f), np.nanmax(np.where(np.isfinite(f), f, -np.inf), 0),
             ok.mean(0), qh.sum(0).astype(np.float32), lq.astype(np.float32),
             self._nb(self.inc, ls), self._nb(self.out, ls), self._nb(self.inc, lf), self._nb(self.out, lf),
             self._nb(self.inc, lq.astype(np.float32)), self._nb(self.out, lq.astype(np.float32)),
             C(np.isfinite(s).mean()), C(np.nanmin(s) if np.isfinite(s).any() else 1.0)]
        F[25] = np.where(np.isfinite(F[25]), F[25], np.nan)
        F[24] = np.where(np.isfinite(F[24]), F[24], np.nan)
        for lag in (1, 2, 3, 6, 9, 12):
            F.extend((s[-lag], f[-lag]))
        F.extend((_minf(s[-3:]), _minf(s[-6:]), _avg(s[-3:]) - _avg(s[-6:-3]), _avg(f[-3:]) - _avg(f[-6:-3]),
                  self._nb(self.inc, _avg(s[-3:])), self._nb(self.out, _avg(s[-3:])),
                  self._nb(self.inc, _avg(f[-3:])), self._nb(self.out, _avg(f[-3:])),
                  self._nb(self.inc, qh[-1].astype(np.float32)), self._nb(self.out, qh[-1].astype(np.float32))))
        F.append(self.mp)
        for off in (-2, -1, 1, 2):
            nm = self._mpn(self.mp, off)
            F.extend((np.abs(nm - self.mp), self._mpn(ls, off), self._mpn(rs, off), self._mpn(lf, off),
                      self._mpn(rf, off), self._mpn(lq.astype(np.float32), off)))
        X = np.column_stack(F + [C(HORIZON)]).astype(np.float32)
        X[~np.isfinite(X)] = np.nan
        return X
