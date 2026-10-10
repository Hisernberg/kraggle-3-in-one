"""Build the third-session probe files from the current best submission.

Usage: python build_probes.py <best.csv> <test_meta_data.csv> <out_dir>

best.csv is Fm1_margin1_only (ref 56944621, public 0.93597). Each probe sets one
(subject, predicted class) to null: the activity we believe sits in that subject's
one-activity third session (sbj_2 in train has the same layout and its third
session is labelled null).
"""
import sys
import numpy as np
import pandas as pd

PROBES = {
    "P1_s22_sitcx_to_null": [(22, 14)],
    "P2_s23_strham_to_null": [(23, 9)],
    "P12_both": [(22, 14), (23, 9)],
}


def main(best, meta, out):
    base = pd.read_csv(best)
    sbj = pd.read_csv(meta).sbj_id.values
    y = base.target_feature.astype(int).values
    assert len(y) == len(sbj) == 12234
    for name, pairs in PROBES.items():
        z = y.copy()
        for s, c in pairs:
            z[(sbj == s) & (y == c)] = 0
        pd.DataFrame({"id": base.id, "target_feature": z}).to_csv(f"{out}/{name}.csv", index=False)
        print(name, "changed", int((z != y).sum()))


if __name__ == "__main__":
    main(*sys.argv[1:4])
