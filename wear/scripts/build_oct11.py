"""Build the 2026-10-11 WEAR candidate files on top of the current best (L11, ref 57035427).

Run from the work dir after bootstrap.sh:  python3 build_oct11.py [base_csv]
Writes cands/O11_*.csv and prints what each one changes.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from band import band_fix  # noqa: E402

base_path = sys.argv[1] if len(sys.argv) > 1 else "own_subs/sub_57035427.csv"
meta = pd.read_csv("/home/user/wear_data/test/test_meta_data.csv")
sb = meta.sbj_id.values
base = pd.read_csv(base_path)
cur = base.target_feature.astype(int).values
q = (np.load("kout/wear-good-fork/final_probabilities.npz")["test"]
     + np.load("kout/wear-good-fork2/final_probabilities.npz")["test"]) / 2


def save(name, y, note):
    pd.DataFrame({"id": base.id, "target_feature": y}).to_csv(f"cands/{name}.csv", index=False)
    ch = np.where(y != cur)[0]
    print(name, "changes", len(ch), [(int(w), int(sb[w]), int(cur[w]), int(y[w])) for w in ch], "|", note)


# O11_V3: variant-pair share band (complex share 45-58%), windows chosen by fork decoded probabilities.
# OOF +0.0002/+0.0003 on both fork runs; on L11 it moves 3 sbj_23 push-ups(complex) windows to push-ups.
save("O11_V3", band_fix(cur, q, sb, 0, 999, use_pairs=True), "variant share band")

# O11_A2: activity->null where both fork decodes, window, fusion and IMU-only models agree on null,
# for classes that are not arm-confusable (str-triceps 2197 sbj_25, lunges 9783 sbj_23).
y = cur.copy()
for w, a in ((2197, 6), (9783, 16)):
    if y[w] == a:
        y[w] = 0
save("O11_A2", y, "all-source act->null")
