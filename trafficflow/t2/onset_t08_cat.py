"""CatBoost onset member on the T08 rows (T08 CatBoost recipe: depth 7, lr 0.05, l2 10, 5 % history-dropout copies),
trained Jun-Jan and early-stopped on February, CPU.

    python -m trafficflow.t2.onset_t08_cat NAME [seed] [depth]   -> t08/model_NAME.cbm
"""
import sys

import numpy as np
from catboost import CatBoostClassifier

W08 = "/home/urad/tfbroot/work/t08"
name = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
depth = int(sys.argv[3]) if len(sys.argv) > 3 else 7
z = np.load(f"{W08}/rows.npz"); X, y, meta = z["X"], z["y"], z["meta"]
if len(sys.argv) > 4 and sys.argv[4] == "prior":
    X = np.column_stack([X, np.load(f"{W08}/rows_prior.npz")["Xr"]])
feb = meta[:, 1] >= 245
tr = ~feb; va = feb & (meta[:, 2] == 0)
m = CatBoostClassifier(iterations=3000, depth=depth, learning_rate=.05, l2_leaf_reg=10, thread_count=8,
                       random_seed=20261005 + seed, loss_function="Logloss", early_stopping_rounds=150, verbose=250,
                       train_dir=f"{W08}/catboost_{name}")
m.fit(X[tr], y[tr], eval_set=(X[va], y[va]), use_best_model=True)
m.save_model(f"{W08}/model_{name}.cbm")
print(name, "best iteration", m.get_best_iteration())
