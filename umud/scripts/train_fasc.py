"""Train a fascicle (or aponeurosis) U-Net locally on the DGX with umud.seg (same data, dedupe, fixed split SPLIT_SEED 42,
augmentation and loss as the Kaggle kernels umud-seg-fasc*), under the shared-GPU limits.

    python scripts/train_fasc.py --out /home/urad/data/umud/work/models/cnxs --encoder-fasc tu-convnext_small \
        --size 640x960 --bs 6 --epochs-fasc 25 [--seed 7] [--mem-gb 22]

All umud.seg.main flags pass through (--kinds fasc is the default here). Test inference is skipped: run
scripts/maps.py infer on the output dir (test / val_fasc) afterwards. A copy of the apo model is not needed.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--mem-gb", type=float, default=22.0)
ap.add_argument("--threads", type=int, default=4)
a, rest = ap.parse_known_args()
os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))

import torch  # noqa: E402

torch.set_num_threads(a.threads)
if torch.cuda.is_available():
    tot = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gb / tot))

from umud import seg  # noqa: E402

seg.infer_test = lambda *args, **kw: print("test inference skipped (use scripts/maps.py infer)", flush=True)
if "--kinds" not in rest:
    rest += ["--kinds", "fasc"]
if "--data" not in rest:
    rest += ["--data", "/home/urad/data/umud/comp"]
sys.argv = [sys.argv[0]] + rest
seg.main()
