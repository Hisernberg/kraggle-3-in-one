# UMUD — muscle architecture from ultrasound (`umud/`)

Pipeline for the Kaggle competition
[UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data):
predict pennation angle (PA, °), fascicle length (FL, mm) and muscle thickness (MT, mm) per image.

Metric: `mean(MAE_PA/6, MAE_FL/12, MAE_MT/3)`, lower is better.

## Status (2026-10-08)
Best public LB **0.30074** (rank 13): `submissions/d12_S4_img303.csv` (two fascicle models, physiological failure check, Lumify weights PA 0.30 / FL 0.15, visually arbitrated FL on 3 gross Lumify rows, FL level x0.988, IMG_00303 FL patch).
Reproduce it with `python daily/run.py build NAME` (args in `daily/state.json`): pipeline `configs/a5_pa_avg.json`
(FL = mean of extrapolated-fragment FL and depth-fitted chord FL; PA = mean of all-fragment and deep-half PA) blended
per target with the public Vera CSV. The pipeline alone (`kaggle/build_kernel.py --mode submit`) is the prize-eligible
part. Blends with the hard-coded Vera CSV are not. External validation on the public UMUD expert sets:
`scripts/external_bench.py` + `scripts/eval_external.py`.

Layout: `umud/` package (scale, seg, geometry, predict) · `scripts/` (blends, OSF benchmark, diagnostics) ·
`configs/` · `kaggle/` (kernel builder) · `submissions/` (every submitted CSV) · `daily/` (5-per-day submission loop, see `daily/RUNBOOK.md`) · `RESULTS.md`.

## Method
1. **Scale / FOV** (`umud/scale.py`): per-device tick-mark parser → px/mm and B-mode crop box.
2. **Segmentation** (`umud/seg.py`): two U-Nets (ResNet-34 encoder, ImageNet init) trained on the
   provided aponeurosis (1,048) and fascicle (2,761) masks; letterboxed 512×768 input, hflip TTA.
   Masks stored inverted in the training set are auto-corrected.
3. **Geometry** (`umud/geometry.py`): aponeurosis bands → robust line fits (upper edge, lower edge,
   centre); MT = perpendicular inner-edge distance averaged at 25/50/75 % of the width; fascicle
   fragments → weighted PCA lines → PA vs. the deep aponeurosis and FL by linear extrapolation to both
   aponeuroses (length-weighted median); structure-tensor PA as a second estimator.
4. **Submission** (`umud/predict.py`): documented config (`configs/*.json`) chooses estimators,
   blends geometric and trigonometric FL (MT / sin PA), applies cine-loop temporal smoothing
   (the test set has 28 five-frame clips) and clips to the physiological ranges given by the hosts.

## Run
```bash
pip install -r umud/requirements.txt
# GPU training + test inference (≈1 h on a Kaggle T4). Or build a Kaggle kernel:
python umud/kaggle/build_kernel.py --user <kaggle-user> --out build/kernel && kaggle kernels push -p build/kernel
python umud/umud/seg.py --data <competition dir> --out work/seg
cd umud && python -m umud.predict features --probs ../work/seg --out ../work/features.csv
python -m umud.predict submit --features ../work/features.csv --config configs/final.json \
    --test-dir <competition dir>/test_images_v2/test_set_v2 --out ../work/submission.csv
```
Never commit `kaggle.json` or competition data.
