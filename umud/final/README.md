# UMUD final submission — Muscle Architecture in Ultrasound Data

**Kaggle competition:** [umud-challenge-muscle-architecture-in-ultrasound-data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data)
**Team:** Kragglenote2 Forwork · **License:** GPL-3.0 (see `../LICENSE` — to be added at final selection)

## What this is
A fully automated, re-runnable pipeline that estimates pennation angle (PA, deg), fascicle length (FL, mm)
and muscle thickness (MT, mm) from B-mode ultrasound images. It blends two independent re-runnable members
with hand-tuned, documented weights (host-approved hyperparameters, topic 744191):

1. **Own pipeline** (this repo, `umud/` package): device scale parsing (`umud/scale.py`) → aponeurosis/fascicle
   U-Nets (`umud/seg.py`, ResNet-34 @512×768 + ResNeXt50 @640×960 ensemble, trained on the competition masks)
   → geometry (`umud/geometry.py`: robust aponeurosis line fits, inner-edge MT, weighted-PCA fascicle lines,
   depth-resolved v2 estimators) → submission (`umud/predict.py`, config `configs/a5_pa_avg.json`).
2. **DL_Track_US 0.3.1** (Apache-2.0, by the competition host): the official VGG16 U-Net apo+fascicle models and
   DLTrack's own geometry (`gui_helpers.do_calculations`), re-run headlessly by
   `kaggle/build_dltrack_kernel.py` (weights: OSF 7mjsc v0.3.1; scale fallback: `umud/scale.py`).

The blend (`scripts/blend_v2.py` + `scripts/build_dl_blend.py`) applies per-device, level-matched corrections that
are documented physics, not leaderboard fits (non-square-pixel geometry: Juniper R 1.12, Lumify R 0.87, etc.).

## Reproduce (end-to-end, offline after setup)
```bash
pip install -r requirements.txt
# member 1: train + infer the own pipeline (~1 h GPU) or reuse the committed frozen inputs
python umud/seg.py --data <competition dir> --out work/seg
python -m umud.predict features --probs work/seg --out work/features.csv
python -m umud.predict submit --features work/features.csv --config configs/a5_pa_avg.json \
    --test-dir <competition dir>/test_images_v2/test_set_v2 --out work/pipeline.csv
# member 2: DLTrack re-run (the kernel fetches the Apache-2.0 weights from OSF at runtime)
python umud/kaggle/build_dltrack_kernel.py --out build/dltrack && kaggle kernels push -p build/dltrack
# blend (manual weights, level-matched per device)
python scripts/build_dl_blend.py --member <dltrack output> --out submission.csv
```
On a NEW dataset with no leaderboard this exact chain runs unchanged: both members are code + weights, and every
blend constant is a documented, hand-set hyperparameter. No public-Leaderboard artefact is required at any step.

## Provenance & licenses
- DL_Track_US 0.3.1: Apache-2.0 (github.com/PaulRitsche/DL_Track_US, tag v0.3.1; weights: OSF 7mjsc v0.3.1).
- This repo's own code: GPL-3.0. Trained weights: from the competition's own training data (CC BY-NC-SA).
- External expert sets used for validation only (osf.io/xbawc, CC-BY-4.0; NeuAge VL; GM video).

## FAIR metadata
See `codemeta.json` (identifier, license, requirements) — generated at final selection.
