# UMUD — muscle architecture from ultrasound (`umud/`)

Pipeline for the Kaggle competition
[UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data):
predict pennation angle (PA, °), fascicle length (FL, mm) and muscle thickness (MT, mm) per image.

Metric: `mean(MAE_PA/6, MAE_FL/12, MAE_MT/3)`, lower is better.

## Status (2026-10-21)
Best public LB **0.20036** (public rank 1, 2026-10-23): `submissions/d25_S2_pas090.csv` = d23_S2 + FL anchor 0.85 Vera / 0.15 DLTrack prior + PA spread x0.90 within each device family (`blend_v2.py --spread`). Previous best 0.20227 `submissions/d23_S2_junpa075.csv` (Juniper PA weight 0.75; base: fascicle maps from resnet34 @512x768 + two resnext50 seeds @640x960; device-specific pipeline weights PA / FL: Lumify 0.80 / 0.45, Juniper 0.75 / 0.45, Telemed + left_ticks + small families 0.60 / 0.29, MT 1.0; true-geometry corrections for non-square pixels: Siemens Juniper PA (R 1.12), Philips Lumify PA (R 0.87); device FL levels (Juniper ~x0.92, Lumify ~x1.12, left_ticks ~x1.016, gray853/h512 ~x1.03) kept by level-matched scales; IMG_00303 FL patch; Telemed MT -0.5 mm).
Per-family weight space exhausted (day-23 curvature bounds). The per-target weights/scales are hand-tuned with offline evidence (prize-eligible per the host's criterion), but the **Vera reference CSV is a hard-coded public notebook output and NOT re-runnable** - the same is true of every other public reference (variational, eas 0.374; verified in their sources). The re-runnable final therefore uses **DL_Track_US 0.3.1** (host-authored, Apache-2.0) as the member: level-matched DLTrack-anchored blend = 0.22451 (day-23 S4), to be improved by re-running DLTrack + re-implementing Vera's seg-centerline MT correction. Reproduce the Vera blend with `python daily/run.py build NAME` (args in `daily/state.json`); pipeline `configs/a5_pa_avg.json`. External validation: `scripts/external_bench.py` + `scripts/eval_external.py`.

Layout: `umud/` package (scale, seg, geometry, predict) · `scripts/` (blends, OSF benchmark, diagnostics) · `configs/` · `kaggle/` (kernel builder) · `submissions/` (every submitted CSV) · `daily/` (5-per-day submission loop, see `daily/RUNBOOK.md`) · `RESULTS.md`.
