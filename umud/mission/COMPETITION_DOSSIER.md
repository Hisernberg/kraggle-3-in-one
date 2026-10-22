# Competition Dossier — `umud-challenge-muscle-architecture-in-ultrasound-data`

> Living document. Maintained by `brain/zero`, filled by `ops/onboarding`.
> Facts only — hypotheses go to the Reef, results to the Ledger.
> Source of truth for day-to-day ops: `../RESULTS.md`, `../daily/state.json`, `../daily/plans/dNN.md`.

## 1. Facts

- **Competition:** UMUD Challenge: Muscle Architecture in Ultrasound Data · **URL:** https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data
- **Task (one sentence):** predict pennation angle (PA, deg), fascicle length (FL, mm) and muscle thickness (MT, mm) per ultrasound image (309 test rows, 28 five-frame cine clips).
- **Metric (exact formula):** `mean(MAE_PA/6, MAE_FL/12, MAE_MT/3)` — lower is better; additive per target and per row cell.
- **Submission format:** CSV `image_id,pa_deg,fl_mm,mt_mm` (309 rows); physiological ranges PA 5–45, FL 30–200, MT 10–50.
- **Timeline:** closes **2026-11-14 23:00 UTC**; final selection (2 submissions) due internally 2026-11-07.
- **Submissions/day:** 5 · **Team size cap:** 10 · **Prize:** CHF 3000 (1500/1000/500) · **Teams:** 459.
- **Standing (2026-10-21):** public LB **0.20227 = rank 1** (d23_S2_junpa075); #2 0.20337 (active daily).

## 2. Rules-critical clauses

- External data: allowed if public/reasonably accessible; host allows hand-tuned (manual) ensemble weights; **LB-probed automatic weighting is NOT prize-eligible**.
- **Reproducibility (the binding constraint):** top-3 provisionals must pass FAIR + OSI-license + reproducible-execution review; host criterion = "can your final submission be (re-)run on a new dataset where no leaderboard exists?" (topic 744191 + Overview). Winner license GPL-3.0.
- Verified 2026-10-21: public references Vera (0.45134), variational (0.49332), eas (0.374) are **all hard-coded CSVs** → NOT re-runnable members.
- Re-runnable member found: **DL_Track_US 0.3.1** (host-authored, Apache-2.0, pip-installable).
- 670 exact duplicate fascicle image/mask pairs in train (host topic 740356) — de-duplicated in training.

## 3. Data map (gm-data)

| File | Rows | Cols | Key | Notes |
|---|---|---|---|---|
| train apo imgs/masks v1 | 1,048 | - | image_id | U-Net training (val Dice 0.844) |
| train fascicle imgs/masks v1 | 2,761 (−670 dupes) | - | image_id | sparse partial annotations (Dice ~0.33 structural) |
| test_images_v2 | 309 | - | image_id | devices: juniper_right 91, left_ticks 90, lumify_png 58, telemed_644 50, gray853 12, h512 8 |
| OSF expert benchmark | 35 | PA/FL/MT x7 raters | image | external validation (Telemed-like devices) |
| NeuAge VL / GM video sets | 180 / 84 | expert lines | image | external validation |
| daily/inputs/* | 309 | features/pipeline/ref | image_id | frozen blend inputs (git-tracked) |

- **Device physics (settled d15–d17):** Juniper (12L3, R 1.12) and Lumify (L12-4, R 0.841→0.87) have non-square pixels, labels in TRUE geometry; left_ticks labels in pixel geometry; Telemed/h512/gray853 square pixels. Host: labels scaled with vertical scale bars; PA vs lower aponeurosis; random public/private split.

## 4. Metric dossier (gm-metric)

- Optimal behavior: minimize each MAE term independently — the metric is a SUM, so per-target/per-family changes compose EXACTLY on the public LB (verified to the 5th decimal across d13/d20).
- Public LB ≈ 103 of 309 rows (~1/3); one public FL mm ≈ 0.00081; deltas < ~0.001 are noise for generalization but exact for the touched cells.
- Traps: (1) public-only fitting of per-row constants (host warn + private risk); (2) anchors IMG_00001/2 are made-up (host topic 743111, never pinned); (3) rater bias toward longer fascicle lines in visual arbitration (d11); (4) expert-set biases do NOT transfer to test devices (A2, d8, d23 S3); (5) cine-clip smoothing is load-bearing (alpha 1.0).

## 5. Validation protocol lock (gm-validation)

- Scheme: three external expert sets (OSF 35 / NeuAge 180 / GM 84) with paired bootstrap + device-level Laplace fits on LB probes; NO CV (no regression labels on train — masks only).
- Gates (protocol v3, 2026-10-07): estimator changes must beat incumbent on ≥2 expert sets (CI excluding 0, identical rows); row-level changes need physics or validated blind visual evidence; no per-row/per-device sweeps without offline support.
- Lock status: **LOCKED 2026-10-07** (`../daily/RUNBOOK.md`).

## 6. Strategy (gm-strategy)

- Current phase: **consolidation (rank defence) + eligibility engineering** (final-72h doctrine pulled forward — the compliance check IS the endgame here).
- Two-track plan:
  - **Track A (LB):** Vera-anchored blend, exhausted weight space (d23 curvature bounds <0.0001). Remaining: hedge-B diagnostic, LB watch.
  - **Track B (prize):** re-runnable final = own pipeline + DL_Track_US 0.3.1 member, level-matched blend = **0.22451** (d23 S4). Improve: re-run DLTrack properly (GPU kernel), re-implement Vera's seg-centerline MT correction re-runnably, re-tune weights on OSF sim.
- Final-two doctrine (T-1): A = best public (Vera blend); B = best re-runnable (prize candidate). B must rank top-3 among ELIGIBLE entries.
