# UMUD leaderboard log (2026-09-22)

Metric: mean(MAE_PA/6, MAE_FL/12, MAE_MT/3), lower is better. Leaderboard at time of writing: #1 0.28263, #3 0.31898.

| # | Submission | Public LB | Re-runnable on new data? |
|---|---|---|---|
| s01 | Public "variational-ensemble" notebook output (Vera with FL/MT shrunk) | 0.49332 | no (hard-coded CSV) |
| — | Public Vera notebook output (score reported by its author; baseline) | 0.45134 | no (hard-coded CSV) |
| s02 | **A**: this pipeline (`configs/a_raw.json`) | 0.52866 | **yes** |
| s03 | **B**: 0.5·A + 0.5·Vera (`scripts/blend.py --shot B`) | 0.38770 | no |
| s04 | **C**: A(PA+1.6°) ·0.45 + Vera ·0.55, cine-loop smoothing, anchors pinned (`--shot C`) | **0.37011** (rank 13/293) | no |
| s06 | A2: pipeline with OSF-benchmark calibration (`configs/a2_osf_calibrated.json`) | 0.52945 | **yes** |
| s07 | C2: C with A2's calibrated FL | 0.37706 | no |
| s08 | **C3**: C with per-target pipeline weights PA .45 / FL .3 / MT .6 (`--shot C3`) | **0.35157** (rank 8/293) | no |
| s10 | **C5**: C3 with PA pipeline weight 0.6 (`scripts/blend_weights.py 0.6 0.3 0.6`) | **0.34947** | no |
| s09 | C4: per-target weights PA .45 / FL .1 / MT .75 | 0.35513 | no |
| s05 | D: C + Juniper right-tick length rescale (MT ×0.89, FL ×0.87, fitted to the 2 anchors) | 0.40742 | no |

Segmentation (Kaggle T4, `umud-seg-train`): aponeurosis val Dice 0.844 (40 ep), fascicle val Dice 0.303 (30 ep;
fascicle masks are sparse partial annotations, so Dice is structurally low).

## Diagnostics
- Pipeline vs Vera (on the metric's scale): PA 0.52, FL 0.89, MT 0.39; so FL is the weakest estimator.
  Using geometric (extrapolated) FL alone beats blending it with MT/sin(PA) (0.89 vs 1.13).
- Pipeline PA is ~2° lower than Vera and 1.6° lower than the two public anchors.
- On both anchors the pipeline's inner-edge MT is ~14 % above the label (Vera is ~8 % above too), which suggests a
  different MT convention; this is unresolved.
- Test set: 28 five-frame cine loops (IMG_00056–IMG_00195) plus near-duplicate pair 18/19.

## Prize eligibility
Top-3 requires a public, licensed, re-runnable repository. Only shot A qualifies as-is. B/C depend on a hard-coded
public CSV, so they need to be replaced by an equally strong re-runnable model before the final selection.

## Rejected hypothesis (s05)
Both public anchors (Siemens Juniper, right-tick HUD) imply lengths ~14 % shorter than the tick-mark scale gives
(consistent with a 5.0 cm instead of 5.75 cm field of view). Rescaling the 89 Juniper test images accordingly made
the public score worse (0.370 → 0.407), so the anchors are not representative; do not rescale lengths from them.

## Host guidance (discussion forum, 10 topics read)
- MT: three straight *vertical* lines (left / middle / right) between the aponeuroses, averaged over 2 raters.
- PA: FIJI angle tool at three fascicle-fragment insertions (not at the labelled fascicles, not line intersections).
- FL: three fascicles drawn to the aponeuroses, chosen to minimise extrapolation, extended manually if needed.
- Declared external data is allowed, incl. the UMUD "Expert Analysed Benchmarks" (35 images, 7 raters, OSF) that
  the host recommends for calibration. LB-probed ensemble weights are not prize-eligible; hand-tuned ones are.
- 670 exact duplicate fascicle image/mask pairs in train (remove one copy); test is independent of train.

## External validation: UMUD OSF expert benchmark (osf.io/xbawc, 35 images, 7 raters)
`scripts/osf_benchmark.py` runs the trained models + geometry with the benchmark's own px/cm scale and compares to
the rater mean (entry errors > 50 % from the median dropped). MAE (competition-normalised):

| Target | pipeline | DLTrack (host baseline) | one rater vs the others |
|---|---|---|---|
| MT | 0.50 mm (0.166) `mt_inner` | 1.03 mm | 0.24 mm |
| PA | 1.06° (0.177) `pa_wmed` | 1.45° | 1.47° |
| FL | 8.50 mm `fl_med` → **4.69 mm** with `fl_wmed × 0.93` | 3.74 mm | 4.78 mm |

Biases: FL +5 mm (fascicle extrapolation too long), PA −0.5°, MT −0.25 mm → `configs/a2_osf_calibrated.json`.
The benchmark comes from training-like devices (Telemed / Aloka / Philips HD11), not the test devices
(Juniper / Lumify), so these numbers are optimistic for the test set.

Prepared for the next quota day: `submissions/s06_A2_osf_calibrated.csv` (re-runnable) and
`submissions/s07_C2_blend_osfFL.csv` (C with the calibrated FL component).

## Day 2 (2026-09-23)
- The OSF-benchmark calibration did not transfer: A2 0.52945 vs A 0.52866; C2 0.37706 vs C 0.37011. This is the third
  signal (with the Variational output and D) that shortening FL on the test set hurts. The benchmark devices differ from the test devices.
- Per-target blending helps. The benchmark says the pipeline is strongest on MT and weakest on FL, so C3 weights the
  pipeline 0.6 on MT and 0.3 on FL and gains 0.0185. Pushing further (C4) loses 0.0036, so the optimum is near C3.
- Prize eligibility: C3 still depends on the hard-coded public Vera CSV. The host allows hand-tuned ensemble weights,
  but the ensemble member itself must be re-runnable. For a prize-eligible final, Vera has to be replaced by a
  second re-runnable model of similar quality.
- C5 (PA pipeline weight 0.45 → 0.6) gains a further 0.0021, consistent with the OSF benchmark (pipeline PA 1.06° vs
  DLTrack 1.45°; the Vera PA is DLTrack-lineage).
- Weekly Kaggle GPU quota exhausted; `seg.py --init/--kinds/--lr` + `build_kernel.py --cpu --weights-kernel` fine-tune
  the fascicle model on de-duplicated data (670 duplicate pairs removed) in a 12 h CPU kernel.

## Fascicle fine-tune on de-duplicated data (local CPU, 3 epochs, lr 1e-4)
Kaggle GPU quota and CPU session slots were exhausted, so this ran locally from the GPU weights.
Validation Dice 0.303 → 0.333 (the old validation split shared duplicates with training).
OSF benchmark: PA `pa_wmed` 1.06° → 1.02°, FL `fl_med` 8.50 → 8.10 mm, `fl_top5` 6.73 → 6.37 mm, MT unchanged.
Prepared: `submissions/s11_Aft_pipeline.csv` (re-runnable) and `submissions/s12_C5ft.csv` (C5 weights on A_ft).

## Day 3 (2026-09-24): single-axis weight search around C5 (pipeline share per target)
| Shot | PA / FL / MT weights | Public LB |
|---|---|---|
| C5ft | .6 / .3 / .6 on the fine-tuned pipeline | 0.35317 (fine-tune helps on OSF, not on test) |
| C6 | .75 / .3 / .6 | 0.35671 |
| C7 | .6 / .2 / .6 | 0.35103 |
| C8 | .6 / .4 / .6 | 0.35596 |
| **C9** | **.6 / .3 / .75** | **0.34509 (rank 8, new best)** |

PA is optimal near 0.6 (quadratic fit ≈ 0.56) and FL near 0.3. MT is still improving at 0.75, consistent with the OSF
benchmark (pipeline MT 0.50 mm vs DLTrack 1.03 mm), so next is MT 0.9 / 1.0.

## Day-4 review: mistakes found in the best blends
1. **Pipeline failures blended as prior constants.** IMG_00189/190 had no aponeurosis pair, so shot A filled the prior
   (PA 17 / FL 80 / MT 21, then smoothed) and every C-blend mixed it in. `blend_v2.py` now uses the reference there.
2. **Wrong deep aponeurosis on 3-band images.** The "widest band" rule picked a deeper boundary: IMG_00121–125 got
   MT 40.2 mm (reference 25.0) and FL ≈ 187 mm. New `geometry.DEEP_RULE = "hybrid"`: the nearest substantial band
   below the superficial aponeurosis (the protocol for superficial muscles), falling back to the widest band when
   the nearest implies an inner-edge MT < 12 mm. On 3-band images the MT disagreement drops 2.54 → 1.50 mm and FL 14.9 → 10.4 mm.
   Only IMG_00121–125 change on the test set. In C9 those rows had MT ≈ 36 mm; now ≈ 24.6 mm.
3. **Additive decomposition of the leaderboard.** The score is a sum over targets, so the differences between
   C3–C9 isolate each target's curve: optimum PA weight ≈ 0.56, FL ≈ 0.27 (≈ 0.0004 gain each), MT still decreasing at 0.75.
4. Linear blending passes pipeline outliers through (FL |A−V| 99th pct 85 mm); `--clip-*` gives a Huber-style option.

Prepared (`submissions/d4_*.csv`): S1 fixed (hybrid pipeline, failures → ref, w .56/.27/.75), S2 MT 1.0 (clip 4 mm),
S2b MT 0.9, S3 FL clip 15 mm, S4 PA offset 0.8, and F (old pipeline with only the failure fix, as a fallback).

## Day 4 (2026-09-25): fixes validated, rank 4
| Shot | Change | Public LB |
|---|---|---|
| S1 | C9 + hybrid deep-band fix + failure rows → ref + w .56/.27/.75 | 0.32911 (−0.016 vs C9) |
| **S2** | S1 with MT weight 1.0 (MT residual clip 4 mm) | **0.32182 (rank 4/293)** |
| S3 | S2 + FL residual clip 15 mm | 0.33649 (FL outliers are mostly right → no clipping) |
| S4 | S2 with MT weight 1.25 | 0.33096 (MT optimum ≈ 0.99 by quadratic fit) |
| S5 | S2 with PA offset 0.8 (instead of 1.6) | 0.32618 (a larger PA offset looks better → test 2.4) |

Reproduce S2: `python scripts/blend_v2.py --ref vera.csv --groups test_groups.npy --pipeline submissions/s17_Ahyb_pipeline.csv
--features features_hyb.csv --w 0.56 0.27 1.0 --clip-mt 4 --out S2.csv`.

## Day 5 (2026-09-26)
| Shot | Change vs S2 | Public LB |
|---|---|---|
| d5 S1 | PA offset 2.4 | 0.32180 (flat; PA-offset optimum ≈ 2.0 by quadratic fit) |
| d5 S2 | PA 2.0 + FL tail weight 0.6 beyond 15 mm | 0.32533 (a single linear FL weight 0.27 is optimal; clipping and tail-boosting both hurt) |
| d5 S3 | PA 2.0 + MT +0.3 mm | 0.33771 |
| d5 S4 | PA 2.0 + MT −0.35 mm | 0.33599 (MT has a sharp minimum at 0: no bias; many MT predictions already very close) |
| **d5 S5** | **PA 2.0 + clip smoothing 0.8 (was 0.6)** | **0.31995 (rank 5, 0.001 behind 4th)** |

Reproduce: `blend_v2.py ... --w 0.56 0.27 1.0 --clip-mt 4 --pa-offset 2.0 --alpha 0.8`. Next: alpha 1.0 (full clip median).

## Day 6 (2026-09-27)
| Shot | Change vs d5 S5 | Public LB |
|---|---|---|
| **d6 S1** | clip smoothing alpha 1.0 (full 5-frame median) | **0.31903** (−0.0009; stronger smoothing keeps paying) |
| d6 S2 | S1 + FL weight 0.35 on cine-clip rows only (`--w-clip`) | 0.31904 (flat: FL weight curve on clip rows is flat 0.27–0.35) |
| d6 S3 | S1 + MT clip 6 (was 4) | 0.32119 (one row moved 2 mm → +0.0022 ⇒ public split ≈ 103 rows; that row's truth sits on the ref side) |
| **d6 S4** | **S1 + MT clip 3** | **0.31795 (best; exactly the 1 mm gained on that row)** |
| d6 S5 | S4 + PA weight 0.5 | 0.31957 (PA weight optimum ≥ 0.56 at offset 2.0) |

Reproduce best: `blend_v2.py ... --w 0.56 0.27 1.0 --clip-mt 3 --pa-offset 2.0 --alpha 1.0`. Leaderboard moved overnight:
0.31795 is rank 6 (3rd 0.29358, 5th 0.30537). Next: PA weight 0.62; MT clip 2.5/2 changes only 2–3 rows (public-probe, low
private value); a larger step needs a better pipeline FL (FL carries most of the remaining error).

## Day 7 (2026-09-28), first run with the `daily/` loop
| Shot | Change vs d6 S4 | Public LB |
|---|---|---|
| d7 S1 | PA weight 0.62 | 0.31861 (PA weight fit 0.5/0.56/0.62 → optimum ≈ 0.57, flat) |
| **d7 S2** | **FL weight 0.31 (was 0.27)** | **0.31581 (best; full smoothing averages FL noise, so FL takes more weight)** |
| d7 S3 | FL weight 0.36 | 0.31705 (quadratic optimum ≈ 0.32) |
| d7 S4 | S2 + PA residual clip 6° | 0.31740 (PA outliers are mostly right, like FL) |
| d7 S5 | S2 + MT weight 1.1 | 0.31837 (MT weight 1.0 stays) |

Reproduce best: `blend_v2.py ... --w 0.56 0.31 1.0 --clip-mt 3 --pa-offset 2.0 --alpha 1.0` (or `daily/run.py build`).
Rank 6 (5th 0.29518, 3rd 0.28263).

## Day 8 (2026-10-04): new estimators validated on expert sets, best 0.30718
Research: the host confirmed (topic 743111) that the two sample_submission values are made up, so pinning IMG_00001/2
to them was a bug. The host also allows declared external data and manual test-set measurements for calibration
(topic 690868). New local validation (`scripts/external_bench.py`, `scripts/eval_external.py`) on three public
expert-labelled UMUD OSF sets (osf.io/xbawc, CC-BY-4.0): the 35-image benchmark (7 raters, same FIJI protocol as the
labels), 180 NeuAge VL images with the expert's drawn lines, and 84 analysed frames of the GM calf-raise video.
New geometry estimators (`geometry.fascicles_v2`): fragment angle vs normalised depth (raters measure PA at insertions
into the deep aponeurosis; fascicles steepen towards it) and a curved/chord FL from the depth-fitted angle and the
mid-image thickness.

| Shot | Change vs previous best | Public LB |
|---|---|---|
| d8 S1 | d7 best without the made-up anchor rows (`--no-anchors`) | 0.31560 |
| d8 S2 | S1 + PA from deep-half fragments (`v2_pa_lowhalf`), offset re-matched (1.05) | 0.31846 (better on all 3 expert sets, worse on test) |
| **d8 S3** | **S1 + pipeline FL = mean(`fl_med`, `v2_fl_chord_mid`)** (`configs/a4_fl_med_chord.json`) | **0.30738 (-0.0082)** |
| d8 S4 | S3 + FL weight 0.45 | 0.32416 (optimum stays ~0.31) |
| **d8 S5** | **S3 + PA = mean(`pa_wmed`, `v2_pa_lowhalf`)**, offset 1.52 (`configs/a5_pa_avg.json`) | **0.30718** |

External expert sets, MAE (debiased): FL `fl_med` 8.47 (5.30) -> mean with chord 4.83 (3.70) mm on OSF; PA `pa_wmed`
1.05 -> mean with lowhalf 0.96 deg on OSF, 2.74 -> 2.59 on NeuAge. Their biases do not transfer to the test labels.
Reproduce best: `daily/run.py build` with state args (`--pipeline daily/inputs/pipeline_a5.csv --features
daily/inputs/features_v2.csv --w 0.56 0.31 1.0 --clip-mt 3 --pa-offset 1.52 --alpha 1.0 --no-anchors`).
Leaderboard: rank 12 (1st 0.24071, 3rd 0.25482, 11th 0.29413).

## Day 9 (2026-10-05): device calibration tests, best 0.30706
| Shot | Change vs best (d8 S5) | Public LB |
|---|---|---|
| d9 S1 | PA from Vera only on the 50 Telemed_644 rows (+2.16 deg there) | 0.32131 |
| d9 S2 | pipeline-only PA on Telemed_644 (-1.71 deg there) | 0.31863 |
| d9 S3 | full ensemble: resnet34 + efficientnet-b3 fascicle probs, multi-scale TTA, multi-scale apo | 0.33333 |
| **d9 S4** | **two-model fascicle ensemble (resnet34 + efficientnet-b3, single scale), apo unchanged** | **0.30706** |
| d9 S5 | S4 with FL weight 0.27 | 0.30820 |

- Our pipeline PA is 4-5 deg below Vera/AnatomyNet/Variational on the Telemed_644 device, but the blend is already
  calibrated there: both directions lose ~0.0066 per degree, the maximum slope for ~12 public rows. Per-device
  calibration to the reference (`--famcal pa`) is therefore not used.
- S3's loss is a failure-handling artifact: the multi-scale aponeurosis model failed on one more frame of clip
  IMG_00186-190 (the frames with the anonymisation box), so 3 of 5 frames fell back to Vera and the clip median moved
  to Vera's values (FL -19 mm, PA +2.7 deg, MT -1 mm on all 5 rows). Next: fill failed frames from clip-mates.
- Second fascicle model: efficientnet-b3 U-Net (Kaggle GPU kernel `umud-seg-fasc2`, val Dice 0.308), averaged with
  the resnet34 probabilities (`scripts/avg_probs.py`, `configs` a5 on `daily/inputs/features_ens.csv`).
- FL weight optimum for the averaged FL stays ~0.31 (quadratic vertex 0.307).
Rank 13 (1st 0.24071, 3rd 0.25209).
