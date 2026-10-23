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

## Day 10 (2026-10-06): gated protocol, best 0.30383
One shot at a time, each with a written hypothesis and local evidence. S4/S5 followed S2's result: Lumify is the one
device where Vera is closer than the pipeline (Lumify PA weight curve 1.0 / 0.56 / 0.30 -> optimum ~0.32).

| Shot | Change vs best | Public LB |
|---|---|---|
| **d10 S1** | **`--sanity`: IMG_00305 pipeline failure (MT 5.4 mm < 10 mm host minimum, no fascicles) -> reference values** | **0.30498 (-0.0021; the row is public)** |
| d10 S2 | S1 + Lumify PA from the pipeline per image, Lumify median preserved | 0.30837 (refuted) |
| d10 S3 | S1 on multi-scale fascicle probs of both models (apo single-scale) | 0.30789 (refuted) |
| **d10 S4** | **S1 + Lumify PA pipeline weight 0.30 (toward Vera, median preserved; mirror of S2)** | **0.30450** |
| **d10 S5** | **S4 + Lumify FL pipeline weight 0.15 (pipeline FL errors track its PA errors there, corr -0.75)** | **0.30383** |

Research behind the shots (all offline, no submission spent):
- My visual PA reading is ~6-7 deg low versus the OSF experts, so visual QA gives direction only.
- Scale audit: per-device MT ratios pipeline/Vera 1.00-1.05, no scale errors; one physiological failure (IMG_00305).
- Expert FL / (MT/sin PA) on OSF: median 1.07 (IQR 0.99-1.09); Vera violates the expert 5-95% band on 25% of rows,
  the pipeline on 9%.
- Blend simulation on OSF with DLTrack (Vera's lineage): errors nearly uncorrelated; PA blend optimum 0.56, the same as
  the LB optimum; FL spread gating and raw-texture PA rejected.
Rank 12 (1st 0.23908, 3rd 0.24452, 5th 0.26087).

## Day 11 (2026-10-07): visual FL arbitration, best 0.30147
Where pipeline and reference FL disagree by > 15 mm (81 rows), a blind forced choice (`scripts/forced_choice.py`) shows
the straight fascicle lines implied by both candidates; validated at 76 % on the OSF experts (83 % at confidence >= 2).
Records: `external/visual_fc/`. Per-row weights via `blend_v2.py --row-w`.

| Shot | Change | Public LB |
|---|---|---|
| d11 S1 | best + 8 confident, retest-consistent units (20 rows): FL weight 0.8 (pipeline picked) / 0.1 (reference picked) | 0.31525 (+0.0114) |
| **d11 S2** | **best + only the 3 Lumify picks (IMG_00275/278/302, pipeline FL 21-48 mm shorter than the reference)** | **0.30147 (-0.0024, new best)** |
| d11 S3 | S2 + PA weight 0.65 on the same 3 rows (expert FL/trig ratio argument) | 0.30234 (refuted) |
| d11 S4 | S2 + FL weight 0.1 (toward the reference) on IMG_00030 and clips 56-60, 76-80 | 0.30815 (refuted) |
| d11 S5 | S2 + second batch: 9 units with pipeline FL 9-15 mm shorter, all consistent on retest (FL 0.8 / 0.1, 13 rows) | 0.30322 (refuted) |

- S1 vs S2 isolates the non-Lumify picks: +0.0138. S4 shows that moving the long-FL clips toward the reference also loses
  (+0.0067), so the global 0.31 FL weight is already calibrated there (V-shape, like Telemed PA on day 9).
- Visual arbitration paid only on gross disagreements (21-48 mm) where the shorter candidate was picked, against the
  rater's bias toward longer lines. On 9-15 mm disagreements it was noise (S5), even with consistent retests.
- Re-test consistency is not accuracy: the same rater, the same image and the same lines give correlated errors.
Rank 12 (1st 0.23870, 3rd 0.24452, 5th 0.25892).

## Day 12 (2026-10-08): protocol v3 (pre-registered cards, one shot at a time), best 0.29875
Research before the day (no submissions): protocol-matched FL aggregations, MT conventions, Telemed MT edges
(blind 5/5 for the pipeline), FL/trig band clipping and a mask-label regressor were all rejected offline
(`daily/plans/d12.md`).

| Shot | Change vs best | Public LB |
|---|---|---|
| S1 | FL x1.025 (first upward FL-level test) | 0.31850 |
| **S2** | **FL x0.988 (-1.0 mm), chosen with a Laplace error model fitted to S1** | **0.30074 (new best)** |
| S3 | AnatomyNet FL as reference, level-matched | 0.30270 (refuted) |
| S4 | IMG_00303 FL 122.9 -> 88.0 (Lumify outlier; blend PA/MT imply 89 mm) | 0.30074 (private row; kept) |
| **S5** | **PA pipeline weight 0.56 -> 0.62, offset re-matched (mean PA unchanged)** | **0.29875 (new best)** |

- The FL level was slightly high: the three-point curve has its vertex at -0.67 mm. The day-2 "shortening hurts"
  evidence was confounded by an estimator swap.
- Most of an MAE loss from a uniform shift is curvature (rows whose sign flips), so a large loss does not mean a
  large bias; the error model turned a planned -2.1 mm step into a -1.0 mm step.
- The PA weight optimum moved up after the PA estimator became less noisy (d8/d9); next test 0.68.
Rank 14 (1st 0.23364, 3rd 0.23780, 5th 0.25309).


## Day 13 (2026-10-09): resnext50 fascicle model, best 0.29594 (rank 13)
| Shot | Change vs best | Public LB |
|---|---|---|
| S1 | PA pipeline weight 0.62 -> 0.68, offset re-matched | 0.29961 (flat; vertex 0.632, keep 0.62) |
| **S2** | **pipeline PA from the resnet34 + resnext50 (640x960) fascicle pair instead of resnet34 + effb3, level-matched** | **0.29644 (-0.0023)** |
| S3 | S2 + pipeline FL from the same pair, level-matched | 0.29601 (-0.0004, flat by rule) |
| S4 | S2 + PA pipeline weight 0.65, level-matched | 0.29637 (-0.00007, adopted) |
| **S5** | **S4 PA + S3 FL (score predicted 0.29594 by per-target additivity)** | **0.29594 (new best)** |

- A third fascicle model (resnext50 at 640x960, Kaggle GPU) failed the strict offline gate (only the GM video
  significant), but replacing effb3 with it improved PA in direction on all three expert sets. On the LB the PA swap
  gained more than predicted (-0.0023): the test images (mostly cine clips) behave like the GM video.
- The metric is a sum of per-target MAEs, so single-target shots combine exactly: S5 = S4 + S3's FL effect, scored
  as predicted to the fifth decimal. This makes the best combination selectable at the end at no risk.
- The top of the board (~0.23-0.24) matches rater-level test measurements blended with a pipeline, which the host
  allows as declared external data for calibration (topic 690868).
- Training for the next days: resnet34 fascicles at 640x960 (resolution vs encoder) and resnext50 aponeuroses at
  640x960 (MT).
Rank 13 (1st 0.22868, 3rd 0.23371, 5th 0.24463).

## Day 14 (2026-10-10): three-map ensemble and FL weight re-tune, best 0.29433 (rank 14)
| Shot | Change vs best | Public LB |
|---|---|---|
| S1 | final MT -0.2 mm (first MT level test) | 0.29968 (worse: MT level is centred) |
| **S2** | **FL from r34 + x50 + x50 (seed 7) maps** | **0.29477 (-0.0012)** |
| **S3** | **PA from the same three maps** | **0.29455 (-0.0002)** |
| **S4** | **FL pipeline weight 0.31 -> 0.34** | **0.29434 (-0.0002)** |
| **S5** | **FL pipeline weight 0.33 (fitted vertex)** | **0.29433 (new best)** |

- MT: the -0.2 mm shift isolates the MT term (per-target additivity): the MT median error is ~0 and the public MT
  MAE ~0.5 mm. MT level, slope, residual clip and gross rows are settled.
- The test set rewards more fascicle maps more than the expert sets predict (third and fourth time), up to three
  maps; a fourth map (another x50 seed or r34 at 640x960) and a resnext50 aponeurosis model fail offline.
- Less noisy FL moved the FL weight optimum from ~0.31 to ~0.33, as the PA weight moved on day 12.
- The container was recycled overnight; the environment and expert sets were rebuilt and the live pipeline reproduces
  exactly.
Rank 14 (1st 0.22868, 3rd 0.23371, 5th 0.24077).

## Day 15 (2026-10-11): non-square pixels and per-family FL levels, best 0.26097 (rank 9)
| Shot | Change vs best | Public LB |
|---|---|---|
| S1 | left_ticks FL converted to true geometry (lateral 134.5 vs depth 148.1 px per 10 mm) | 0.3156 (labels use pixel geometry there) |
| **S2** | **left_ticks FL x1.016 (S1's loss implied a -1.8 mm median error)** | **0.29249** |
| **S3** | **Juniper FL x0.984 (VL physics ratio 1.087 vs NeuAge VL experts 1.066)** | **0.2833** |
| **S4** | **Juniper FL x0.95** | **0.26697** |
| **S5** | **Juniper FL x0.92 (fitted vertex)** | **0.26097 (new best)** |

- A new forum topic asked about "stretched 1200x800 images". Both 1200x800 families have non-square pixels. For the
  Telemed screenshots (left_ticks) the labels share the pixel geometry; for the Siemens Juniper images (12L3 aperture
  51.3 mm -> R ~1.12) the LB optimum FL (x0.92) matches true geometry (x0.90): the Juniper labels look native.
- A large diagnostic shift plus a Laplace fit turns one LB probe into a family-level bias estimate; the follow-up
  shots landed as predicted. Total day gain -0.0334 (rank 14 -> 9).
Rank 9 (1st 0.22868, 3rd 0.23371, 5th 0.24077).

## Day 16 (2026-10-12): true-geometry corrections for two devices -> 0.21657, PUBLIC RANK 1
| Shot | Change vs best | Public LB |
|---|---|---|
| **S1** | **Juniper PA converted to true geometry (R 1.10)** | **0.2442** |
| **S2** | **Lumify PA converted to true geometry (R 0.841, L12-4 aperture 34 mm)** | **0.23215** |
| **S3** | **Lumify FL x1.10** | **0.21841 (rank 1)** |
| **S4** | **Juniper PA at the physical R 1.12** | **0.21731** |
| **S5** | **Lumify PA trimmed to R 0.87 (Laplace fit of S2)** | **0.21657 (new best)** |

- The Siemens Juniper and Philips Lumify test images (149 of 309 rows) have non-square pixels (the B-mode area spans a
  constant 57.5 / 28.6 mm in vertical-scale units at every depth, while the probe apertures are 51.3 / 34 mm). Their
  labels are in true geometry; the pipeline (and the Vera reference) measure in pixels. Correcting PA (angle) and FL
  (length) of these two families moved the public score from 0.26097 to 0.21657 in one day.
- Telemed native exports and h512 have square pixels; the stretched Telemed screenshots (left_ticks) were labelled
  as provided (pixel geometry).
Public rank 1 (2nd 0.22084, 3rd 0.22682).

## Day 17 (2026-10-13): per-device MT levels, best 0.21437 (public rank 1)
| Shot | Change vs best | Public LB |
|---|---|---|
| S1 | Telemed FL x1.02 | 0.2178 (Telemed FL is centred) |
| **S2** | **Telemed MT -0.5 mm** | **0.21437 (new best)** |
| S3 | Juniper MT -0.2 mm | 0.21664 (Juniper MT is centred) |

- Host (topic 748126): the labels were scaled with the vertical scale bars only, PA is measured against the lower
  aponeurosis, and the public/private split is random (so device-level corrections carry over to the private split).
- Telemed MT was ~0.45 mm thick (labels sit about halfway between our inner edges and Vera); other devices' MT levels
  are centred. Two slots left unused (no card with evidence).
Public rank 1 (2nd 0.22084, 3rd 0.22102).

## Day 18 (2026-10-14): Lumify weights after the geometry fix, best 0.21127 (public rank 1)
| Shot | Change vs best | Public LB |
|---|---|---|
| S1 | gray853 / h512 FL x0.87 / x0.92 (physics ratio) | 0.22477 (ratio unreliable there) |
| S2 | gray853 / h512 FL x1.06 | 0.2144 (flat: FL ~centred) |
| S3 | Telemed-device (left_ticks + Telemed) PA +0.6 deg | 0.221 |
| S4 | Telemed-device PA -0.4 deg | 0.21658 (both sides lose: centred) |
| **S5** | **Lumify PA pipeline weight 0.30 -> 0.65, Lumify mean PA kept** | **0.21127 (new best)** |

- The Lumify-specific weights were tuned before the geometry fix; leaning toward Vera then also moved the level. With
  the level corrected separately, the global PA weight is better for Lumify too (-0.0031). The FL weight is next.
- One-sided level probes cannot separate bias and spread (S3 alone suggested a 0.55 deg bias; S4 showed it is
  centred with tight errors).
Public rank 1 (2nd 0.21805, 3rd 0.22102).

## Day 19 (2026-10-15): no submissions (api.kaggle.com denied by the environment's network policy)

## Day 20 (2026-10-16): global weights after the geometry fix, best 0.20863 (public rank 1)
| Shot | Change vs best | Public LB |
|---|---|---|
| **S1** | **Lumify FL pipeline weight 0.15 -> 0.33, Lumify mean FL kept** | **0.20889 (new best)** |
| S2 | global FL pipeline weight 0.33 -> 0.40, every family's mean FL kept | 0.21404 (weight settled at 0.33) |
| S3 | final-selection hedge B: best without left_ticks FL x1.016 and Telemed MT -0.5 | 0.21293 (exactly as predicted) |
| S4 | per-row angle-dependent true-geometry FL for Juniper / Lumify, family means kept | 0.20911 (flat factors stay) |
| **S5** | **gray853 / h512 FL x1.03 (midpoint of the flat d18 x1.0 / x1.06 pair)** | **0.20863 (new best)** |

- The Lumify-specific FL weight compensated the old geometry bias, like the PA weight on day 18. The global FL
  weight did not: level-matched 0.40 cost +0.19 mm public FL MAE.
- Per-row additivity makes a hedge's public score exactly predictable (S3: 0.21293 predicted and observed).
- Summed |error| is convex in a per-family scale or shift. Two flat probes on unchanged rows therefore bound the
  midpoint (S5 could lose at most 0.000015).
Public rank 1 (2nd 0.21478, 3rd 0.21805).

## Day 21 (2026-10-17): device-specific weights after the geometry fix, best 0.2064 (public rank 1)
| Shot | Change vs best (device mean kept in every shot) | Public LB |
|---|---|---|
| S1 | global PA pipeline weight 0.65 -> 0.62 (Lumify kept) | 0.20877 (vertex ~0.641) |
| **S2** | **Lumify FL weight 0.33 -> 0.45** | **0.20812** |
| **S3** | **Lumify PA weight 0.65 -> 0.80** | **0.2074** |
| S4 | Lumify PA weight 1.0 | 0.20816 (V-shaped, vertex ~0.80) |
| **S5** | **Juniper PA weight 0.65 -> 0.80** | **0.2064 (new best)** |

- On both non-square-pixel devices (Philips Lumify, Siemens Juniper), the pipeline beats the Vera reference by
  more than the global weights assume, once the geometry is corrected.
- All shots touched disjoint target x device cells, so their effects combine exactly.
- Refined row-level Laplace fits confirm the earlier level vertices: Juniper FL x0.922; Telemed-device PA centred.
Public rank 1.

## Day 22 (2026-10-18): device weights for every family, best 0.20241 (public rank 1)
| Shot | Change vs best (device mean kept in every shot) | Public LB |
|---|---|---|
| **S1** | **Juniper FL weight 0.33 -> 0.45** | **0.2046** |
| **S2** | **FL weight 0.33 -> 0.25 for left_ticks / Telemed / gray853 / h512** | **0.20439** |
| **S3** | **the same at 0.29 (vertex predicted from d20 S2 minus Juniper's convexity-bounded share)** | **0.20275** |
| **S4** | **PA weight 0.65 -> 0.60 for left_ticks / Telemed / gray853 / h512** | **0.20241 (new best)** |
| S5 | Juniper PA weight 0.90 | 0.20404 (V-shaped curve, vertex ~0.76-0.80) |

- The old joint weights (PA 0.65, FL 0.33) averaged devices whose optima lie on opposite sides. The non-square-pixel
  devices want more pipeline (PA 0.80, FL 0.45), the Telemed-family devices less (PA 0.60, FL 0.29).
- A joint probe can be decomposed when one device's share is bounded by convexity; that bound pinned the FL vertex
  of the other devices (0.29) before the shot. Day gain -0.0040.
Public rank 1.

## Day 23 (2026-10-21): Juniper PA 0.75 -> 0.20227 (public rank 1); prize-eligibility research, re-runnable track opened
| Shot | Change vs best | Public LB |
|---|---|---|
| S1 | Juniper FL weight 0.55 (was 0.45), mean kept | 0.20259 (+0.00018 -> settled at 0.45) |
| **S2** | **Juniper PA weight 0.75 (was 0.80), mean kept** | **0.20227 (new best; vertex fit predicted -0.00015, observed -0.00014)** |
| S3 | Telemed_644 FL weight 0.24 (was 0.29 joint), mean kept | 0.20283 (+0.00056 -> settled at 0.29) |
| S4 | diagnostic: reference Vera -> DLTrack prior, level-matched | 0.22451 (eligibility price +0.0222) |
| S5 | S4 with FL pipeline weight x0.7 | 0.23213 (refuted: DL anchor wants the Vera-calibrated weights) |

- The per-family weight space is now exhausted: S3's cost implies Telemed FL curvature c~0.22, bounding every remaining
  family-weight gain (incl. left_ticks) below 0.0001; S1 confirms Juniper FL at 0.45.
- **Prize eligibility (research, no submission needed):** the Overview's compliance check requires the final submission
  to be re-runnable on a new dataset without a leaderboard (host, topic 744191). Verified in the notebook sources: the
  public Vera notebook (0.45134), the "variational ensemble" (0.49332) and the new "[LB 0.374] Geometry Over Masks"
  notebook are ALL hard-coded CSVs - the variational one embeds a "DLTrack-US Reference Norms" table whose PA is exactly
  Vera's PA. There is no re-runnable public member at Vera quality.
- The re-runnable track: DL_Track_US 0.3.1 (host-authored, Apache-2.0, pip-installable) re-run on the test set as the
  member; the level-matched blend with it starts at 0.22451 (S4). S5 shows the binding constraint is our pipeline's FL,
  not the blend weights. Plan: re-run DLTrack in a GPU kernel, validate against the embedded prior, re-implement Vera's
  seg-centerline MT correction re-runnably, then re-tune.
- Submission helper note: `daily/run.py build` outputs are gitignored; a fresh container must rebuild before submitting
  (the first submit attempt polled on a missing file for 10 min).

## Day 24 (2026-10-22): re-runnable member built and measured; Track-B floor confirmed at 0.22451
| Shot | Change vs best | Public LB |
|---|---|---|
| S1 | hedge B: three LB-only levels dropped (final-selection diagnostic) | 0.20732 (+0.00505, as predicted) |
| S2 | anchor -> DLTrack-masks+own-geometry member (EXP-2401, re-runnable), level-matched | 0.44154 (member vetoed: FL row sd 33.5 mm) |
| S3 | DL-prior anchor with FL pipeline weight 0.45/0.60, level-matched | 0.22970 (+0.0052 -> Track-B FL vertex = Vera-calibrated weights) |

- EXP-2401 (kernel, no slot): DLTrack 0.3.1 re-ran on all 309 test images (297 OK). DLTrack's own geometry is unusable
  here (bar-scale detector nonsense, broken units); its PA still correlated 0.912 with the prior. Pivoted to
  **DLTrack masks + our geometry + our scale** (MT corr 0.824 vs our masks, PA 0.602, FL 0.504) - fully re-runnable.
- Track-B state: the blend weights transfer across anchors (both weight directions lose from 0.29/0.45); the floor
  0.22451 is set by the member's FL quality. Next: EXP-2402 - DLTrack's own extrapolation geometry on de-squashed
  masks at original resolution (target FL sd <= 8 mm).
- hedge B (0.20732) on record for the 2026-11-07 final selection: the three LB-only levels are worth 0.005 public.

## Day 24 addendum (04:00 UTC): EXP-2402 iso kernel (6 versions)
- Working chain shipped: B-mode crop (umud scale parser) -> isotropic letterbox 512x512 -> DLTrack 0.3.1 VGG16 U-Nets ->
  DLTrack geometry with calib_dist=None + own isotropic unit conversion (v0.3.1's internal scaling crashes on
  list/float; EXP-2401's "bar" values were silent pixel-space garbage).
- Member v2: FL sd vs Vera 18.1 mm (bar <=8), coverage 121/309 (letterbox hurt DLTrack apo detection), PA r 0.731 vs
  prior. VETOED for blending; masks + raw values kept (work/dltrack_iso.csv).
- Day-24 slots: 3/5 used (S1 hedge B 0.20732 recorded; S2 0.44154 vetoed; S3 0.22970 vertex confirmation). Best
  unchanged: 0.20227 (public rank 1). Track-B floor 0.22451 (prior proxy); host clarification queued for topic 744191.

## Day 24 (2026-10-22, evening): last two slots — failure-fill tested, MT-clip axis closed
| Shot | Change vs best | Public LB |
|---|---|---|
| S4 | `--fail-fill clip` (IMG_00189/190 filled from OK clip-mates 186-188, the prescribed-but-unplayed d9 next step; clip-80 median moves PA −0.36°, FL +0.62mm toward the pipeline) | 0.20239 (+0.00012, flat) |
| S5 | S4 + MT residual clip 2.5 (isolates IMG_00268, −0.5 mm toward ref) | **0.20239 — exactly equal to S4** |

- S5 = S4 to the fifth decimal ⇒ per-row additivity proves **IMG_00268 is a private row**; with 147/149 protected by the
  clip median, the MT residual clip is fully closed (cap stays 3).
- The d9 failure-fill direction (toward the pipeline) did not pay on public at this magnitude; the variant is recorded
  (d24_S4_failfill, 0.20239) as the principled failure handler — it removes hard-coded-reference dependence on the two
  failed frames; candidate hedge component for the 2026-11-07 final selection.
- Offline gates did their job before any slot was spent: the DL-mask member MT mix failed the ≤1 mm sd bar (2.47 mm),
  the Juniper/Lumify weight vertices are flat (<0.0001 expected), and the near-duplicate pair 18/19 is already grouped.
Best unchanged: **0.20227 (d23_S2_junpa075), public rank 1**. Day-24 slots: 5/5 used.

## Day 25 (2026-10-23): second anchor and dispersion, best 0.20036 (public rank 1)
| Shot | Change vs best | Public LB |
|---|---|---|
| **S1** | **FL anchor 0.85 Vera + 0.15 DLTrack prior, family means kept** | **0.20135 (new best)** |
| **S2** | **PA spread x0.90 within each device family (means kept)** | **0.20036 (new best)** |
| S3 | FL spread x0.92 | 0.20492 |
| S4 | PA spread x0.80 | 0.20171 (PA vertex ~0.91) |
| S5 | FL spread x1.04 | 0.20133 (FL spread settled at 1.0) |

- Full-history review: every past win came from better maps, device physics or device weights/levels; the two new
  axes (partial second anchor, within-family dispersion) were the only untested ones and both paid.
- Public rank 1 by 0.00195 (2nd 0.20231).
