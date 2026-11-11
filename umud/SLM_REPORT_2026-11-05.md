# SLM work report — 2026-11-05 (MuscleFM-SLM build + benchmark)

## TL;DR
The compact foundation model (MuscleFM-SLM) is built and its two benchmark kernels are training on
Kaggle GPU right now (`umud-seg-slm0` v6 bridge + `umud-seg-slm1` v5 comparison, both still RUNNING
at 18:25 UTC; Kaggle kernels finish independently of this session). The ground-truth training layer
was finally unlocked: **5,905 image/mask pairs**. Three kernel-crashing bugs were found and fixed;
the mask annotation format was decoded from the data itself. Competition submissions resume Nov 6
(today's 5/5 slots were used this morning; scored best 0.19813 public, 0.19747 offline reference).

## What was built
1. **Bridge kernel `umud-seg-slm0` v6** (builders: `umud/kaggle/build_slm0_bridge.py`):
   trains U-Net ResNet-34 (arm A control) vs U-Net SegFormer-mit_b1 (arm B) on identical protocol
   (Dice+BCE, TrivialAugment, 6 epochs, bs8, 384px, seeded 85/15 holdout of 4,000 pairs);
   picks the winner by fascicle IoU; predicts apo+fascicle masks on all 297 `IMG_*.tif` test
   images with hflip TTA; geometry layer (aponeurosis envelope line fits → MT, fascicle PCA →
   PA + FL); mm conversion via per-image bar calibration; writes `slm_member.csv` (submission
   format), `geometry_detail.csv`, `run_record.json`, both arm weights.
2. **Comparison kernel `umud-seg-slm1` v5** (`umud/kaggle/build_slm1_kernel.py`): 3-arm
   encoder comparison on identical targets/protocol — ResNet-34, mit_b1, and **US-foundation arm:
   ViT+FPN initialized from `thelets/UltraSAM-3D-Ultrasound-MVSeg`** (an UltraSam-lineage
   checkpoint found and downloaded at runtime by a HuggingFace probe; fallback chain to timm
   ViT-S augreg_in21k if unreachable).
3. **Private Kaggle dataset `umud-test-scales`** (297 rows: `image_id`, `calib_px_per_10mm`, bar
   source) so kernels convert pixels→mm autonomously.
4. **`download/umud_d32/COMPARISON.md`**: same-benchmark comparison table (fill-in when the
   kernels finish) + published-numbers context table for USFM / UltraSam / USF-MAE / EchoFM /
   MedSAM2 / nnU-Net-ResEnc / DL_Track_US / UltraTimTrack / DUSTrack, explicitly labeled as
   different benchmarks (no cross-benchmark SOTA claims).

## Findings so far
- **Data**: train layer = `image_*.tif` + exact-stem GT masks (5,905 pairs); test = `IMG_*.tif`
  (297 with bar calib). Train filenames carry no device tags → seeded 85/15 holdout protocol
  (LODV returns when devices are identifiable).
- **Mask format decoded from pixel histograms**: instance/level-coded — muscle body at one mid
  gray (e.g. 52), fascicles at distinct levels (87/88/89…), background 0/1. My original
  gray-apo/white-fasc assumption was wrong (apo channel empty on both sides → degenerate IoU 1.0).
  Fix: geometric channel decode — apo = envelope band of the closed muscle region (what MT
  measures), fascicles = interior structures (PA/FL).
- **Pilot comparison** (slm1 v4, mis-decoded targets — treat as handicap numbers):
  mit_b1 0.1329, ResNet-34 0.1318, UltraSam-init ViT+FPN 0.0575 → the US-foundation encoder did
  NOT transfer better than ImageNet inits at this scale (3D-vessel domain shift + hand FPN head).
  Definitive numbers come from v5/v6 with proper targets.

## Debug history (all fixed in persisted builders)
1. v1: glob without `.tif` → 0 pairs found (root-caused via recon dump in v2).
2. v2: LODV folds skipped (no device tags) → seeded holdout fallback.
3. v3: `np.rot90` default axes rotated the `[C,H,W]` channel dim → DataLoader shape mismatch.
4. v4/v5: `iou()` bitwise-AND of float targets with bool masks (TypeError); bridge train()
   unpacked 3 of 4 returned values; ViTSeg head resolved to 168×168 not 384×384; checkpoint key
   is `model_state_dict`.

## Not finished / why
- Arm IoUs with proper targets + the member CSV: kernels still training (ETA was ~18:50 UTC for
  the bridge, ~19:25 for slm1 v5; the session's polling window closed first).
- Offline gate of the SLM member vs the scored best (paired bootstrap vs 0.19747) and the d35
  submission cards: deliberately NOT spent today — quota 5/5 used this morning; submissions
  resume Nov 6.

## Next session (exact steps, also in `umud/daily/plans/d35.md`)
1. `kaggle kernels output kragglenote2forwork/umud-seg-slm0 -p /tmp/s0` (and `...slm1 -p /tmp/s1`).
2. Check `run_record.json` (status OK, winner, non-degenerate IoUs) + pull `slm_member.csv`.
3. Sanity: PA ~10-40°, FL ~20-120 mm, MT ~15-45 mm vs `features_ens.csv` columns.
4. Offline gate vs BEST_d34_lupa092 (0.19747), paired bootstrap 2,000×.
5. d35 batch 1 = pas092 (queued) + best card per gate; batch 2 = 2 cards; batch 3 = ONE card.

Repo state: `work/kraggle-3-in-one` pushed through `b1a825f` (all builders, plans, COMPARISON.md).
