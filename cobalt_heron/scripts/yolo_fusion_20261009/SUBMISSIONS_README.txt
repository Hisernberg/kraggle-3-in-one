Helio builder - 2026-10-09 - honest cobalt_heron models only (no 0.55 payload, no MAGFiLO/SWEFil external GT).
OOF = host-exact pooled PQ at 2048 (ch/metric.py replica) on fold-model OOF maps. YOLO OOF exists only for fold 0 (144 imgs, ch-yolo-f0),
so YOLO recipes are scored on fold 0 with 4-group inner month CV (filter also trained on folds 1-4 with YOLO feats = NaN, fold-0 rows weight 3).
Reference = s11 recipe re-created on the same protocol: fold-0 0.4144 (all-5-fold 0.409). Forest/DP decision layer tested: no gain (0.4055 vs 0.409), not used.
Inner-group PQ (4 month groups of fold 0): s11-ref 0.398/0.405/0.429/0.429 | Y-recipe 0.416/0.418/0.437/0.453 | pool 0.422/0.427/0.443/0.439
f1.csv | maps mean(r18-full-s5 4500, r34-full2 4000) [=s11 maps] -> CC t0.5 merge4 -> GBM hit filter + YOLO11m evidence feats (IoU/conf vs YOLO-full insts) thr0.35 -> 2048 x1.2 -> add uncovered YOLO insts conf>=0.5 | OOF f0 0.4304 (+0.016 vs ref, wins 4/4 groups) | exp public 0.38 (0.37-0.39)
f2.csv | same maps; joint candidate pool (U-Net CC insts + YOLO insts conf>=0.1), one learned hit scorer, greedy disjoint selection thr0.40 cov<0.2 | OOF f0 0.4320 (+0.018, wins 4/4 vs ref, 3/4 vs f1) | exp public 0.38 (0.37-0.39)
f3.csv | f1 recipe on mean(r18-full-s5, r34-full2, r34-full 2500) [=s13 maps] | OOF f0 0.4304 (recipe; 3-model test ens not OOF-testable) | exp public 0.38
f4.csv | f2 (pool) recipe on s13 maps | OOF f0 0.4320 (recipe) | exp public 0.38
f5.csv | f1 recipe on mean of all 13 U-Nets (4 full + 9 fold models) | OOF f0 0.4304 (recipe; OOF 2-model mean beat best single by +0.003) | exp public 0.37-0.38
ref_s11_recreation.csv | s11 recipe re-created (not for submission unless a calibration anchor is wanted) | OOF f0 0.4144, 5-fold 0.409 | ~0.37
All files: format OK (filament_id,segmentation_rle COCO-RLE 2048), 0 overlapping px, 177/180 images with preds (3 low-signal images empty), min area >=224 px.
