# Submission log (all competitions)

## Submission policy (set by the user on 2026-10-09)

- **No automatic submissions.** Starting 2026-10-10, nothing is submitted to any competition unless the user
  explicitly says "submit" and names what to submit.
- No daily loops or Routines submit on their own. Research, building and local validation may be prepared, but each
  upload waits for the user's instruction.
- 2026-10-09 was the last day with blanket approval (all knee and CASMI slots, plus whatever else was ready that day).

## 2026-10-09

Scores are public LB; "pending" means the submission was still scoring when this log was written.

### RSNA Knee (`rsna-knee-abnormality-detection`), 5 of 5 used

| Ref | What | Kernel / version | Public |
|---|---|---|---|
| 57012226 | Fork of `heliosli/rsna-knee-abnormality-detection` (submitted by the user before this session) | `rsna-knee-abnormality-detection` v1 | **error**: dataset `rsna-knee-compact-v1` retired by its owner at 13:50 UTC |
| 57014992 | K1: Apex exact (R384 mirror × CNXT, V3 per-target weights) | `rsna-knee-apex-hardened` v1 | **0.950** |
| 57015519 | K2: K1 + C224 0.3 inside the CoAtNet side | `rsna-knee-apex-hardened` v2 | **0.950** |
| 57015521 | K3: K2 + MaVIT 0.12 | `rsna-knee-apex-hardened` v3 | 0.949 |
| 57016276 | K4: K3 with MaVIT 0.20 | `rsna-knee-apex-hardened` v4 | 0.949 |

Knee best is now 0.950 (rank 759 of 5,610). Analysis and plan: `docs/research_2026-10-09/KNEE_RESULTS_AND_NEXT.md`.

### CASMI 2026 (`enveda-CASMI26-molecule-id-mass-spectra`), 5 of 5 used

| Ref | What | Kernel | Public |
|---|---|---|---|
| 57014911 | C3: huseyin 0.421 + `GL_MH_ONLY`, `POST_ICE_LAM=0`, `FILL_25` | `casmi-c3-huseyin-glmh` v1 | pending |
| 57014934 | C1: haideptry v32 anchor | `casmi-c1-v32-anchor` v1 | pending |
| 57015095 | C4: amanatar v18 (CPU) | `casmi-c4-v18-cpu` v1 | pending |
| 57015409 | C2: v32 + PubChem JOIN | `casmi-c2-v32-pcjoin` v1 | pending |
| 57015887 | C5: flexonafft V1.1 + cross-formula GLACIER | `casmi-c5-flexon` v1 | pending |

### Solar Filament (`filament-segmentation-2026`), 5 of 5 used

Honest `cobalt_heron` models only. Code: `cobalt_heron/scripts/yolo_fusion_20261009/`. None of these files uses the
2026-09-10 0.55 entry (the copied `lamhuy8904` payload); **never select that entry as a final.**

Out-of-fold scores are host-exact PQ on fold 0, the only fold with YOLO predictions. On that fold the s11 recipe
re-created scores 0.4144.

| File | What | OOF f0 | Public |
|---|---|---|---|
| s16 (f1) | s11 maps + GBM hit filter with YOLO11m evidence, thr 0.35, + uncovered YOLO instances with conf ≥ 0.5 | 0.4304 | pending |
| s18 (f3) | s16 recipe on s13 3-model maps | (recipe) | pending |
| s17 (f2) | s11 maps, joint U-Net + YOLO candidate pool, learned scorer, greedy disjoint selection thr 0.40 | 0.4320 | pending |
| s19 (f4) | s17 recipe on s13 maps | (recipe) | pending |
| s20 (f5) | s16 recipe on the mean of all 13 U-Nets | (recipe) | pending |

### TrafficFlowBench (`2026-ieee-big-data-traffic-flow-bench`)

See `trafficflow/docs/EXPERIMENTS.md` and `trafficflow/docs/lb_log.csv` for this competition's own log.
