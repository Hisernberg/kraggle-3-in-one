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

Best: **C5 0.435** (was 0.328); C1 and C4 0.433, C3 0.412.

**C2 (PubChem JOIN) lost 0.038 against its parent C1.** This notebook ranks its PubChem channel by popularity as well as fingerprint score, so the top-50 join pulled in different candidates than in huseyin's branch, where the join was measured at +0.011. Do not reuse the JOIN on this lineage without popularity turned off and offline validation first.

| Ref | What | Kernel | Public |
|---|---|---|---|
| 57014911 | C3: huseyin 0.421 + `GL_MH_ONLY`, `POST_ICE_LAM=0`, `FILL_25` | `casmi-c3-huseyin-glmh` v1 | 0.412 |
| 57014934 | C1: haideptry v32 anchor | `casmi-c1-v32-anchor` v1 | **0.433** |
| 57015095 | C4: amanatar v18 (CPU) | `casmi-c4-v18-cpu` v1 | 0.433 |
| 57015409 | C2: v32 + PubChem JOIN | `casmi-c2-v32-pcjoin` v1 | 0.395 |
| 57015887 | C5: flexonafft V1.1 + cross-formula GLACIER | `casmi-c5-flexon` v1 | **0.435** (best) |

### Solar Filament (`filament-segmentation-2026`), 5 of 5 used

Honest `cobalt_heron` models only. Code: `cobalt_heron/scripts/yolo_fusion_20261009/`. None of these files uses the
2026-09-10 0.55 entry (the copied `lamhuy8904` payload); **never select that entry as a final.**

Out-of-fold scores are host-exact PQ on fold 0, the only fold with YOLO predictions. On that fold the s11 recipe
re-created scores 0.4144.

| File | What | OOF f0 | Public |
|---|---|---|---|
| s16 (f1) | s11 maps + GBM hit filter with YOLO11m evidence, thr 0.35, + uncovered YOLO instances with conf ≥ 0.5 | 0.4304 | 0.37 |
| s18 (f3) | s16 recipe on s13 3-model maps | (recipe) | 0.37 |
| s17 (f2) | s11 maps, joint U-Net + YOLO candidate pool, learned scorer, greedy disjoint selection thr 0.40 | 0.4320 | 0.37 |
| s19 (f4) | s17 recipe on s13 maps | (recipe) | 0.37 |
| s20 (f5) | s16 recipe on the mean of all 13 U-Nets | (recipe) | 0.36 |

### TrafficFlowBench (`2026-ieee-big-data-traffic-flow-bench`), 5 of 5 used

Details: `trafficflow/docs/EXPERIMENTS.md` (2026-10-09 section). Reproduce with `trafficflow/run_20261009.sh`.

| Ref | File | Parent | Change | Public |
|---|---|---|---|---|
| 57016351 | H15b_monthbias | H11P (0.87249) | Task 1 month speed bias | 0.87257 |
| 57016422 | H16b_monthbias_og_v7v11 | H15b | ongoing = v7+v11 mix (193 queue rows) | 0.87310 |
| 57018612 | H17_og_v12v11 | H16b | v12 (hybrid labels, p3) replaces v7 og_v3/noloc (68 queue rows) | **0.87377** (best) |
| 57018690 | H18_aprAdapt | H16b | April-only ongoing adapted on March pseudo windows (71 April rows) | 0.87310 (= parent, as designed) |
| 57018918 | H19_H17_aprAdapt | H17 | the same April adaptation on H17 (58 April rows); pseudo April +0.0100 (paired +0.0113 ± 0.0006) | 0.87377 (= parent, as designed) |

Recommended final pair: **H17 (public line) + H19 (private line)**.

Award admin: registration e-mail by **Oct 25**; code package and PDF by **Nov 10**.

## 2026-10-10

### WEAR @HASCA 2026 (`3rd-wear-dataset-challenge-hasca-2026`), 10 of 10 used

The user said to submit one at a time and learn from each. Base file for all three: `Fm1_margin1_only` (ref 56944621, 0.93597). Details: `wear/RESULTS.md`.

| Ref | File | Change | Public | Δ |
|---|---|---|---|---|
| 57034250 | P1_s22_sitcx_to_null | sbj_22 predicted sit-ups (complex), 128 windows → null (third-session null-label probe) | 0.92537 | −0.01060 |
| 57034443 | LAGp1_c875 | +1 s label lag at link-confirmed set boundaries (131 windows) | 0.93038 | −0.00559 |
| 57034490 | P2_s23_strham_to_null | sbj_23 predicted stretching (hamstrings), 111 windows → null | 0.92836 | −0.00761 |
| 57034785 | M6_meta_top6 | Fm1 + 6 flips ranked by the leaderboard meta-model | **0.93640** | **+0.00043 (new best)** |
| 57034827 | M17_meta_tier2 | M6 + 11 null→activity flips with zero committee support (window log-odds 2.9–4.3) | 0.93596 | −0.00044 vs M6 |
| 57034867 | M11_supported5 | M6 + 4 committee-supported flips (3314 was already in the losing T3) | 0.93610 | −0.00030 vs M6 |
| 57034882 | M4_drop_support0 | M6 minus 6213 and 1751 | 0.93598 | −0.00042 vs M6: those two flips are the whole gain |
| 57034921 | M8_jogarm2 | M6 + 6629 (null→jogging) + 11652 (null→jogging (butt-kicks)) | 0.93640 | 0 (private windows or cancelling) |
| 57035427 | L11_s24_limbbalance | M6 + 11 sbj_24 arm windows jogging → jogging (butt-kicks) (limb-balance anomaly + link neighbours) | **0.93756** | **+0.00116 vs M6 (new best)** |
| 57035458 | L15_s24_limbbalance | L11 + 4 more sbj_24 arm windows (link neighbour share 0.33–0.50) | 0.93678 | −0.00078 vs L11: weaker-link windows really are jogging |

### DataVerse Hindi speech emotion (`dataverse-detecting-emotions-from-hindi-speech`)

The user asked for five submissions today and a plan for five tomorrow. Code, CV protocol and plan: `bhav/README.md`, `bhav/PLAN.md`.
CV = mean over 3 fold seeds of macro-F1 on "novel" OOF rows (no duplicate in the training folds) / full pipeline.
Every file copies the train label onto the 78 test clips that have an exact-length train duplicate.

| Ref | File | What | CV novel / all | Public |
|---|---|---|---|---|
| 57047493 | s1_wavlm_l8 | WavLM-large layer 8 mean+std, logistic regression | 0.785 / 0.852 | 0.82442 |
| 57048523 | s2_wl3_wm_xlsr | Whisper-large-v3 L32 + Whisper-medium L24 + XLS-R L18, log-prob mean | 0.899 / 0.930 | 0.92944 |
| 57049609 | s3_greedy | Whisper-large(v1) L32 + Hindi Whisper-medium L23 ×2 + w2v-BERT 2.0 L11 + Whisper-large-v2 L32 (greedy forward selection) | 0.926 / 0.949 | **0.94230** |
