# MuscleFM / SLM Plan — Muscle-Ultrasound Small Foundation Model
Created 2026-11-05 (d34). Owner: campaign agent. Horizon: 2026-11-14 (deadline).

## 0. Why (the honest strategic picture)
The carve space is EXHAUSTED as of d34: every FL-share, PA-weight, PA-offset, and dispersion axis is
bracketed to a vertex or closed. Marginal cards now yield ≤0.0001 (sub-resolution). Rank 6 at 0.19747;
#5 = 0.18833 (−0.0091), #1 = 0.17651 (−0.0209). The only multi-thousandth lever left is a better
segmentation/geometry stack — i.e. the SLM below. If the SLM fails its gates, we keep A = d33_S1_gh60
lineage (0.19747) and B = d32_S4_hedgeB2026 (0.21150) and bank the carve.

## 1. Design (adopted from the 2026-11-05 research drop; every row is grounded)
| Stage | Design | Grounded in |
|---|---|---|
| Init | USFM / UltraSam / USF-MAE encoder if weights reachable from the kernel; else ImageNet ResNet-34. NEVER a generic DINOv3 off the shelf (DINOv3 lost to US-specific models in the 2026 obstetric comparison) | USFM (2.1M US images, 20% annotations suffice); UltraSam (US-43d, 280k images); USF-MAE (OpenUS-46, 370k) |
| Continued SSL | Masked modeling on UMUD images with masking elongated ALONG the fascicle direction; pixel-spacing + device-ID as conditioning | USF-MAE recipe; EchoFM periodic masking (adapted: fascicle direction replaces cardiac periodicity) |
| Encoder | Small ViT-S/16 ("SLM") with multi-scale token aggregation from intermediate blocks (2,5,8,11), plus a CNN branch for thin linear structures | MedDINOv3 token-reuse recipe |
| Heads | (a) aponeurosis mask, (b) dense fascicle-orientation field (sin/cos of 2θ), (c) PA/FL/MT regressors with uncertainty; loss weights follow the metric (PA/6, FL/12, MT/3) | task-specific |
| Geometry layer | MT = aponeurosis-to-aponeurosis distance; PA/FL from fitted lines with FOV extrapolation (classic-tool behavior) | DL_Track/UltraTrack lineage |
| Video (later) | MedSAM2-style memory propagation + fine-tuned point tracker (CoTracker3 pseudo-labeled) + Kalman fusion of per-frame and temporal estimates | MedSAM2 (19,232 annotated US frames); CoTracker3; UltraTimTrack (Kalman beat DL_Track 3.8mm/3.9° on TA) |
| Augmentation | TrivialAugment default; domain-independent augs beat US-specific ones; custom speckle simulators only if TrivialAugment plateaus | Tupper & Gagné (TMLR 2025) |
| Validation | Leave-one-device-out (LODV) by device family (telemed/juniper/lumify/gray853/h512/left_ticks) | our cross-device generalization problem |
| Gate | Must beat an nnU-Net-style ResEnc U-Net baseline (P0) on offline benchmarks BEFORE any FM claim; then paired-bootstrap 2,000× vs the current ensemble, ≥2 CIs excluding 0 | nnU-Net Revisited (2024); campaign gate protocol v3 |

## 2. Phases (kernel runs are async; submissions stay on the 2+2+1 schedule)
- **P0 — gate baseline (2026-11-05, kernel `umud-seg-slm0`):** 2D U-Net (smp, ResNet-34 encoder,
  ImageNet init — deliberately the nnU-Net-style baseline, NOT the FM), TrivialAugment, LODV,
  apo + fascicle masks, Dice+BCE. Deliverable: per-device IoU table = the number every later stage must beat.
- **P1 — US-init fine-tune (11-06..07):** same trainer, encoder swapped to USFM/UltraSam (weights
  downloaded in-kernel, cached to a Kaggle dataset). Gate: LODV IoU beats P0 by >1 point IoU on ≥2 devices.
- **P2 — SLM pretrain (11-08..10):** ViT-S/16 masked-modeling on UMUD images (+sampled US-43d if
  reachable), elongated-along-fascicle masking, 20-30 epochs max (Kaggle 9-12h GPU budget).
  Gate: linear-probe segmentation beats P1 or matches it at lower LR/epoch budget.
- **P3 — geometry heads + submission (11-10..13):** PA/FL/MT regressor heads + geometry layer →
  a full submission CSV via the same level-matching machinery; offline gate vs ensemble; only a
  PASS becomes a submission card. ABORT LINE: if P1 gate fails by 11-08, no P2/P3 — bank the carve.
- Kernel lineage note: this EVOLVES `umud-seg-train` → `apo2`/`fasc3..6` (the existing ensemble
  members). The SLM stack plugs into the same member/gate harness (scripts/gate_members.py logic).

## 3. What the SLM must produce to matter
A member whose per-row PA/FL/MT estimates are decorrelated from the current ensemble (the d14 OSF
measurement showed DLTrack-vs-ensemble FL error correlation 0.10 — decorrelation is where the
thousandths live). Target: ensemble MAE 0.19747 → ≤0.193 offline before any slot is spent; realistic
public target if P3 passes: 0.190–0.195 (a #3-#5 move), with private-split robustness as the priority.

## 4. Standing rules this plan inherits
- No submission without a pre-registered prediction interval.
- Batch-3 manifest = exactly ONE card (d34 lesson: a 2-card batch-3 manifest overwrote the schedule
  and the quota guard rejected the 6th — pas092 slipped to d35).
- Kernel outputs are gated offline BEFORE any submission; the ensemble keeps its A/B until the gate passes.
- All artifacts to download/umud_d32 + git push (automatic in bootstrap v4+).
