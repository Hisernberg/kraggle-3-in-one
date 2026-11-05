# MuscleFM-SLM vs the field — comparison document (2026-11-05)
Live version: `umud/kaggle/` kernels; this file consolidates results as they land.

## 1. Our same-benchmark comparison (identical protocol — the honest table)
Protocol: 2-class seg (aponeurosis, fascicle), Dice+BCE, TrivialAugment (one-op random magnitude),
10/8 epochs, bs8, 384px, leave-one-device-out (LODV) validation, GT mask pairs from the
competition train layer. Runners: Kaggle GPU kernels `umud-seg-slm0` (v2) + `umud-seg-slm1`.

| Arm | Encoder / init | Params (approx) | LODV fascicle IoU | LODV aponeurosis IoU | Status |
|---|---|---|---|---|---|
| A (P0 control) | U-Net ResNet-34, ImageNet | ~24M + head | _pending kernel run_ | _pending_ | slm0 v2 + slm1 RUNNING |
| B (transformer) | U-Net mit_b1 (SegFormer), ImageNet | ~14M | _pending_ | _pending_ | slm0 v2 + slm1 RUNNING |
| C (US-foundation) | ViT + FPN head, USFM/UltraSam init if loadable from HF (runtime probe); fallback timm vit_small augreg_in21k | ~22-86M | _pending_ | _pending_ | slm1 RUNNING |

Fill-in rule when run_record.json lands: paste `arms.<arm>.lodv.<device>` per device + mean;
a winner must beat arm A on BOTH classes' mean IoU AND on ≥2/3 of devices (gate protocol v3).

## 2. Published-numbers context (DIFFERENT benchmarks — orientation only, not same-data SOTA claims)
| Model | Year | Pretraining corpus | Reported strengths (their benchmark) | Relevance to us |
|---|---|---|---|---|
| USFM | 2023-24 | ViT, spatial-frequency dual masked image modeling, >2.1M US images | robust with 20% of annotations | the natural arm-C init; probe + fine-tune |
| UltraSam | 2024 | SAM adaptation on US-43d: 43 datasets, >280k images+masks, >50 structures | UltraSam-init ViT beat ImageNet/SAM/MedSAM inits on downstream US tasks | alternative arm-C init |
| USF-MAE | 2025-10 | MAE on OpenUS-46: 370k 2D/3D US images, 46 datasets | evaluated on 3 classification benchmarks only (no dense geometry) | pretraining recipe for P2 (elongated-mask adaptation) |
| EchoFM | 2024-10 | MAE + periodic contrastive, >290k echo videos | video-side SSL template | P3 video stage blueprint (periodicity -> fascicle direction) |
| MedSAM2 | 2025-04 | SAM2 fine-tune, 455k 3D pairs + 76k frames (incl. 19,232 US frames) | >85% annotation-cost cut in user study | pseudo-labelling route for GT-mask scarcity |
| MedDINOv3 | 2025-09 | DINOv3 adaptation, 3.87M CT slices (not US) | multi-scale token reuse (blocks 2,5,8,11) | architecture recipe for thin linear structures |
| nnU-Net ResEnc | 2024 | supervised baseline framework | "beats fancy models when properly configured" | our mandatory gate baseline (arm A) |
| DL_Track_US | 2024 | U-Net + VGG16 U-Net (host's flagship) | vs manual: FL −2.4mm, PA 1.5°, MT <1mm | existing ensemble member; published error floor |
| UltraTimTrack | 2025-01 | Kalman fusion of UltraTrack + TimTrack | beat DL_Track (3.8mm/3.9° on TA @0.2s/frame) | video arm for P3 |
| DUSTrack | 2025-07 | DL + optical-flow hybrid | beats zero-shot point trackers; fascicle use case | drift-free tracking arm |

**Why our numbers will differ from all published ones:** every model above reports on its own
benchmarks (segmentation IoU/Dice on public US sets, or classification). Our gate is LODV IoU on
competition GT masks + downstream PA/FL/MT MAE under the challenge metric mean(MAE_PA/6, MAE_FL/12,
MAE_MT/3). Cross-benchmark SOTA claims are NOT made here; the same-benchmark table in §1 is the
only comparison that counts for adopting an encoder.

## 3. Decision rules (unchanged from FM_PLAN.md)
- Arm C (or B) is adopted only if it beats arm A on both classes AND ≥2 devices' LODV.
- Any adopted encoder feeds P3 (PA/FL/MT heads + geometry layer); a submission card requires the
  offline MAE gate vs the current ensemble (paired bootstrap 2,000×, ≥2 CIs excluding 0).
- Abort line: if by 2026-11-08 no encoder clears arm A, P2/P3 stop and the campaign banks the carve.
