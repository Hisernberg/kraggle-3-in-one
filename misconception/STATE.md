# Misconception K-12 - account B state (2026-10-28 12:45 UTC, B2 run)

Best public: V7 0.64900 (V4 0.64766 + pseudo-label student as 4th VLM member). Rank 88/282. Top 0.83300 / 0.83183 / 0.82900.
Deadline 2026-11-17. Final-selection candidates so far: V7 (0.64900), V4 (0.64766) - both OOF-validated.

## Where the V-series artefacts live
V2-V7 (own Qwen3-VL-8B LoRA + linear head, 3 fold models, pseudo-label student, D2b base = logp v11c:2 + v12a:1)
were trained on the DGX session. Their OOF/test probability arrays are NOT on GitHub, HF or a Kaggle dataset,
so a cloud run (2 CPU / 7 GB) cannot build on V7. Kernels k12-pack-images / k12-pack-unlabelled (10-27) only pack inputs.

## Needed from the DGX session (one-time)
Upload to a private Kaggle dataset `kragglenote2forwork/k12-vseries-probs` (or HF dataset):
test/OOF prob arrays for v11c, v12a, VLM f0/f1/f2, pseudo-label student; row ids; misconception code list;
OCR answer caches; the V7 blend script. Then cloud runs can do blend/promotion changes on CPU.

## Gap analysis / next idea (0.65 -> 0.83)
- Our VLM alone is 0.55-0.58 per fold trained on 1000 rows; the ensemble adds only ~+0.04 over the text stack.
  The top cluster (0.82-0.83, 10+ teams) implies one much stronger single model, not an ensemble trick.
- Most likely lever: fine-tune a stronger VLM end-to-end on all 1500 rows + filtered pseudo-labels
  (Qwen3-VL-32B / 30B-A3B LoRA, or 8B with full image resolution), as a listwise classifier over the
  question's candidate misconceptions (retrieval: restrict to codes seen for that question + NONE), with
  the misconception description text in the prompt. Evaluate on the same 3-fold split; gate: fold held-out >= 0.65 alone.
- No public notebook newer than 10-11 (best public 0.597), so no public shortcut exists.
