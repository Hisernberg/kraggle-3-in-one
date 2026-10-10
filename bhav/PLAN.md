# BhavVaani plan (2026-10-10 to the 2026-10-12 18:30 UTC deadline)

Goal: public/private top 3 (public top today: 0.96544, three teams tied; then 0.95387). Final score =
0.4 public + 0.6 private, two final selections. Public has ~84 rows, so 1 clip is ~0.012 of public F1; pick finals by
CV first, public second.

## What decides the score

1. **Duplicates (37 % of test).** 78 test clips have an exact-length copy in train; their label is copied
   (98.8 % reliable on train-train pairs). Every submission does this; it is not a lever anymore.
2. **The model on the 131 novel test clips.** Only lever left. CV "novel" macro-F1 is the number to move.
3. Test-test duplicate clusters (16 clips) get averaged probabilities so both copies agree.

## Findings that shaped the ladder

| Model (frozen, pooled, LR probe) | best layer | CV novel F1 |
|---|---|---|
| Whisper-large-v3 encoder | 32 (last), mean+std | **0.884** |
| Whisper-medium encoder | 24 (last), mean | 0.862 |
| XLS-R 300M | 18, mean+std | 0.813 |
| w2v-BERT 2.0 | 11 | 0.803 |
| MMS 300M | 15 | 0.797 |
| WavLM-large | 8 | 0.787 |
| audeering w2v2 (MSP-dim) | 7 | 0.790 |
| HuBERT-large | 16 | 0.782 |
| emotion2vec+ large (utterance) | - | 0.541 (zero-shot 0.485) |
| log-mel statistics | - | 0.560 |

Things that did **not** help: per-speaker (ECAPA cluster) normalisation (-0.10, clusters are emotion-correlated),
layer ensembles of one encoder, LDA / PCA-whitened LR, class-bias tuning (does not transfer across fold seeds),
equal-weight blending of the weaker SSL models. Small gain: self-training on confident test pseudo-labels (+0.005).

## Today (2026-10-10), 5 of 5 used

| Slot | Content | CV novel / all | Public |
|---|---|---|---|
| S1 | WavLM-L L8 LR + override (calibration) | 0.785 / 0.852 | 0.82442 |
| S2 | Whisper-L3 L32 + Whisper-M L24 + XLS-R L18 LR blend | 0.899 / 0.930 | 0.92944 |
| S3 | greedy: Whisper-large(v1) + Hindi Whisper-medium ×2 + w2v-BERT 2.0 + Whisper-large-v2 (LR probes) | 0.926 / 0.949 | **0.94230** |
| S4 | S3 + attention head on frozen Whisper-L3 frames | 0.929 / 0.951 | **0.94230** |
| S5 | greedy over probes + heads (head Whisper-large-v2 ×2, ...) | 0.933 / 0.953 | 0.93019 |

Result: 9th of 44 by tie-break (0.94230, tied 7th-9th). Top 3 = 0.96544 (≈2 more public clips right).

Lessons from today:
* Whisper encoders (any size/language variant) beat every wav2vec2-family model by 0.07-0.10 on novel rows; the
  original **Whisper-large (v1) and large-v2** are the best, Hindi-ASR fine-tunes are close, turbo is weak.
* An attentive-pooling head on frozen frames beats the LR probe for large-v2 (0.904 vs 0.885 novel).
* Partial fine-tuning (top 8 of Whisper-L3, 12 epochs) was worse than the frozen probe (0.861) and hurt the blend.
* External Hindi emotion corpora hurt at any weight (domain shift); speaker normalisation hurt; kNN, stacking,
  class-bias tuning and self-training of the blend gave nothing.
* CV saturates near 0.93 novel; S3-S5 differ by 2-9 clips; public (≈84 rows) cannot separate them.

## Tomorrow (2026-10-11), 5 submissions

Priority is variance reduction and a stronger single family, not more greedy selection (greedy on 566 rows is
noisy: two different pools gave 0.926 and 0.925 with different members).

GPU work (one kernel each, moderate):
1. **Bagged heads**: 5 seeds per backbone for the large-v2, large-v1, Vaani-L3 and Hindi-medium heads (head
   training is ~2 min per backbone on a T4 once frames are computed).
2. **Multi-layer heads**: learnable softmax weights over the last 4 encoder layers of large-v2 / large-v1 before
   pooling (late layers 29-32 all score > 0.86).
3. **TTA frames**: re-extract with 0.1 s and 0.2 s leading silence and average head/probe outputs (Whisper is
   position-sensitive; cheap robustness).
4. Partial fine-tune retry only on large-v2 with K=4, lr 1e-5, 8 epochs; keep it only if novel > 0.90.

Ladder (one change per slot):
* T1 = S4 members + bagged large-v2 head (fixed weights, no greedy): robust successor of the public best.
* T2 = T1 + multi-layer heads.
* T3 = T2 + TTA.
* T4 = equal-weight average of the best LR-probe blend and the best head blend (two families, 50/50).
* T5 = hedge: whichever of T1-T4 has the best CV with members picked by rule (all Whisper members with novel > 0.87),
  not by greedy.

Final selection (2 slots): the best-CV rule-based blend and the best public of S3/S4/T*; never two near-identical
files.

## Day 3 (2026-10-12, until 18:30 UTC)

Remaining 5 slots only for confirmed improvements; final selection by 17:00 UTC.

## GPU budget

Kaggle T4, moderate: extraction round 1 = 14 min, round 2 ~40 min, partial fine-tune ~25 min per variant. Everything
else (probes, heads, blends) runs on CPU locally. The account shares 2 concurrent GPU sessions with other work.
