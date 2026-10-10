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

## Today (2026-10-10), 5 submissions

| Slot | Content | CV novel / all | Public |
|---|---|---|---|
| S1 | WavLM-L L8 LR + override (calibration) | 0.785 / 0.852 | 0.82442 |
| S2 | Whisper-L3 L32 + Whisper-M L24 + XLS-R L18 LR blend + override | 0.899 / 0.930 | 0.92944 |
| S3 | S2 + best of the extra Whisper encoders (large-v2/v1/turbo, Hindi-ASR fine-tunes) | tbd | |
| S4 | S3 + attentive-pooling head on frozen Whisper-L3 frames (and/or self-training) | tbd | |
| S5 | partial fine-tune of Whisper-L3 top-K layers (5x3 folds) blended with S4 | tbd | |

## Tomorrow (2026-10-11), 5 submissions

1. External data: 2,598 clips of the four classes from four public Hindi/Indian emotion datasets (none overlaps the
   competition audio). Train with them added (down-weighted) for the LR probes and the fine-tune; keep only if CV
   novel rises.
2. Fine-tune variants: K (top layers) 8 vs 16, the best Hindi Whisper backbone, 2 more seeds; multi-seed averaging.
3. Self-training round on the full blend (confident test pseudo-labels, CV-checked threshold).
4. Stacking: logistic regression on the OOF probabilities of all members instead of a fixed log-prob mean.
5. Choose the two finals: best CV blend and the most different strong blend (hedge).

Ladder for 2026-10-11 (one change per slot so each public score is interpretable):
T1 best of today + external data in the LR members; T2 fine-tune K=16 / Hindi backbone; T3 multi-seed fine-tune +
stacking; T4 self-training on T3; T5 hedge (best blend without the fine-tune, or with external data off).

## Day 3 (2026-10-12, until 18:30 UTC)

Remaining 5 slots only for confirmed improvements; final selection by 17:00 UTC.

## GPU budget

Kaggle T4, moderate: extraction round 1 = 14 min, round 2 ~40 min, partial fine-tune ~25 min per variant. Everything
else (probes, heads, blends) runs on CPU locally. The account shares 2 concurrent GPU sessions with other work.
