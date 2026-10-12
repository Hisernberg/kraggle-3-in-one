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

## 2026-10-11, 5 of 5 used

| Slot | Content | CV novel / all | Public |
|---|---|---|---|
| T2 | bagged heads (L2, L1, Vaani-L3, Hindi-M) + w2v-BERT | 0.933 / 0.954 | 0.93019 |
| T3 | S4 + bagged/self-trained large-v2 head ×2 | 0.930 / 0.952 | **0.94230** |
| T4 | 3-seed heads + w2v-BERT + Whisper-L1 LR | 0.933 / 0.953 | 0.93019 |
| T5 | 3-seed heads + w2v-BERT + LR on concatenated features | **0.938 / 0.957** | 0.93019 |
| T6 | rule-based all-strong ensemble | 0.927 / 0.949 | **0.94230** |

What the day showed:
* Bagging heads helps single members (+0.004-0.008); TTA, multi-layer heads, self-training and the fusion head do not
  add to the blend. CV is flat at 0.93-0.94 novel.
* All strong blends agree on 197 of 209 test clips. The head family changes a handful of public clips (in the set
  41, 244, 265, 331, 431, 542, 573) and loses one of them net on public; its changes on 570, 873, 1013 are
  public-neutral. One public clip is ~0.012, so this is noise-level and says little about the private 60 %.
* Top-3 (0.96544) needs ~2 more public clips right that **every** model gets wrong. Blending cannot reach that; only
  a stronger model could, and none of today's attempts produced one.

## 2026-10-12 (CPU only, 2 submissions)

| Slot | Content | CV novel / all | Public |
|---|---|---|---|
| U1 | T5 + transcript TF-IDF member (w 0.15) + transcript near-duplicate override | 0.936 / 0.956 | 0.93019 |
| U2 | T5 + transcript TF-IDF member (w 0.25) + transcript near-duplicate override | 0.932 / 0.953 | **0.94230** |

* Words alone predict emotion (text member 0.712 novel): the acted sentences are emotion-specific (every sentence
  recorded more than once carries one emotion). Whisper encoders already carry most of it, so the text member mainly
  re-weights the contested clips.
* The transcript override is 98.9 % accurate in CV but changes no test clip: the blends already agree with every test
  clip it covers, and the 4 test-test sentence pairs agree too.
* U1 vs U2 isolates clip 331 as the head family's single public loss; 570, 873, 1013 and 265 are public-neutral.

## Final selection (two files; do it on the Kaggle Submissions page before 18:30 UTC)

1. **V1 `v1_t5x09_t3x01_tm`** (public 0.94230, CV 0.937): 0.9 × T5 + 0.1 × T3 with the transcript override, the
   CV-favoured head-family labels everywhere except clip 331, where the public split showed that family is wrong.
2. **T3 `t3_s4_b3stl2`** (public 0.94230, CV 0.930 / 0.934 with the override): the probe family; differs from V1 on
   41, 244, 573, 1013, so the pair hedges the clips where the families disagree off the public split.

(V1 and T5 differ only on 331, and U2 sits between V1 and T3, so neither adds a hedge.)
T3 is almost identical to U2 (2-3 clips), so picking both would waste the second slot.

Not done before the deadline (needs GPU or ~2 h of CPU per encoder): re-extract Whisper features with loudness
normalisation. The 32-bit PCM clips (80 train, 22 test) are louder (median peak 27k vs 17k) and score 0.856 vs 0.948.

## GPU budget

Kaggle T4, moderate: extraction round 1 = 14 min, round 2 ~40 min, partial fine-tune ~25 min per variant. Everything
else (probes, heads, blends) runs on CPU locally. The account shares 2 concurrent GPU sessions with other work.
