# WEAR: plan for 2026-10-11 (10 slots, run automatically from 00:05 UTC)

The user asked for an automatic run. Start: **#6, 0.93756** (L11, ref 57035427).
Targets: #5 0.93766 (+0.00010), #4 0.93825 (+0.00069), #3 0.93849 (+0.00093).

## Honest outlook

Today's evidence leaves a small pool of candidates that have real support:

- The leaderboard meta-model, refit on all 67 nearby files, has **no** positive candidates left (best predicted +4e-7).
- Generic limb-balance repair is a coin flip out-of-fold (55% of moves right, ΔF1 ≈ 0). Slot 9's +0.00116 was one strong bout (sbj_24 butt-kicks arms look like jogging). The only other signature bout found is the same one, and its extension lost (L15 −0.00078).
- "All sources agree against the decode" is not trustworthy for arm-confusable classes: L11's winners were exactly such windows, and moving them against the sources won.

**Reaching top 3 needs about +0.0009. The prepared candidates are worth roughly ±0.0003 each, so top 3 is unlikely (my estimate: under 15%) unless one of the unlocks below arrives.** #5 (+0.0001) is within reach.

## Ladder: probe and keep

Each probe goes on top of the current best. Keep it if the public score rises by at least +0.00005, otherwise discard it. A composition uses only winners.

| Slot | File (from `scripts/build_oct11.py`) | What | Expected |
|---|---|---|---|
| 1 | O11_V3 | variant-pair share band: 3 sbj_23 push-ups (complex) → push-ups, chosen by fork decode (out-of-fold +0.0002–0.0003) | ±0.0003 |
| 2 | O11_A2 | 2 activity→null windows where both fork decodes, window, fusion and IMU-only models agree (sbj_25 triceps stretch 2197, sbj_23 lunges 9783) | ±0.0003 |
| 3 | winners of 1–2 combined (skip if fewer than 2 winners) | | sum of winners |
| 4–10 | **only if an unlock arrived** (below); otherwise keep the slots | | |

Stop rule: do not submit a file without fresh evidence. An unused slot costs nothing; a mean-zero probe only adds public-LB noise.

## Unlocks (from the user) that would change this

1. **The f4n/mv4/mv6 probabilities and link sets from the PC** (private Kaggle dataset or private repo). Re-decode the best family with count and limb-balance constraints, gated out-of-fold. This is the only lever with out-of-fold evidence of +0.01-scale effects.
2. **Accept the rules of `second-wear-dataset-challenge`** as kragglenote2forwork. Its 4-limb test data holds twins of about 85% of this year's windows: true limb-joint evidence for every second, and real timeline order.
3. A new public kernel at 0.936 or above: read once, mine its labels only through the same probe-and-keep gate.

## Final selection (before 2026-10-12 21:00 UTC)

Select the best **bout-level** file: L11 (0.93756), or a later winner that only adds group-level changes. Do not select single-window public flips over it.

## Restart after a container reset

```bash
bash wear/scripts/bootstrap.sh     # data, kernel outputs, all own submissions, scripts → /home/user/wear_work
cd /home/user/wear_work && python3 build_oct11.py own_subs/sub_57035427.csv
```
