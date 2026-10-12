# WEAR: what moved the score, across all ~190 submissions (to 2026-10-11)

## What helped (in the current best file)
| Change | Public effect | Why it worked |
|---|---|---|
| goodpjw learned links + counts → 4-run chain-link fusion + log-count prior (`f4n`) | 0.927 → **0.93541** | real model gain: better links and better per-bout counts |
| Margin-1 committee flips (Fm1, 8 windows) | +0.00056 | only flips where the committee was split by one vote |
| 2 null → jogging (skipping) windows (sbj_23, arm) with very strong evidence from every window model and the fork decode | +0.00042 | dynamic class, all sources agree, fork Q around 0.7 |
| **L11:** 11 sbj_24 arm windows jogging → jogging (butt-kicks) | **+0.00116** | bout-level: the limb split was impossible (28 arm / 54 leg) and this subject's butt-kicks arms look like jogging |
| P2: 2 sbj_22 right-limb windows push-ups (complex) → push-ups | +0.00028 | the right-limb shortfall matched the link evidence |

## What hurt (avoid)
| Mistake | Public effect |
|---|---|
| Smoothing or removing 1-second blips | −0.0169 |
| Probability fusion without re-decoding; decode-less argmax | −0.0106; window-only 0.71 |
| Assuming one-activity third sessions are labelled null (train sbj_2 precedent) | −0.0106 and −0.0076 |
| Global +1 s label-lag shift at bout edges | −0.0056 |
| Cross-family consensus flips (119 windows) | −0.0031 |
| Many-knob out-of-fold-tuned stacks (G stack: out-of-fold 0.9357) | −0.0029 |
| Changes past the boundary of a winning repair (L15, 4 weaker-link windows) | −0.00078 |
| The reverse direction of a winning pair (P4: push-ups → push-ups (complex) in sbj_22) | −0.00056 |
| Weak-evidence null → activity on floor or static exercises (M17) | −0.00044 |
| Limb-imbalance repairs without a style signature (J4); count-pair rule in sbj_23 | −0.00029 each |
| Activity → null even when every source agrees (A2) | −0.00015 |

## Rules that hold up
1. Only bout-level structure or very strong multi-source evidence on *dynamic* classes pays; single-window flips are mostly public noise.
2. All window models can be confidently wrong in the same way for one subject's style (sbj_24 butt-kicks arms). Use limb balance and links, not model agreement, to catch it.
3. Never extend a winning repair beyond its evidence boundary (link-neighbour share about 0.36).
4. Every candidate must pass an out-of-fold gate. Generic rules (video coherence, re-decode, learned limb mover) fail it.
5. The remaining big lever is external: the 2nd-challenge test file (same seconds, 4 limbs). See GAP_ANALYSIS.md.

## 2026-10-12: public boundary-refiner notebooks

- The new public forks (goodpjw Part 2 + chain boundary refiner, OOF 0.927–0.930) lose to our fusion on null ↔ activity
  disagreements: 17 null → activity flips cost −0.00046 even with 8–10 of 10 runs agreeing.
- Activity → activity flips from their committee were the only non-negative group (+0.00003 on 6 sbj_25 windows);
  adding the weaker (6–7/10) and sbj_23 push-ups → complex ones lost −0.00055.
- Committee strength measured within one family does not transfer to a stronger family: check public pairwise
  evidence (Leaf It 0.9304 vs our 0.9378 on 269 differing windows) before trusting an external committee.
- External committee flips only pay at full agreement: 10/10 runs +0.00020 (10 windows); 8–9/10 and 6–7/10 lose.
