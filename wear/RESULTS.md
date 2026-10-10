# WEAR: 2026-10-10 submission results

Base file: `Fm1_margin1_only` (ref 56944621, public 0.93597, #8).

| Slot | File | Hypothesis | Out-of-fold check | Public | Δ | What we learned |
|---|---|---|---|---|---|---|
| 1 | P1_s22_sitcx_to_null | One-activity third sessions are labelled null, as in train sbj_2. sbj_22's is sit-ups (complex) | (none; label convention) | 0.92537 | −0.0106 | All 128 sit-ups (complex) windows of sbj_22 are labelled. Either the convention does not hold or the class is wrong. |
| 2 | LAGp1_c875 | Test labels lag the sensors by 1 s (sbj_7 precedent); trim set starts, extend set ends at link-confirmed boundaries (16 fork matchings, agreement ≥ 0.875) | −0.004 if there is no lag (train has none) | 0.93038 | −0.0056 | No +1 s lag. The best file's boundaries are mostly right; −1 s would lose the same way. |
| 3 | P2_s23_strham_to_null | Same as slot 1, for sbj_23 | — | 0.92836 | −0.0076 | Also labelled. **The third-session null idea is closed.** |

Out-of-fold checks on the fork probabilities (`scripts/band.py`, `count_shift.py`, `lagshift.py`) that did not justify a slot:
- Count band 78–126 / 70–135: −0.0004 to +0.0001. Complex-variant share band: +0.0002–0.0003 on about 20 windows (noise).
- Global trim or extend of every bout by k windows: every k loses (k = −1: −0.0003). The fork's counts are unbiased (median count error 0, mean absolute error 7.9).
- Video-only session recovery (k-means, mutual kNN, posture-removed residuals): does not separate sessions. Posture dominates.

## What is left that can move the score

1. **Re-decode the f4n/mv4/mv6 family.** In goodpjw's ablations, true counts are worth +0.011 out-of-fold and true boundary order +0.016. This needs the raw probabilities and link sets, which exist only on the user's PC (`E:\Claude code\wear`); none of the team's Kaggle kernels has them.
2. **2nd-challenge 4-limb test data.** Twins of about 85% of this year's test windows are in it. It was used only for link re-scoring so far. Downloading it needs this Kaggle account (kragglenote2forwork) to accept the rules of `second-wear-dataset-challenge` in the web UI (the API returns 403).
3. Everything else measured here is at noise level (≤ 0.0003 out-of-fold).

## Final selection (unchanged)
`Fm1_plus2` (ref 56944691) or `f4n` (ref 56884414). None of today's files.
