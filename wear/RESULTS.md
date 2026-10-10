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

## Slots 4–8: leaderboard meta-model (user asked for analysis-driven submissions)

Method (`scripts/lbreg.py`, `meta.py`, `meta2.py`):
- Every scored file within 300 windows of Fm1 is decomposed into its (window, label) changes.
- **Per-window regression:** 1,295 changes collapse into 492 co-moving groups against 60 files. Leave-one-file-out shows it learns nothing beyond a flat cost of about −4.5e-5 per changed window. Single windows cannot be identified, which is why window-by-window probing stalled.
- **Feature meta-model:** a change's value is modelled from a few features: constant, window-model log-odds (`dlogWin`), the share of strong files (≥ 0.933) that carry the new label (`support`), and null→activity. Leave-one-out RMSE 0.00099 against 0.00171 for the constant alone.

| Slot | File | Public | Read |
|---|---|---|---|
| 4 | M6: Fm1 + 6 top-ranked flips (bootstrap P(positive) ≥ 0.84) | **0.93640** | +0.00043, about 4× the prediction |
| 5 | M17: M6 + 11 zero-support null→activity flips (log-odds 2.9–4.3; mostly lumbar rotation and sit-ups) | 0.93596 | −0.00044: weak-evidence floor exercises are wrong |
| 6 | M11: M6 + 4 supported flips | 0.93610 | −0.00030: the supported pool gives nothing more |
| 7 | M4: M6 without 6213 and 1751 | 0.93598 | Those two flips (sbj_23 left arm, null→jogging (skipping), log-odds 6.9 and 5.3) are worth +0.00042; the other four are about 0 |
| 8 | M8: M6 + 2 more jogging-family arm flips (6629, 11652) | 0.93640 | 0 |

Lessons:
- Gains come only from **very strong window evidence for dynamic classes** (jogging family) on null windows. After M6, no window with log-odds above 5 remains.
- Committee-supported flips from earlier files are used up.
- These are public-window corrections; expect little transfer to private.

## What is left that can move the score

1. **Re-decode the f4n/mv4/mv6 family.** In goodpjw's ablations, true counts are worth +0.011 out-of-fold and true boundary order +0.016. This needs the raw probabilities and link sets, which exist only on the user's PC (`E:\Claude code\wear`); none of the team's Kaggle kernels has them.
2. **2nd-challenge 4-limb test data.** Twins of about 85% of this year's test windows are in it. It was used only for link re-scoring so far. Downloading it needs this Kaggle account (kragglenote2forwork) to accept the rules of `second-wear-dataset-challenge` in the web UI (the API returns 403).
3. Everything else measured here is at noise level (≤ 0.0003 out-of-fold).

## Final selection
Public best is now M6 (ref 57034785, 0.93640). It differs from Fm1 by 6 windows, 2 of them verified, so it is safe to select. Alternatives: `Fm1_plus2` (ref 56944691) or `f4n` (ref 56884414).
