# counts agent: bout-count anomalies and intruders

## Train truth
Bout length divided by the recording's median bout: 5th/50th/95th percentile 0.84/1.00/1.30. 8.1% of true bouts are >1.2x and 4.9% are >1.3x.
So the test bouts at 1.2-1.38x (sbj22 sit-ups-cx 128, lumbar 114, hamstrings 113; sbj23 hamstrings 111; sbj25 str-lunging 120) are within normal range by length alone.

## Rules tested (OOF: fork / fork2; precision of moves; dF1 x1e-4)
- **Inflated bout -> null** (alt = 2nd decoded class = null, top dQ, k = 25-50% of the excess): precision 0.15-0.33, dF1 -0.6 to -38 in every setting. **Rejected.**
  In test, every inflated sbj22/23/25 bout has null as the alternative for nearly all of its windows, and own neighbour share is >= 0.5 for all of them. They look like genuinely long bouts.
- **Short bout fill** (from null or any class): precision 0.25-0.73 depending on settings. From null alone it is always < 0.6. Not robust.
- **LightGBM intruder scorer** (40 features: decoded/window/IMU/fusion/QA/B2/TA log-ratios, neighbour shares, bout ratio of own and alt bout, limb z-scores; LOSO over 22 subjects; target = wrong and alt is right): AUC 0.80/0.81 on active windows.
  Global threshold 0.7: 196/184 moves, prec 0.643/0.598, dF1 +12.9/+9.6. Threshold 0.8: prec 0.75/0.73, dF1 +5.9/+5.4.
  On test it fires on 1 window (thr 0.8) or 12 (thr 0.7). Most of the 12 are boundary windows moved to null, and several have dQ>0 (L11 already disagrees with the fork argmax there, which never happens in OOF). **Not proposed.**
  Using the scorer to rank windows inside the pair rule below gave no gain over dQ.
- **Sibling-pair count balance (PROPOSED)**. For bout c with n_c > 1.15 x median, let a = the most common 2nd-decoded class among c's windows (a > 0). Require 0.5 x med <= n_a < 0.95 x med (excludes whole-bout swaps) and >= 10 such windows.
  Move the k = floor(0.25 x (n_c - n_a)) windows with highest dQ = logQ[a] - logQ[c], keeping only windows with dQ > -1.
  - fork: 116 moves, prec 0.664, dF1 +9.5. fork2: 128 moves, prec 0.703, dF1 +13.9. Averaged-fork OOF: 121 moves, prec 0.793, dF1 +17.4.
  - Excluding the anomalous subjects 0 and 2: prec 0.81/0.77.
  - Without the dQ>-1 gate it fails (prec 0.46/0.52, dF1 -2.0/+3.7). The gate was chosen after seeing per-pair results, so it is mildly tuned on OOF.
  - Stable for rhi 1.1-1.2 and rlo 0.95-1.05 (prec 0.64-0.75).
  - With k = 50% of the gap: prec 0.646/0.714, dF1 +14.5/+20.3. The marginal 2nd quarter has precision about 0.61-0.74.
  - Push-ups/push-ups-cx pairs on OOF: 4 recordings, 37/48 and 38/54 correct.

## Test changes (L11 -> candidate)
Only one pair fires: **sbj23 push-ups-cx (12, n=128) -> push-ups (11, n=88)**. sbj22 str-triceps -> str-shoulders also triggers, but all its candidates have dQ < -2.3, so the gate removes them.
- `cand_pair_q25.csv` (10 windows, 12->11): 1433, 2639, 4323, 5133, 5201, 10374, 10444, 10740, 11537, 12079 (6 arm / 4 leg). The window model favours push-ups on 8 of the 10 (win_d > 0).
- `cand_pair_q50.csv` (20 windows): the above plus 220, 727, 900, 1304, 1591, 2959, 4312, 5566, 7643, 10620.

## Confidence
Moderate for q25: expected about 6-8 of 10 correct.
- The 11/12 pair is exactly the confusable-pair type that works in OOF.
- In test the 11-bout ratio (0.91) is borderline for rlo, and sbj23's 12-bout limb split (54 arm / 74 leg) suggests the excess may be leg windows, while the rule picks 6 arm.
- fork and fork2 OOF are highly correlated, so "both pass" is weaker evidence than it looks.
