# Structured re-decode (ICM energy) -- result: NEGATIVE, do not use

Energy per subject: E = -sum_i U_i(l_i) + lam_link*#(16 fork+fork2 successor links crossing labels)
 + lam_limb*sum_c (z_arm^2+z_left^2)/2 (z vs subject-level limb rate; optional hinge |z|>z0); U = log Q (fork / fork2 / mean).
Optimised by ICM (6 sweeps, random order, candidates within 6 nats of best). Code: rd.py, load.py, grid*.py, check15.py, anchor.py.

## OOF (subject-out OOF of the fork pipelines; base macro-F1 fork 0.92304, fork2 0.91830)
- limb term alone hurts monotonically (fork: lam_limb 0.1/0.3/1/3 -> +0.00002/-0.00005/-0.00079/-0.00196, move precision 0.37-0.47).
  Reason: on OOF the fork argmax bouts are NOT more limb-imbalanced than truth (sum z^2/2: truth 442.6, fork 434.5, fork2 434.9;
  expected ~396). Only ~2 of 396 predicted bouts (sbj15 c12, sbj10 c15) show the "error-induced imbalance" pattern. Note OOF
  sbj10 has left_arm dropout (18% vs 25%), so truth itself is imbalanced there.
- link term (unweighted) is the only gain: lam_link 0.02-0.03 -> fork +0.00141 (279-344 moved, prec 0.57-0.59),
  fork2 +0.00132/+0.00136, mean-Q +0.00140; positive in all 5 folds on both. Below the +0.002 gate.
- adding limb (z0=2-2.5, lam 0.3-3) on top: changes within +-0.0001 (noise). Score-weighted links and mixing window logp: worse/higher variance.

## Sanity check on test (U = mean log Q, lam_link 0.03)
- Energy wants to REVERT all 11 verified-correct sbj24 jog-butt moves (dE -0.7 to -2.0 per window, i.e. unary ~-1.1..-2.0 dominates
  link +0..+0.96). With lam_limb 3 only 2/11 (5264, 2158) are kept. It correctly keeps 8717, 9823, 9927, 9656 as jog (dE +1.2..+2.0).
- Anchored ICM from L11: margin 0/0.5/1/2/3 -> 273/142/71/15/5 changes; >90% are reversions to the fork consensus argmax
  (at margin<=1 they include the 11 verified moves). Margin-2 list in anchor_m2_changes.csv (15 windows, 14 = fork-argmax reversions,
  mostly null->activity at bout edges); file cand_anchor_m2_NOT_RECOMMENDED.csv. L11 beats the fork by +0.011 public, so these are
  very likely harmful.

Confidence: high that the generic joint energy does not help L11; limb prior carries no OOF signal; link smoothing ~+0.0014 on fork-level labels only.
