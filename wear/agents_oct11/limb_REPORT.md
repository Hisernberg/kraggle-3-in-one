# limb agent: learned limb-specific confusion-signature repair

## Rule / model
Candidate = (window w labelled d>=1, target class c) where the depth-2 link-neighbour share of c is >= 0.15 (16 matchings, fork+fork2).
Features (lib.py): neighbour shares of c and d at depth 1/2/3 (+ the original multiset share), arm/leg and left/right balance
z-scores and binomial log-LR gain for both bouts, subject-specific limb confusion rate (fraction of bout c windows of w's limb
type whose window-model argmax is d, plus related rates), window-model logp diff (fork, fork2), decoded Q of c and d (fork, fork2),
and count deviation of both bouts. Label = oof_y[w]==c. HistGradientBoosting, subject-out (LOSO static, 6-fold subject groups for greedy).
Greedy: one move per step, features recomputed, stop when the best score < threshold, max 40 moves per subject.

## OOF (28.5k candidates, 11.7% correct; AUC 0.891 sibling-only)
Greedy, sibling-only, all features:
| th | fork n / prec / dF1 | fork2 n / prec / dF1 |
|---|---|---|
| 0.5 | 252 / 0.575 / +0.00132 | 310 / 0.629 / +0.00232 |
| 0.6 | 160 / 0.600 / +0.00100 | 184 / 0.674 / +0.00167 |
| 0.7 | 89 / 0.674 / +0.00085 | 121 / 0.702 / +0.00126 |
| 0.8 | 37 / 0.649 / +0.00027 | 48 / 0.708 / +0.00048 |
Without the limb features: th0.7 fork 0.604/+0.00066, fork2 0.688/+0.00133 -> the limb features add almost nothing.
Permutation importance: q2c, q1d, q1c, wl1 dominate. The wins are mostly 11<->12 (push-ups vs push-ups-cx) boundary fixes.
Without Q/window-logp features (limb+links only): fork th0.7 0.422/-0.00006; only th0.8 passes (14 and 18 moves, prec 0.64/0.61, dF1 +0.00009).

The sbj_24 signature is outside the OOF distribution: no OOF candidate has zc_arm<-2 and conf_ct>0.44 (sbj_24: zc_arm=-2.87, conf_ct=0.96).
The closest OOF analogues (sbj 9 jog-side arm windows predicted jog-skip, conf>0.6, zc about -1.5) have precision 0.36-0.40.

## Test verification (sbj_24, state = L11 with the 11 verified windows set back to jog)
Full model: FAIL. Only 1844 scores >0.5. Correct-vs-wrong AUC is 0.59. Run greedy on L11 and it wants to move all 11 verified-correct windows back 5->1 (scores 0.48-0.72).
Q-free model: ranks well (AUC 0.95; all 4 wrong ones <0.02). But only 3138/7186/1844 score >0.4, and its OOF fails.

## Test proposals (full model, th=0.7, greedy on L11): 14 moves, 11 of them just revert L11 to the fork Q argmax
22: 5777 12->11, 3855 5->17; 23: 5201 12->11, 2639 12->11, 10740 12->11, 9367 15->11, 6393 2->1, 4323 12->11;
24: 5309 11->15 (0.956); 25: 8894 15->12, 6114 11->12, 9288 7->16, 6944 7->16, 4675 12->11.
File: L11_limbclf_th07_NOT_RECOMMENDED.csv (moves in proposed_moves_th07.csv).

## Confidence
Low. Do not submit. This rule passes the OOF gate, but on test it is a "trust Q" model. It contradicts the one verified win and mostly undoes
L11's deliberate deviations from the fork argmax. Those deviations are absent on OOF, where the base is the fork argmax.
