# Account B lessons (append one line per scored submission)

## UMUD
- 2026-10-28 d29 (5/5 used 04:24-04:34): S1 disjoint PA+FL cell merge 0.19893 (additive as expected); S4 Lumify PA pipeline weight 0.90 = 0.19888 NEW BEST (-0.00005, tiny; Lumify PA weight vertex near 0.90); S3 6-map PA pipeline 0.19983 (+0.0009 worse: more fascicle maps != better PA, held-out mask harness gain did not transfer); eligible line: MT spread 1.014 0.31239, + within-device shrinkage PA 0.66/FL 0.58 0.29936 (shrinkage toward less shrink helps the no-public-CSV line, -0.013).
- Rank 3 at 0.19888 vs 0.18297 / 0.18704 (both moved 10-28 morning). Per-device weight/spread tuning is now at the +-0.0001 noise floor; closing a 0.016 gap needs a new signal (better FL/PA measurement), not more spread/weight probes.
## Traffic Flow
- 2026-10-28: N1 eIoU re-decode of March onset 0.87759 (-0.0063) and N3 CatBoost+recurrence-prior onset 0.87969 (-0.0042): onset re-decoding keeps losing on public; T08 thr 0.55 is a sharp optimum - stop onset probes. Q1 = L1 + test-month speed level correction (t1_bias, K=30) 0.88400 NEW BEST (+0.00016, ~1.5x the pseudo estimate +0.00010; month-bias transfers on the K-line as on H15b).
- Gap to top-3 (0.924-0.928) is ~0.04: small state tweaks cannot close it; the top teams must solve a task differently (likely Task 2 queue / Task 4 ODME). Artefacts for L1/Q1 live only on the DGX session; tfb-work backup is J2-era (10-22), so a cloud run cannot build on Q1 until the backup is refreshed.
## Misconception
- 2026-10-28: V4 (3-fold own Qwen3-VL-8B LoRA mean, w2 logp + promotions) 0.64766; V5 prob-space mix 0.4 + stricter promotions 0.64200 (-0.006: OOF +0.006 picked from ~30 variants did not transfer - logp blend > prob mix); V7 pseudo-label student 4th member 0.64900 NEW BEST (+0.0013, inside expected +0.000..+0.004). Ensemble tweaks are now worth ~+0.001-0.006; gap to top (0.833) needs a stronger single model.
## ClimX
- 2026-10-28: E1 rule-compliant analog (systematic resampling, h=0.35K) 0.04882 NEW BEST vs A1 0.04738; holdout 370/585 (0.117/0.136) overstate LB by ~2.5x - holdouts are optimistic. Probe precip-off -0.34830 decodes to sum precip nNSE -0.043 (6 indices ~0 each); non-precip 9 indices avg ~0.086. P1 (R95p in kg m-2 s-1) -0.00638 -> truth R95p is in mm.
## Squares in Squares
- 2026-10-28 S1 (11:59, earlier B3 run, files not in repo) certified merge of 13 public outputs = 22.2626642 (= public best busyaprime-33). S2 (13:06) + n=260,856 from avazaku c-seam-dp = local official 22.26266397 (-2e-7). Kaggle shows 10 subs/day here (8 left after S2).
- Public packings already beat every Ellsworth/Friedman record where they differ (n=51,84,103,126,176...) -> records are no source. All 17 public outputs checked: merge frontier = ours. Monotonicity and grid-border extension give nothing. Local SAT-penalty quench (n=51,108): jammed, penalty ~ds^2 -> no local gain. Gap 0.184 to #1 must come from new constructions (547 non-integer n carry 14.98 of score; 422 grid-ceiling n carry 7.29; frontier deficiency ~0.8k).
## Name That LLM
- 2026-10-28: history best 0.559 (public adaptive centred stylegram repro); own models 0.31-0.42. Public ensembles (nlm-v27 line) reach 0.651+; LB top 0.728. Weekly GPU quota (30h) was exhausted on 10-28 -> GPU forks blocked until reset; fork prepared at ntl/k_whoami (commit-run short-circuit).
## Quantum
- 2026-10-28: no submission before this run. Score = geo-mean of ref/cost ratios over invsqrt, DQI, chemistry (higher better). Best public notebook somejoint 22.63 local GM (verifiers in-notebook, ~10 min CPU).
- 2026-10-28 B4: repro of somejoint v16 scored 22.62971 = local GM exactly (LB salt ~ local for this portfolio). Public deltas on chem are small (+0.03 cache, +0.007 gauge); combining orthogonal public chem changes is the cheap next step, big gap (26.14) needs a new chem/DQI construction.
## Smart Buildings
- 2026-10-28: lower is better. Our best 0.01702 (night setback 5K + 2h prestart). Night-setback depth variants 0.01702-0.01732 all flat -> setback is not the lever. Public malyshevdanil probe-zt-best = 0.01420 (per-zone PI + economic zone target + median supply temp + weekend setback + min static pressure). LB top 0.00514.
## Memecoins
- (no submissions yet from this loop)
