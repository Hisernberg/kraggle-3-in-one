# Visual forced-choice FL arbitration (2026-10-06)

Human-in-the-loop check of the pipeline against the reference where their FL disagrees by > 15 mm (declared
manual analysis; the host allows manual test-set measurements for calibration, topic 690868). Renderer and scorer:
`scripts/forced_choice.py`. Each panel set shows a clean crop plus two candidate line sets (A/B, random order); the
rater picks the set that follows the visible fascicle fragments, with confidence 1-3 (X = cannot tell). Keys were
written before rating and not looked at until all answers were recorded.

| Set | Candidates | Picks | Correct | Confidence >= 2 |
|---|---|---|---|---|
| OSF benchmark (35 images, 7 experts) | expert FL vs expert FL x 0.78 / 1.28 | 34 | 26 (76 %) | 20/24 (83 %) |
| NeuAge VL (14 of 32 rendered) | expert FL vs expert FL x 0.78 / 1.28 | 14 | 9 (64 %) | 5/7 |
| Test, 35 units (clips counted once) | pipeline FL vs reference FL | 29 (6 X) | unknown | 11 |

Retest (`test_retest_*`): the 11 confidence >= 2 test units re-rendered on another frame of the same clip (or the same
image with 3 lines, seed 777); 8 of 11 kept their pick. Those 8 units (20 rows) define shot d11 S1
(`daily/inputs/roww_d11s1.csv`): FL weight 0.8 where the pipeline was picked, 0.1 where the reference was picked.
Bias seen on OSF: the rater prefers the longer (flatter) candidate (picked 68 % vs 56 % true).

## Leaderboard outcome (2026-10-07)
S1 (all 8 units) +0.0114; S2 (3 Lumify units only: gross 21-48 mm gaps, shorter candidate picked) -0.0024, the new best;
S5 (batch 2: `test_batch2_*`, 9 units with 9-15 mm gaps, all consistent on retest) +0.0018. Visual arbitration only
transfers on gross disagreements where the shorter candidate is picked.
