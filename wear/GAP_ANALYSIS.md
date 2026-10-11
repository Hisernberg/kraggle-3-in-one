# Why two teams are at 0.944–0.945 and we are at 0.9376 (research of 2026-10-11)

## Facts (verified)
- The top teams' methods are not published (no writeups, repos or forum posts from Anonym, WearWise AI or #3–#6). goodpjw2008, the strongest public author, also says the leaders' gain "is not in anything public".
- The leaders' submission metadata shows only dates: Anonym 0.94535 on Oct 7 → 0.94538 on Oct 10 (an increment typical of window-level probing); WearWise AI 0.94355 on Oct 6, with about 6 entries (a late team, so a strong method or asset, not tuning).
- **The 2nd WEAR challenge test file is the same 12,234 seconds with all four limbs:** `test.csv`, 48,936 rows = 4 × 12,234, ids shuffled, part of the rows augmented by the organisers (jitter, scaling, rotation, masking), no labels ever released. One row was matched exactly (to float32) against this year's test window 5469. Sources: inagege/2nd_wear_dataset_challenge notebook outputs, arXiv 2510.21282, arXiv 2511.23173. The public WEAR release does not contain test subjects 22–25.
- Downloading that file requires accepting the 2nd challenge's rules on this Kaggle account; the API currently returns 403. The user's earlier PC sessions had it (`w25`) and used it only to re-score links, for about +0.002 LB.

## What I tested on train (complete 4-limb recordings), so this is not speculation
| Idea | Result |
|---|---|
| Re-chain complete single-limb streams by raw continuity (linear or quadratic extrapolation, Hungarian) | exact successor **12–33%**; same-label links about 50% |
| Group the 4 limbs of the same second by synchrony (magnitude-envelope correlation, ±3 sample lag) | **≈0%**: arm and leg signals are not similar enough |
| Learned same-limb successor scorer (LightGBM, subject-out) on complete streams | exact successor **43–56%** (public multi-limb links with video: about 65%) |
| The four bout-repair agents (video coherence, count intruders, learned limb signature, constrained re-decode) | only the sibling-count rule passed out-of-fold; on LB −0.00029 |

So the 2nd-challenge data does **not** trivially give back the time order or the 4-limb seconds. A learned linker with video features might reach about 70–80% on complete streams; that is the only route still open with a real chance.

## Plausible explanations for the leaders (unverified)
1. A much stronger linker or timeline decoder built on the 2nd-challenge 4-limb streams (complete streams remove the 75% "missing same-limb successor" problem).
2. Heavy window-level leaderboard probing (fits Anonym's small increments; mostly public-only gains).
3. If the 2nd challenge accepts late submissions, its scorer would work as a label oracle for the same seconds. **We will not do this:** it extracts hidden labels through another competition's scoring rather than modelling them, and is against the spirit of the competition.

## Plan (only with the user's go-ahead; nothing is guaranteed)
1. The user accepts the 2nd challenge rules (Kaggle web, as kragglenote2forwork) so `test.csv` can be downloaded. Before using it, check that the 3rd challenge allows it: its rules have no external-data clause, but asking the host in the forum is the safe move.
2. Build complete per-limb chains with a learned linker that uses accelerometer continuity plus shared video features. **Gate: at least 80% exact successor subject-out on train.**
3. Map every 3rd-test window to its exact twin, giving within-limb order for all windows. Feed the chains to the decode as extra link sets and as within-limb bout boundaries and counts. **Gate: out-of-fold gain of at least +0.003 on a train simulation.**
4. Submit at most 2–3 gated files on Oct 12; the final pick stays L11 unless one beats it.

Honest odds of reaching 0.94 by the deadline: low (about 10–20%), and only through step 2 passing its gate.
