# Kaggriculture — status & decision log

Target: top 10 (LB ≥ ~2900). Checkpoints: CP1 ≥ 2600 (≈top 120) · CP2 ≥ 2720 (top 50) · CP3 ≥ 2830 (top 20) · CP4 ≥ 2900 (top 10).
Only the latest 2 submissions are active and the team is ranked by the better one. Pace (user): wait 5-6 h after a
submission, read the ladder, learn, then submit the next.

## Ladder log (UTC)

| # | when | agent | idea | rating (time) |
|---|---|---|---|---|
| P01 | 09-27 05:20 | tetsutani demand-preserving (public) | probe of the strongest current public agent | 2093 (07:49, then retired) |
| P02 | 09-27 06:05 | DSM tape 113970425_0 | replay a top player's recorded game + land catch-up | 1123 after 45 games (09:37) |
| P03 | 09-27 06:26 | cha22_edge_arm | cha22 + market layer (sell-ahead-1, order search, on-sight switch) | 2377 after 48 games (09:37); 2372 after 78 (11:46) |
| P04 | 09-27 11:50 | router r3_cg | tape router (280 tapes, day-start switching) + cash guard (fixes P02) + lazy per-tape decode (load 8 s → 0.1 s) | pending |

## What the ladder taught us

- **P02 failed for a mechanical reason**: the tape's opening ends day 0 with ~$5; ladder opponents buy more wheat on
  turn 0, so we reached day 1 with $0-1 and the 3 hires at step 24 failed. Every game where that happened was lost;
  games with the hires intact were mostly won. Fix: `_cg_guard` (keep cash for the next hour-0 hires; drop the least
  important evening seed buys, sell shed stock at hour 0 if needed). Local opponents never reproduced this.
- **P03 loses mostly by tiny margins** (−$50…−$200) to near-mirror cha22-family opponents with their own market layers.
  Fertilizer/egg order-index races decide those games; forcing fertilizer first or selling it ahead is net negative.

## Local evidence (fresh seeds, vs 9 strongest public agents, both seats)

| agent | win% | notes |
|---|---|---|
| cha22_edge_arm (P03) | 87-97% | most robust on every seed set |
| router r2_cg (tape router + fuzzy layout matching + cash guard) | 68-78% | switches among ~280 template-A tapes at day starts (layouts coincide days 1-6) by shop-draw match |
| single tape + cash guard | 63% | world-dependent: a tape either fits the world (wins all) or not |

vs top-team recorded games (forced worlds): every public agent and our tapes win ~37-42%; top teams are ~10-20% more
productive than the public cluster (eggs/geese, wheat, wool pricing).

## Next

11:45 review: P03 plateaued ~2375 (≈rank 250); P02 stuck ~1110 (cash bug). Submitted P04 = router + cash guard.
17:30 review → P05 (last of 09-27).
