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
| P04 | 09-27 11:50 | router r3_cg | tape router (280 tapes, day-start switching) + cash guard (fixes P02) + lazy per-tape decode (load 8 s → 0.1 s) | 1291 after 54 games (17:05); 1309 after 64 (23:00) — **failed** |
| (ext) | 09-27 13:50 | SB18 Macro-1 (not from this session) | tetsutani + turn-0 wheat duel | 1860 after 54 games (17:05), 34/40 wins; 1947 after 93 (23:00) |
| P05 | 09-28 00:08 | d_sb_s150a0 (id 56623202) | cha22 + SB18 opener + armed sell-ahead-2 (sell sub-1.3×-base stock first from step 150) | 1926/17 games (01:13) · 2231/39 (02:14) · 2310/56 (03:15) · 2282/71 (04:16) · 2273/82 (04:51) — plateau ≈ 2280, below P03 |
| P06 | 09-28 04:51 | t_sb_s150 (id 56629305) | tetsutani base + SB18 opener + armed sell-ahead-2 from step 150 (targets the duel-opener wheat route) | 1440/12 (05:22) · 1961/30 (06:30): 21/27, vs duel openers 11/12 (+5560) |

## What the ladder taught us

- **P02 failed for a mechanical reason**: the tape's opening ends day 0 with ~$5; ladder opponents buy more wheat on
  turn 0, so we reached day 1 with $0-1 and the 3 hires at step 24 failed. Every game where that happened was lost;
  games with the hires intact were mostly won. Fix: `_cg_guard` (keep cash for the next hour-0 hires; drop the least
  important evening seed buys, sell shed stock at hour 0 if needed). Local opponents never reproduced this.
- **P03 loses mostly by tiny margins** (−$50…−$200) to near-mirror cha22-family opponents with their own market layers.
  Fertilizer/egg order-index races decide those games; forcing fertilizer first or selling it ahead is net negative.

- **P04 (router) failed on the ladder**: 24/40 wins against low-rated opponents, many losses by 7k-120k. The day-1
  hires worked every game (cash guard OK), so the loss is the known world-dependence of tapes: the switch picks
  a tape whose production does not fit the world. Local 68-78% did not transfer. **The tape/router line is closed.**
- **P03 by opponent class** (87 games): vs "duel" openers (BUY wheat, SELL wheat on turn 0; tetsutani family) 24/42,
  +164 per game on average; vs cha22-family "BUY 5 wheat" 22/33, +72 per game; vs template-A animal-first 2/4. At
  ~2370 the ladder is near-mirror copies of the same production route, and games are coin flips decided by
  market micro-edges. Of 32 losses, 20 were by under $1000.
- **Mistake: an external submission displaced our best agent.** SB18 (13:50, from another session/person on the
  same account) was the 5th submission of 09-27. It used the last daily slot and pushed P03 (2370) out of the
  active pair. The active pair is now SB18 + P04, so the team shows ~1860. **Only one process may submit.** If
  another session keeps submitting, the slots below collide.

## Local evidence (fresh seeds, vs 9 strongest public agents, both seats)

| agent | win% | notes |
|---|---|---|
| cha22_edge_arm (P03) | 87-97% | most robust on every seed set |
| router r2_cg (tape router + fuzzy layout matching + cash guard) | 68-78% | switches among ~280 template-A tapes at day starts (layouts coincide days 1-6) by shop-draw match |
| single tape + cash guard | 63% | world-dependent: a tape either fits the world (wins all) or not |

vs top-team recorded games (forced worlds): every public agent and our tapes win ~37-42%; top teams are ~10-20% more
productive than the public cluster (eggs/geese, wheat, wool pricing).

## Mistakes & lessons (running list)

1. The loader takes the LAST callable in the file: always end with a unique entry function (`router_agent`, `edge_agent`).
2. Local win rate vs the public field does not predict the ladder for tape agents (P02 53/54 local → 1110; P04 68-78% → 1290).
   Edge-layer agents do transfer (P03 87-97% local → 2370). Trust only agents whose production adapts to the world.
3. Opponents on the ladder differ from local ones on turn 0 (bigger wheat buys, duel), which broke P02's cash. Every
   candidate now gets a day-1 cash/hands check in `harness/ladder_review.py`.
4. Do not submit anything until the daily count is known: the 6th submission of a UTC day gets HTTP 400 (17:10 test).
5. A new submission always displaces the older of the active pair. Plan the pair, not the single submission.
6. (09-28 S1) SB18 at ~1960 is 18/30 in its latest games; vs duel openers only 6/14 with an average margin of +1. The
   near-mirror coin-flip pattern holds at every rating level. Only the market micro-edge (armed selling) separates them.
7. (09-28 04:20) **P05 did not beat P03 on the ladder** (≈2290 vs 2370), despite 45/48 vs P03 locally. Ladder
   review, 68 games: vs duel openers 54% (21/39), 15 losses under $600; vs buy5 86%; vs other 79%. In 12 close
   losses to duel openers, the opponents pick up about 619 wheat per game vs our 488 and sell 902 vs 760.
   **They out-produce us on wheat (the tetsutani-lineage route). They do not out-trade us.** Our local duel proxy
   (public tetsutani) is weaker than the ladder's duel players, so local tests overrated the cha22 base. Next:
   the tetsutani base with our armed edge (t_sb_s150). Locally it beats tetsutani 24/24 and P05 34/48; an official
   game vs P05 won by +621, worst turn 0.18 s.
8. (09-28 06:30) Edge knobs DO matter on the tetsutani base, unlike on cha22. Local test, seeds 10000-10059:
   sa_full_below=2.0 (t6_fb20) beats P06 78/120 (+23), mostly by breaking exact mirror ties. It is equal or
   slightly better vs P05/tets/cha22/metav4 (+6..+30 margin, same win counts). Harmful settings: l2 off, arm_k=2,
   sa_from=216, and adding WHEAT to sell-ahead (−20k, catastrophic). Local wheat output of P05 ≈ P06 ≈ tetsutani
   (552 harvested/game), so the ladder wheat gap comes from ladder opponents' own variants, not from the base.

## Plan: automatic 5 slots per UTC day (09-28 → 09-30)

Slots (UTC), about 4.75 h apart: **S1 00:05 · S2 04:50 · S3 09:35 · S4 14:20 · S5 19:05**, plus a 23:00 prep on 09-27.
Each slot is a scheduled wake-up of this session and runs the **slot procedure**:

1. `kaggle competitions submissions kaggriculture`: record the rating and game count of the 2 active submissions in the
   ladder log; note any submission not made here.
2. `python -m harness.ladder_review <newest id>`: W/L by opponent class, margins, day-1 cash. Write the lessons above.
3. Pick the next agent from the queue (below), adjusted by what the ladder just showed.
4. Validate: one official `kaggle_environments` game (DONE, no errors), plus a local fastsim check vs the field if the
   agent is new.
5. Submit with a descriptive message → commit/push the STATUS update → HF sync.

**Pair policy.** Only the latest 2 are active; LB = the better one. A ~4.75 h slot gives ~50 games, enough for a
rating within ~±50. So every slot measures one candidate while the previous one keeps playing. On 09-30, S4 = the
second-best measured agent and S5 = the best (the final active pair).

**Turn-0 duel, measured locally (09-27 evening)**: prices are quoted per unit in lockstep, and BUY is quoted at
price(inv-1). So [BUY a, SELL b, BUY c] only gains when the rival buys more at the same index, and against our
[BUY 5, SEED] it changes nothing. It does cash-starve the v54-v57 lineage ([B20,S15]): about 24/24 wins, +15k.
The real gain came from "always armed" selling (sell everything priced below 1.3x base, first in the order list,
from step 150). It wins the early milk/strawberry sale races against near-mirrors.

| candidate | vs v_sa2 | vs P03 | vs tetsutani | vs cha22 | field |
|---|---|---|---|---|---|
| **d_sb_s150a0** (cha22 + SB18 opener + armed sa, `submissions/p05_d_sb_s150a0`) | 45/48 +496 · my check 14/16 | 47/48 · 16/16 | 13/24 · 12/16 | 24/24 · 16/16 | 88% (v_sa2 82%) |
| t_sb_s150 (tetsutani base, `submissions/cand_t_sb_s150`) | 38/48 · my check 10/16 −5 | 38/48 · 10/16 | 24/24 · 14/16 | 20/24 · 10/16 | 91% |

Agents are built from `agents/edge/bases/base_sb.py` / `base_tsb.py` with the v_sa2 config plus sa_from=150, arm_k=0.

**Queue for 09-28** (re-ordered at every slot by evidence):
- S1 00:05: **P05 = d_sb_s150a0**. It passed 2 official games (DONE, won both; worst turn 0.07 s).
- S2 04:50 (**decided 04:20: submit t_sb_s150**, see lesson 7): if P05 is ≥ 2400 with ≥ 40 games, submit **t_sb_s150** (a different base, as a diversity probe; worst
  turn about 0.5 s, within the 1 s limit). If P05 is under 2300, re-submit P03 as the anchor and study P05's losses.
- S3 09:35 rule: if P06 ≥ 2250 (or still climbing above P05's pace) → submit **submissions/cand_t6_fb20** (P06 +
  sa_full_below 2.0; official game DONE, worst turn 0.14 s); pair becomes P06 + fb20. Else → re-submit P03 (2370 anchor).
- (earlier notes) S3-S5: driven by P05's ladder losses (ladder_review). The planned knob deltas were tested at 23:00 on seeds
  9900-9907, both seats. They are flat: sa_from=100 and arm_before=720 play identically to P05. fb1.4 goes 9/16 vs
  P05 (+$1), fb1.2 goes 7/16, and sa_from=200 goes 3/16. So P05's knobs sit at a local optimum and won't be
  submitted. P05 on these fresh seeds: v_sa2 15/16 (+777), P03 15/16, tetsutani 14/16, metav4 14/16, cha22 16/16,
  v57 16/16. Next deltas must target what the ladder losses show (e.g. the animal-first/template-A class).

**Checkpoints**: CP1 ≥ 2600 → keep that agent as the anchor and test only small deltas on it. Below 2400 after
09-29 S3 → stop exploring and spend 09-30 on the best measured pair.

**Scheduled wake-ups (send_later, fire into this session; cancel with delete_trigger):**
09-27 23:00 prep `trig_01M1EtUTLmP3UbCF3cCSMQxX` ·
09-28 S1 `trig_0185y4mzWG9DwrLnkmZecihS` S2 `trig_01PPqCKF7VuDB37fUXrc8oRc` S3 `trig_01FyaY6RLKTviP8xpHR3BYh9` S4 `trig_01BW5BcmeCBH79wSmDfUUW4i` S5 `trig_01BnyuvSH9nEKGrqB6naw79M` ·
09-29 S1 `trig_016BBFdXBV5Q1maYN4DCSyka` S2 `trig_018YaoqzoqW1h9aLMFyevzuk` S3 `trig_01YZZXpS5mhthkDBXDvRAspD` S4 `trig_01HJAjLAhkoR6kPE7SchYP8j` S5 `trig_01Xwqqf2RuZGEatHZKy4Gxvm` ·
09-30 S1 `trig_01FJhBAskFm5dWZMFFDgxhu7` S2 `trig_01Xf4ztN7UHHtksUc6QXwRk9` S3 `trig_01PSRZBibM7kLvvR9WXC25bz` S4 (hedge) `trig_01JTS2K7iLK8LP5RXysPHBPS` S5 (keeper) `trig_01MudqtTYzcukZLk6nHuXSqM`.

Overnight (09-27 17:15 → 22:45): background build + local test of duel-answer candidates (v_sa2 + turn-0 duel
opening / counter, edge layer on the tetsutani base) vs a mirror set (P03, v_sa2, duel openers, field_top).
