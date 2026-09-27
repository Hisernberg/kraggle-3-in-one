# Kaggriculture: what the top players do (replay study, 2026-09-27)

## Data and method

- Leaderboard pulled 2026-09-27. Top 3: **DSM** 3083.9, **Boey** 3041.3, **Vadim Vasilenko** 2985.0. Places 4-12: M & M & P & Q, Majkel1337, DECEM, KawattaTaido, Fourth Quadrant, Unknown Mother-Goose, Anton Tikhonov, Azat Akhtyamov, Just A game on your lips.
- For each of the top 12 teams I took the highest-scoring active submission, listed its episodes, and downloaded up to 7 recent ones. Games where both players are top-20 were preferred. That gave **62 unique top-level episodes** in `replays/top/`: 60 are top-20 vs top-20, and the other two involve Pii (not in the top 20). I also downloaded **10 recent episodes of our submissions** 56556103 and 56556096 (team "Fried Chicken Lovers") into `replays/ours/`. The episode metadata, including opponents and rewards, is in `replays/episodes_meta.json`.
- `research/replay_parse.py` re-simulates every turn's unit actions and market phase with the engine's own functions. It reproduces the recorded money exactly, with 0 mismatches over all 72 replays. So every SELL, BUY, HIRE and BUY_LAND fill has its exact per-unit prices. `research/replay_analyze.py` builds the per-team aggregates. Outputs are in `research/parsed/`: `top.json`, `ours.json`, `team_stats.json`, `team_stats.txt`, and `*_table.txt`.
- Replay indexing: `steps[t][p].action` is the action taken at game step `t-1`. The day is `(t-1)//24` and the hour is `(t-1)%24`. The last action an agent sends is at step 718.

Caveats: sample sizes are 5-16 games per team. Final coin totals depend heavily on the random town-shop draws, which both players share. That is why winners range from 58k to 165k. **Compare margins, not raw totals.**

---

## 1. Headline findings

1. **Almost the whole top of the leaderboard runs one public template ("A").** DSM, Vadim, M & M & P & Q and Unknown Mother-Goose issue identical market orders for the first 3 turns, in all 45 of their games. DSM vs Vadim stay action-identical until step 11. KawattaTaido, Anton, Azat, DECEM, Majkel and Just-A-game are small re-orderings of the same template. Games between them are decided by margins of about 3% (median 3.0%, mean $4.4k).
2. **Our submissions run a different, weaker public template ("B").** It opens at t0 with `BUY_PRODUCT WHEAT 20; SELL WHEAT 15` (or 8/3), plants 12 melons on day 0, and buys land 2 on day 11. **9 of our 10 recent opponents run the same B agent**, so our rating is set in mirror matches. In the one game against a template-A player (Humanitis) we lost **107.9k vs 139.8k (-23%)**.
3. **The top template's edge:**
   - it reinvests every coin: cash stays near $0 through day 7 and land 2 comes on day 8, not day 11;
   - it fertilizes wheat so it grows nearly all of its feed: it buys 120 wheat per game ($4.2k), where we buy 375 ($15k);
   - it runs about 7 geese;
   - it **holds premium goods and drip-sells 2-4 units right after each town-consumption tick** (steps with `step % 4 == 1`), never dumping at the $1 floor;
   - it retires its sheep during days 20-28.

---

## 2. Final money: winners vs losers

| set | games | winner mean | loser mean | mean margin | median margin % |
|---|---:|---:|---:|---:|---:|
| top-12 episodes | 62 | 103,696 | 99,317 | 4,379 | 3.0% |
| our episodes | 10 | 110,580 | 105,672 | 4,908 | 1.1% (we lost 8/10; our mean margin −3.6%) |

Per-team results in this sample. Final totals depend on the opponent and the shop draws. The margin % is the more useful column.

| team (LB rank) | games | W | mean final | mean margin | margin % |
|---|---:|---:|---:|---:|---:|
| DSM (1) | 11 | 10 | 112,434 | +3,232 | +2.9% |
| Boey (2) | 9 | 3 | 100,976 | -1,132 | -1.0% (6 of 9 games were vs DSM) |
| Vadim Vasilenko (3) | 16 | 11 | 108,338 | +2,463 | +2.6% |
| M & M & P & Q (4) | 8 | 3 | 103,657 | -455 | -0.6% |
| Majkel1337 (5) | 9 | 6 | 103,114 | +167 | +0.4% |
| DECEM (6) | 8 | 4 | 101,618 | +2,362 | +2.6% |
| KawattaTaido (7) | 11 | 8 | 97,502 | +4,407 | +5.4% |
| Fourth Quadrant (8) | 10 | 6 | 91,602 | -1,260 | -1.2% |
| Unknown Mother-Goose (9) | 7 | 0 | 105,453 | -2,942 | -2.8% |
| Anton Tikhonov (10) | 10 | 4 | 94,319 | -3,333 | -3.0% |
| Azat Akhtyamov (11) | 8 | 4 | 103,322 | -1,005 | -0.6% |
| Just A game on your lips (12) | 5 | 0 | 94,759 | -2,214 | -2.4% |
| **Fried Chicken Lovers (us)** | 10 | 2 | 105,735 | -4,783 | -3.6% |

The full per-episode table is in the appendix.

---

## 3. The canonical top policy: DSM (#1). Vadim (#3) is nearly identical.

### 3.1 Day-0 build order (identical in all 11 DSM games and all 16 Vadim games)

F = main farmer, H = the 5 hired hands, M = market orders (in order). The farmer spawns at (4,4). All animal structures go on tiles next to the shed.

| hour | F | H (hand1..hand5) | M |
|---|---|---|---|
| 0 | PASS | – | `BUY_ANIMAL COW 1`, `BUY_PRODUCT WHEAT 5`, `BUY_ANIMAL SHEEP 1` ×3 |
| 1 | PICKUP COW 1 | – | `SELL WHEAT 1`, `HIRE`×4, `BUY_ANIMAL COW 1`, `HIRE` (5 hires, total cost $12) |
| 2 | BUILD_PASTURE | PICKUP SHEEP, PICKUP SHEEP, PICKUP COW, PICKUP SHEEP, PICKUP COW | `SELL WHEAT 1` |
| 3 | PLACE COW | N, N, N, W, N | `SELL WHEAT 1`, `BUY_PRODUCT WHEAT 1` |
| 4 | PICKUP WHEAT 3 | W, W, N, BUILD_PASTURE, N | `BUY_PRODUCT WHEAT 1` |
| 5 | CARE | BUILD_PASTURE, PLACE SHEEP, N, PLACE SHEEP, N | `BUY_SEED MELON 2`, `BUY_PRODUCT WHEAT 1` |
| 6 | W | PLACE SHEEP, CARE, W, W, W | `BUY_SEED MELON 2` |
| 7 | FEED | CARE, W, BUILD_PASTURE, BUILD_PASTURE, PLANT MELON | – |
| 8 | E | S, N, PLACE COW, PLACE SHEEP, WATER | `BUY_SEED WHEAT 1` |
| 9 | N | N, PLANT MELON, N, W, N | `BUY_SEED WHEAT 1` |
| 10 | FEED | W, WATER, W, PLANT MELON, PLANT WHEAT | `BUY_SEED MELON 2` |
| 11 | S | PLANT MELON, N, PLANT WHEAT, WATER, WATER | `BUY_SEED WHEAT 3` (Vadim: 1) |
| 12 | W | WATER, PLANT WHEAT, WATER, W, W | `BUY_SEED MELON 2` (Vadim: WHEAT 3) |
| 13-21 | FEED/CARE then PASS | plant+water in pairs (PLANT X, then WATER the same tile next turn) | 1 wheat seed per turn, `SELL WHEAT 2` at h14 |
| 22-23 | PASS | PASS | – |

End of day 0: 2 cows and 3 sheep on 5 pastures (fed and cared), **6 melons and 12 wheat** planted and watered, cash about $6.

Key habits:
- Seeds are bought **just in time**, 1-3 per turn, right before a hand plants them.
- Every new plant is watered on its planting turn or the next one. The planting day counts as unwatered.
- All cash is spent.

### 3.2 Days 1-10 (averages over 11 DSM games; Vadim is the same)

- **Day 1:** 3 hires. 2 more melons (8 in total). Buy about 8 wheat for feed. Sell the collected fertilizer 1-2 units per turn for cash.
- **Days 2-5:** 5-6 hires per day. Add 1 cow per day (about 5 cows by day 5). Harvest the day-0 wheat at ages 2-4 and replant those tiles with **strawberries** (2 on day 2, about 4 on day 3, about 2 on day 4). Cash stays at $10-400.
- **Day 6, hour 3:** the first wool is sold (6 units), and **`BUY_LAND` NE ($1k) goes in the same turn**. Within that hour 12-13 strawberries, 4 melons, 3 geese and about 1 sheep are bought, and hires jump to 9.
- **Day 8, hours 4-6:** `BUY_LAND` SW ($2k) is spammed every turn until cash allows. It succeeds on average at day 8.2. Then about 4 more geese (7.3 per game in total by day 9), and SW is filled with wheat.
- **Day 10:** about 47 wheat bought (seed and feed). Cash is about $9k.
- **The $4k SE quadrant is never bought.**

### 3.3 Layout

- Pastures and coops cluster on the tiles next to the shed. In NW that is columns 2-4 and rows 2-4. In NE it is column 5. Geese go in SW at (2-4, 5-6).
- Crops fill the outer tiles.
- A snapshot from DSM vs M&M (ep 113963510), where `C`=cow, `S`=sheep, `G`=goose, `s`=strawberry, `w`=wheat, `m`=melon, `c`=carrot:

```
d10                   d20                   d25
s s s s C s s s s w   s w . w C s s s s w   w c w w C w w w w c
s s C C S s s s s s   w w w C S s s s s s   w w w C S w w w w c
s C C w C C s s s s   w C C w w C s s s s   w C P w w C w w w w
m m S S S C m s s s   w w S S S C w s s s   w w S S S C w w w w
w w S S C C m m s s   s s S S C C w w s s   s s S S C C w w w w
w w G G S # # # # #   w s G G S # # # # #   w s G G S # # # # #
w w w w G # # # # #   w w w w G # # # # #   w w w w G # # # # #
w w s w S # # # # #   w w s w S # # # # #   w c w w c # # # # #
...
```

### 3.4 Product mix (DSM, per game)

- **Tiles:**
  - Day 10: 25 wheat, 24 strawberries, 7 geese, 7 cows, 6 melons, 5 sheep.
  - Day 20: 25 wheat, 25 strawberries, 7 cows, 7 geese, 5 sheep, 3.5 carrots.
  - Day 25: 32 wheat, 14 carrots, 8 strawberries, 7 geese, 7 cows, 3 sheep.
- **Animals bought:** 7.5 cows, 5.8 sheep, 7.3 geese.
- **Seeds bought:**
  - 32.5 strawberries (days 2-15, mostly day 6);
  - 12 melons (days 0-1, plus about 4 on day 6);
  - about 170 wheat seeds;
  - 42 carrots (days 10-26);
  - about 1 tomato.
- **Revenue per game:** strawberry 37.0k (at $150 average), milk 21.4k (at $112), wheat 17.2k (at $33), wool 16.2k (at $130), melon 14.3k (at $200), fertilizer 12.4k (at $53), egg 11.5k (at $43), carrot 5.7k (at $41). Gross is 137k.
- **Spend:** about 19k on seeds and animals, plus 5.0k on hires.
- **Fertilizer use:**
  - 461 collected per game;
  - 223 used (about 110 on **wheat at age 2**, which covers the whole wheat bonus window of ages 2-4 so each wheat tile yields 6);
  - about 70 used on **strawberries at ages 9 and 13**, which doubles all 4 productions;
  - the rest (234 units) sold at about $53.
- **Wheat for feed** comes mostly from their own fields: only 120 wheat bought via BUY_PRODUCT per game.
- **CARE** goes on every animal every day (about 390 CARE actions per game).

### 3.5 Hires

Mean hires per day, days 0-29:

```
5, 3, 5, 5, 6, 5.6, 9, 7.4, 10, 9.9, 11.2, 10.6, 10.6, 10, 9.9, 11.2, 11.6, 10.8, 11.3, 11.4, 11.2, 11.2, 11.5, 10.8, 11, 11.3, 11, 10.6, 10.4, 9.1
```

Hires are issued at hour 0, which gives 11-12 hands (fib cost of about $230 per day). Total hire cost is about $5.0k per game.

### 3.6 Selling policy

Town shops consume every 4 steps: at `step % 4 == 0`, **after** the market phase. The first step at which the replenished demand can be sold into is therefore `step % 4 == 1`, which is hours 1, 5, 9, 13, 17 and 21.

- **Premium goods (strawberry, milk, wool, melon) are sold almost only at hours 0/1, 5, 9, 13, 17 and 21.**
  - DSM premium units by hour: `[594, 792, 148, 224, 197, 467, 338, 102, 62, 823, 186, 131, 136, 796, 58, 116, 104, 672, 123, 24, 51, 656, 40, 143]`.
  - Milk: 1465 of 2110 units were sold at `step % 4 == 1`. Strawberry: 1698 of 2722.
  - Their sale is first in line after each tick, ahead of an opponent who sells later in the 4-step window.
- **Small lots.** The median lot is 2-3 units: strawberry 4.1, milk 3.7, wool 2.9.
- **Stock is held while the price is depressed.**
  - Example: on DSM day 21 the strawberry price was $78 at hour 0 with 21 units in the shed. They held until it recovered to $118 at hour 13, then sold 2 per tick.
  - On day 24 they held 41 strawberries and sold 2-4 per tick at $140-160.
  - 65% of DSM's strawberry units are sold at ≥1.3× base. Only 4% go below 0.3× base.
- **Milk and wool** are sold every tick in lots of 3-6 even when the price is depressed ($30-50 for milk). Both players' cows flood milk, and holding does not help. Still, almost nothing is sold at the $1 floor: about 1 unit per game.
- **Eggs, wheat and carrots** go in larger lots (median 5-8) at any hour.
- **Fertilizer** is sold 1-2 units at a time throughout the game. It pays for the early buys.
- **End game:**
  - Premium stock is kept at a small level until the last day, then **liquidated back-loaded on day 29**. In the DSM ep 113963510 trace: strawberry 23 in stock, sold 2 at h5, 2 at h9, 6 at h13, 12 at h17, 6 at h18, then 1-2 more by h22.
  - The final steps (711-718) sell everything left (wheat, carrots, eggs, fertilizer). The shed ends about empty (about 2 items).
  - The last carrot seeds are bought on day 26 and the last wheat seeds on day 27. No strawberries after about day 15, no melons after day 6, no animals after about day 18.
  - **Sheep are allowed to escape from about day 20 onward**, going from 8-10 down to 0 by day 28, probably by no longer feeding them. That saves feed wheat and hand-turns once little wool will still come.

### 3.7 Turn-0 "wheat duel"

- The t0 `BUY_PRODUCT WHEAT 5` costs $133-139. Two template-A players buy at the same time, which pushes the price up.
- They then sell 1 unit on each of t1, t2 and t3 at $28-29. That gives a few dollars of edge and sets up feed for day 0.
- No other opponent-reactive logic is visible. About 70% of their sells fall within ±1 step of an opponent sell of the same item. This is only because both run the same tick-aligned schedule.

---

## 4. The other top teams

| team | family | how it differs from DSM |
|---|---|---|
| **Vadim Vasilenko (#3)** | A3 | Same code as DSM. Diverges at step 11 with small seed-order and timing changes. Slightly more geese (8.6) and carrots (54), fewer cows (6.1). Lost 0-2 to DSM (−8.9%, −1.8%). Beats Mother-Goose 3-0. |
| **Boey (#2)** | A-like farm + **market churn** | The farm plan is close to A: 3 cows + 2 sheep + 7 melons + wheat on day 0, geese on day 2, land on days 6 and 8.5, never $4k. On top of that it runs massive wheat and fertilizer round-tripping: about 3,500 wheat and 766 fertilizer bought and about 3,800 wheat and 1,000 fertilizer sold per game, 1,444 sell orders per game, all 10 order slots used every turn. Wheat net (revenue minus BUY_PRODUCT) is +9.3k vs DSM's +13.0k, so the churn does not visibly pay. Boey went 1-5 against DSM; five of the six games were within 3%. |
| **M & M & P & Q (#4)** | A3 clone | Identical opening. Diverges around step 4-5. 8 cows, 6.9 sheep, 6.9 geese. |
| **Majkel1337 (#5)** | A variant | Opening: 5 hires, then cow + 1 sheep + `BUY_PRODUCT WHEAT 10`. NE on day 6. SW on day 9.2, sometimes SE on day 10.5. About 14 tomatoes (sold at $75). Fewer strawberries (24.6). |
| **DECEM (#6)** | A3 | Opening orders reordered: cow, 3 sheep, wheat 5, then 5 hires plus a cow at t1. The most sheep of the A3 group (8.4) and the most wool revenue (25k). |
| **KawattaTaido (#7)** | **A4** (4 quadrants) | Opening: `BUY_PRODUCT WHEAT 5; COW 1`, then t1 4 hires + cow + 3 sheep. It buys **all four quadrants**: NE on day 6.2, SW on day 9.1, **SE ($4k) on day 10.6**. It fills them with **tomatoes (19, sold at $93)** and **carrots (102)**. It runs 9.3 cows and only 3.3 sheep, and hires 12 per day from day 10 on ($7.3k). Best margin in the sample: +5.4%, 8-3. |
| **Fourth Quadrant (#8)** | churn + 4 quadrants | 4 cows + 2 sheep on day 0, **no geese**, churns about 1,700 wheat and 760 fertilizer, buys SE on day 10.4. 6-4. Lost 0-2 to KawattaTaido (one by 31%). |
| **Unknown Mother-Goose (#9)** | A3 clone | Same first 3 turns as DSM, but went 0-7 here. It holds more strawberries (30 tiles on day 10), gets fewer sheep and cows, plants about 12 tomatoes, and sells wheat more slowly. |
| **Anton Tikhonov (#10), Azat Akhtyamov (#11), Just A game (#12)** | A4 | Same as KawattaTaido: 4 quadrants plus 12-22 tomatoes. They leave 12-22 tiles EMPTY on day 10 after buying SE, so the land sits idle. Their margins are mixed. |

Head-to-head by family (60 top-20 games, plus Pii):

| matchup | record | note |
|---|---|---|
| A3 (DSM, Vadim, M&M, Mother-Goose, DECEM) vs A4 (4-quadrant plus tomato) | 8-6 | Winning margins run 1-14% either way |
| A3 vs churners (Boey, Fourth Quadrant) | 7-5 | DSM went 5-1 against Boey |
| A4 vs churners | 3-3 | |
| DSM vs everyone | 10-1 | The only loss was to Boey, by 0.4% (ep 113985943) |

Across all top games, the winner minus loser revenue averaged:
- **+1.6k strawberry, +0.6k milk, +0.6k wool, +0.4k egg, +0.2k carrot**;
- about the same on melon;
- **less** wheat and fertilizer churn (the churners lose more often).

**What separates #1-3 (DSM and Vadim) from #10-20:**
- They sell strawberries at a higher average price (150-157 vs 117-133).
- They run the A3 land schedule (days 6 and 8, no SE) instead of paying $4k on day 10.5 and leaving land idle.
- They keep a balanced animal mix (about 7 cows, 6 sheep, 7-9 geese).
- They phase out sheep late in the game.
- They do not churn the market.

KawattaTaido's 4-quadrant plus tomato plan is the only variant that beats A3 on margin. Tomatoes are under-supplied because nobody else grows them: pizza shops and farmers markets drain them, and DSM got $177 for the few it sold.

---

## 5. Why our submissions lose (template B vs template A)

Numbers are per game: ours (10 games) vs DSM (11 games).

| aspect | ours | DSM / top | impact |
|---|---|---|---|
| t0 | `BUY_PRODUCT WHEAT 20` + `SELL WHEAT 15` (or 8/3) + 1 wheat seed | `COW`, `WHEAT 5`, 3× `SHEEP` | Our animals arrive a turn later |
| Day 0 | 2 cows, 2 sheep, **12 melons**, 8 wheat | 2 cows, 3 sheep, 6 (+2) melons, 12 wheat | Fewer wheat tiles, so less feed |
| Land 2 ($2k) | **day 11.0** | day 8.2 | We lose 3 days × 25 tiles. On day 10 we still have 50 LOCKED tiles and **$15.9k idle cash**, vs $9k fully invested |
| Geese | **2.5** | 7.3 | Egg revenue 4.2k vs 11.5k |
| Sheep | 7.9, kept to the end (up to 17 in mirror games) | 5.8, retired from day 20 | Wool is crashed to the floor in our mirror games |
| Fertilizer | 380 collected, **113 used**, 337 sold at $44 | 461 collected, 223 used (wheat at age 2, strawberry at 9 and 13), 234 sold at $53 | Our wheat yields less |
| Feed wheat bought | **375 units, $15.0k** | 120 units, $4.2k | About **−$10k per game** |
| Premium sell lot size | strawberry 7.3, milk 5.2, wool 5.0 | 4.1 / 3.7 / 2.9 | |
| Units sold at ≤5% of base | **strawberry 57, milk 24, wool 33 per game** | 0-3 | Selling at $1 gains $1 and wastes the unit. Roughly −$8-10k per game |
| Strawberry average price | $114 (23% of units below 0.3× base) | $150 (4% below 0.3× base) | −$8.7k strawberry revenue |
| Sell timing | spread over all hours; peaks at h0 and h19 | on the ticks (h1, 5, 9, 13, 17, 21) | We sell after the opponent has taken the fresh demand |
| Hires | 4-5 on days 2-5, 7-8 on days 6-9 | 5-6, then 9-10 | Slower planting |

In our mirror games, both copies of agent B dump the same item on the same step. That drives milk, wool and strawberries to $1, which is visible in the floor sells in `parsed/ours.json`.

Against a template-A opponent, the investment lag and the selling gap compound to about −23% (episode 113978185 vs Humanitis): wool $85 vs $123 average, strawberries $103 vs $120, and cash idle on day 10.

---

## 6. Actionable recommendations, in priority order

1. **Adopt the A3 opening exactly** (section 3.1): `COW`, `WHEAT 5`, `SHEEP` ×3 at t0; 5 hires plus a second cow at t1; 5 pastures next to the shed; 6-8 melons and 12 wheat on day 0; just-in-time seed buys; spend cash to about 0.
2. **Land:**
   - NE as soon as the first wool sells (about day 6, hour 3);
   - SW by about day 8 (spam `BUY_LAND` each turn until it fills);
   - skip SE unless you have a tomato and carrot plan to fill it at once, as KawattaTaido does.
3. **Fertilize** wheat at age 2 and strawberries at ages 9 and 13. Grow your own feed wheat. Use the remaining fertilizer and sell only the surplus, 1-2 units at a time.
4. **Selling:**
   - Sell premium goods only at `step % 4 == 1`, in lots of 2-4, roughly matching the demand consumed per tick.
   - Hold strawberries while the price is below about base, since it recovers with every tick.
   - Never sell at ≤ $5 before the last day.
   - Liquidate on day 29 in a back-loaded way (h13-h22), and sell everything in steps 711-718.
5. **Geese:** about 7 geese on days 6-9. **Cows:** about 7. **Sheep:** about 6, retired from day 20. **CARE** every animal every day.
6. **Late game:** turn freed strawberry and melon tiles into wheat and carrot cycles. Last carrot seeds on day 26, last wheat seeds on day 27.
7. Optional edges seen only in sub-top teams:
   - a handful of tomatoes (demand stays unmet, 1.3-3× base);
   - KawattaTaido's heavier hiring (12 per day).

---

## 7. Tools

```
# parse one replay (prints per-player features)
python research/replay_parse.py replays/top/episode-113963510-replay.json --print
# parse a folder -> json + one line per game
python research/replay_parse.py replays/top --table --json research/parsed/top.json
# per-team aggregates
python research/replay_analyze.py research/parsed/top.json research/parsed/ours.json > research/parsed/team_stats.txt
```

`parse_replay()` returns:
- per-player `timeline`: every turn with the farmer, hands and market actions, money, hand count, shed, seeds, and tile counts at day end;
- exact market `events`: each order with the requested and filled quantity and every unit price;
- per-step `prices`;
- layouts on days 0, 1, 2, 5, 10, 15, 20, 25 and 29;
- the shop unlock timeline.

`features()` derives:
- the day-0 opening and the day 1-2 market orders;
- buys by day;
- land timings;
- hires by day;
- sell events;
- revenue, units and average price by product;
- BUY_PRODUCT events;
- sell-hour histograms;
- the money curve;
- the end game (days 28-29) and the final shed;
- the action mix;
- sells close to opponent sells;
- the turn-0 wheat duel.

---

## Appendix: per-episode final money

**top episodes (62)**

| episode | winner | winner $ | loser | loser $ | margin | margin % |
|---|---|---:|---|---:|---:|---:|
| 113950605 | DSM | 164,716 | Boey | 161,825 | 2,891 | 1.8% |
| 113978085 | DSM | 140,517 | Majkel1337 | 133,904 | 6,613 | 4.9% |
| 113986014 | M & M & P & Q | 133,442 | Vadim Vasilenko | 132,203 | 1,239 | 0.9% |
| 113976995 | Vadim Vasilenko | 131,127 | Azat Akhtyamov | 128,689 | 2,438 | 1.9% |
| 113957215 | DSM | 129,665 | Vadim Vasilenko | 119,104 | 10,561 | 8.9% |
| 113968699 | Anton Tikhonov | 129,453 | My second life | 123,682 | 5,771 | 4.7% |
| 113985941 | Vadim Vasilenko | 125,704 | Anton Tikhonov | 114,228 | 11,476 | 10.0% |
| 113986114 | Fourth Quadrant | 123,625 | Vadim Vasilenko | 122,705 | 920 | 0.7% |
| 113948780 | KawattaTaido | 120,410 | Pii | 111,716 | 8,694 | 7.8% |
| 113946460 | Majkel1337 | 119,918 | Unknown Mother-Goose | 117,936 | 1,982 | 1.7% |
| 113978231 | M & M & P & Q | 119,681 | Russell Kirk | 108,590 | 11,091 | 10.2% |
| 113963510 | DSM | 119,138 | M & M & P & Q | 115,602 | 3,536 | 3.1% |
| 113969983 | Fourth Quadrant | 118,621 | DECEM | 111,942 | 6,679 | 6.0% |
| 113964674 | KawattaTaido | 117,234 | M & M & P & Q | 110,623 | 6,611 | 6.0% |
| 113939686 | Vadim Vasilenko | 116,756 | Unknown Mother-Goose | 115,377 | 1,379 | 1.2% |
| 113985942 | DSM | 116,442 | Boey | 116,271 | 171 | 0.1% |
| 113966431 | DECEM | 116,365 | Azat Akhtyamov | 102,192 | 14,173 | 13.9% |
| 113970141 | Vadim Vasilenko | 116,187 | Azat Akhtyamov | 109,326 | 6,861 | 6.3% |
| 113985947 | Azat Akhtyamov | 115,510 | Anton Tikhonov | 106,413 | 9,097 | 8.5% |
| 113966429 | Yizhou | 114,813 | Anton Tikhonov | 112,567 | 2,246 | 2.0% |
| 113970425 | DSM | 112,354 | Boey | 109,355 | 2,999 | 2.7% |
| 113984725 | Azat Akhtyamov | 110,954 | We wanna be tomatos | 108,310 | 2,644 | 2.4% |
| 113970163 | Majkel1337 | 110,568 | Just A game on your lips | 108,109 | 2,459 | 2.3% |
| 113957149 | M & M & P & Q | 108,714 | Unknown Mother-Goose | 105,875 | 2,839 | 2.7% |
| 113985951 | Majkel1337 | 108,203 | Unknown Mother-Goose | 108,157 | 46 | 0.0% |
| 113976853 | KawattaTaido | 107,026 | Just A game on your lips | 106,409 | 617 | 0.6% |
| 113962528 | Vadim Vasilenko | 107,004 | Unknown Mother-Goose | 98,395 | 8,609 | 8.7% |
| 113959718 | DECEM | 106,439 | Anton Tikhonov | 93,936 | 12,503 | 13.3% |
| 113970139 | Vadim Vasilenko | 104,567 | Unknown Mother-Goose | 101,401 | 3,166 | 3.1% |
| 113963514 | DSM | 104,495 | Vadim Vasilenko | 102,687 | 1,808 | 1.8% |
| 113988881 | Vadim Vasilenko | 102,715 | DECEM | 100,789 | 1,926 | 1.9% |
| 113981752 | DECEM | 102,502 | Fourth Quadrant | 98,158 | 4,344 | 4.4% |
| 113978089 | Vadim Vasilenko | 102,252 | Boey | 98,133 | 4,119 | 4.2% |
| 113990156 | Vadim Vasilenko | 101,174 | Yizhou | 92,919 | 8,255 | 8.9% |
| 113973950 | mtmr_s1 | 100,798 | KawattaTaido | 98,753 | 2,045 | 2.1% |
| 113978094 | Majkel1337 | 100,512 | DECEM | 96,782 | 3,730 | 3.9% |
| 113985949 | Vadim Vasilenko | 97,240 | KawattaTaido | 91,050 | 6,190 | 6.8% |
| 113983150 | KawattaTaido | 96,464 | Fourth Quadrant | 73,408 | 23,056 | 31.4% |
| 113979171 | Fourth Quadrant | 96,118 | Majkel1337 | 93,359 | 2,759 | 3.0% |
| 113971301 | Azat Akhtyamov | 94,802 | KawattaTaido | 92,347 | 2,455 | 2.7% |
| 113970418 | DSM | 94,433 | Majkel1337 | 93,174 | 1,259 | 1.4% |
| 113980502 | KawattaTaido | 94,408 | Fourth Quadrant | 89,278 | 5,130 | 5.7% |
| 113955912 | DECEM | 93,604 | Unknown Mother-Goose | 91,031 | 2,573 | 2.8% |
| 113962363 | Azat Akhtyamov | 93,421 | Just A game on your lips | 91,741 | 1,680 | 1.8% |
| 113985943 | Boey | 93,357 | DSM | 93,021 | 336 | 0.4% |
| 113978091 | KawattaTaido | 91,333 | M & M & P & Q | 86,489 | 4,844 | 5.6% |
| 113976999 | Anton Tikhonov | 90,271 | Just A game on your lips | 88,031 | 2,240 | 2.5% |
| 113963858 | Boey | 87,024 | M & M & P & Q | 83,679 | 3,345 | 4.0% |
| 113986015 | Majkel1337 | 86,961 | Russell Kirk | 83,993 | 2,968 | 3.5% |
| 113970426 | Boey | 86,882 | DECEM | 84,522 | 2,360 | 2.8% |
| 113975260 | KawattaTaido | 85,588 | We wanna be tomatos | 77,879 | 7,709 | 9.9% |
| 113955678 | Pii | 83,694 | 🐚seek inspiration🐚 | 81,963 | 1,731 | 2.1% |
| 113963243 | Fourth Quadrant | 83,580 | Just A game on your lips | 79,504 | 4,076 | 5.1% |
| 113985944 | DSM | 82,835 | Boey | 77,652 | 5,183 | 6.7% |
| 113973892 | Fourth Quadrant | 82,304 | Anton Tikhonov | 76,174 | 6,130 | 8.0% |
| 113978086 | Majkel1337 | 81,429 | Vadim Vasilenko | 80,480 | 949 | 1.2% |
| 113984481 | Fourth Quadrant | 80,491 | We wanna be tomatos | 78,956 | 1,535 | 1.9% |
| 113978087 | DSM | 79,152 | Boey | 78,286 | 866 | 1.1% |
| 113967319 | KawattaTaido | 77,913 | Anton Tikhonov | 75,406 | 2,507 | 3.3% |
| 113978093 | Anton Tikhonov | 72,613 | Fourth Quadrant | 70,441 | 2,172 | 3.1% |
| 113984727 | Anton Tikhonov | 72,130 | Azat Akhtyamov | 71,686 | 444 | 0.6% |
| 113970423 | Vadim Vasilenko | 71,501 | M & M & P & Q | 71,028 | 473 | 0.7% |

**ours episodes (10)**

| episode | winner | winner $ | loser | loser $ | margin | margin % |
|---|---|---:|---|---:|---:|---:|
| 113991460 | Márcio Santos | 153,605 | Fried Chicken Lovers | 151,326 | 2,279 | 1.5% |
| 113978185 | Humanitis | 139,822 | Fried Chicken Lovers | 107,934 | 31,888 | 29.5% |
| 113975639 | dauriel | 129,828 | Fried Chicken Lovers | 121,717 | 8,111 | 6.7% |
| 113976876 | Fried Chicken Lovers | 123,056 | Liyi xin jie | 122,896 | 160 | 0.1% |
| 113983835 | maoxin123 | 113,484 | Fried Chicken Lovers | 112,096 | 1,388 | 1.2% |
| 113992580 | MCZK | 108,591 | Fried Chicken Lovers | 108,274 | 317 | 0.3% |
| 113981679 | KKY | 100,912 | Fried Chicken Lovers | 99,924 | 988 | 1.0% |
| 113982698 | Hamna Kaleem | 97,644 | Fried Chicken Lovers | 94,554 | 3,090 | 3.3% |
| 113984717 | Fried Chicken Lovers | 80,585 | Chahat Mehra | 80,117 | 468 | 0.6% |
| 113983534 | WUZB | 58,274 | Fried Chicken Lovers | 57,881 | 393 | 0.7% |
