# Public Kaggriculture notebooks: agents, strategies, lineage

Collected 2026-09-27 from the top ~75 notebooks by public score (the kernel list sorted by `scoreDescending`), plus
highly voted and recently run (>= 2026-09-15) notebooks, plus the lineage roots (yhay81 Shop Router 0909/0913 and
Rayk Kretzschmar). In total 173 notebooks were pulled. 159 produced a working agent (720 steps, DONE vs `random`),
which comes to **129 distinct `main.py` files**. The agents are in `kaggriculture/public_agents/<owner>__<slug>/main.py`,
and the full table is `public_agents/INDEX.csv`.

## How to read INDEX.csv

- `claimed_score` is the LB figure stated in the title or markdown. Many of these figures are stale: they belong to an
  older version of the agent, or the notebook is a copy of someone else's agent. The notes in that column say which.
- `score_list_rank` is the position in Kaggle's `--sort-by scoreDescending` kernel list. It is the only objective public
  score signal we have. It reflects the rating of the kernel's own submission, and many of those submissions are old.
- `margin_vs_metav4_s7p0_s11p1` is our coin margin against Thomas Tschinkel's Metav4 v13 in two games (seed 7 with the
  agent as player 0, seed 11 with it as player 1). It is only a quick sanity signal. **Use the round-robin for real strength.**
- `identical_main_py_as`: 30 of the extracted agents have byte-identical twins. Several "new" notebooks are straight
  copies. Examples:
  - haideptry "Demystifying 2900" and "Countering the Big 3" are both the 2945 Farm.
  - haideptry "2950 Peak Farm" is Pipe16.
  - guruprasaathas "TOP 2 Master Engine V4" is tetsutani Step1009.
  - guruprasaathas "Master Engine V3" is cha22.
  - guruprasaathas "Master Engine V5" is Ahmed V39.
  - degnonguidi/reyhanksatria are Ahmed V45.
  - Other twins exist; see the column.
- **Local patch.** In `leoprovorov__a-song-of-ice-and-fire-fixed-flexible` and `leoprovorov__kaggricult-man-reverse-engineering`
  (MarketShock-M1-WR1K), the published `main.py` ends with a helper (`install_water_repair_local_patch(parent)`). That
  helper is the *last callable*, so the standard loader picks it and the agent never trades: it idles at 3000 coins. I
  appended a `_collector_entry` wrapper. The original file is kept as `main_original.py`.
- **Multi-file agents** work from any working directory. The loader appends the agent directory to `sys.path`. The
  multi-file agents are:
  - mzcao7
  - nusrati
  - crystalbaby / leoprovorov god's-mode
  - hakdevelopment
  - aurax7 v2
  - yhay81 0909
  - avioon / flexonafft (C++ `agent.so`)

  **Caveat for a same-process tournament:** helper module names collide across agents (`observation.py`,
  `base_agent.py`, `policy.py`). Run each game in a fresh process, or at least never pair two of these agents in the same
  process.
- Runtime: the median is 5.4 s per 720-step game against random. The outliers are Kaito v58 (12 s), flyfarmer (33 s) and
  hesoponyo pure-RL (387 s per game, so skip it or give it its own slot).
- Not extracted (14): analysis, replay or lab notebooks with no agent. Also degnonguidi utils-v1, which needs a dataset
  and is the same as Kaggricult-Man. Reasons are in `fail_reason`.
- No public "koshinm" notebook exists in the competition kernel listings. The name does not appear in any pulled notebook
  or agent source.

## 1. The shared "meta" route: what nearly every strong public agent actually does

Every agent in the top public cluster replays the same **recorded production route** (a "tape"). The tapes originally
came from top-team replays and were packaged by yhay81 as Shop Router 0908/0909. On top of the tape sits a stack of
reactive "reflex" layers. Production is almost identical across the whole family, and the agents differ only in the
market and timing layers. In our traces every modern agent (Ahmed V48+, 2945 Farm, Metav4, Pipe16, tetsutani, cha22,
Dmitrii and others) bought:

- 3 quadrants: NE on day 6 and SW on day 11. SE ($4k) is never bought.
- 8 cows and 9 sheep. The older V25-V46 lineage bought 8 cows, 6 sheep and 3 geese.
- About 160 wheat seeds, 12 melon, 33 strawberry and 31-42 carrot.
- 266 hires in the season, with at most 11 per day.

Canonical timeline (tetsutani Step1009, seed 7, taken from a trace):

| day | purchases (non-SELL market orders) | bank at end of day |
|---|---|---|
| 0 | opening wheat trade; 5 HIRE; 2 COW + 2 SHEEP; **12 MELON seeds**; wheat seeds | ~$16 (all-in) |
| 1-5 | 3-5 hires/day; +1 cow on day 2 and day 3; wheat seeds; strawberry from day 5 | $100 to $800 |
| 6 | **BUY_LAND (NE, $1k)**; +2 cows; 8 strawberry; fertilizer | $1.1k |
| 7-9 | +2 cows (8 total); sheep start day 8; 7-8 hires | $0.5k to $2.6k |
| 10 | **melon harvest sold (≈72 melons, about +$13k)**; 11 hires | $16k |
| 11 | **BUY_LAND (SW, $2k)**; last sheep (9); 13 strawberry | $15k |
| 12-23 | wheat replanting (feed plus sales), fertilizer purchases, ~9-11 hires/day; milk, wool and strawberry income | $19k to $74k |
| 24-27 | late **carrot** plantings (short cycle that finishes before the end), heavy fertilizer | $76k to $83k |
| 28-29 | liquidation only | ~$90k |

Mechanics that the route and its layers exploit:

- **Melon on day 0.** 12 seeds are ready on day 10. This is the first big cash injection, and it funds the herd and the
  second quadrant. Cash on day 0 is so tight that one melon (~$1.2k of fruit) depends on a ~$20 margin at step 17. That
  is the target of the opening wars described below.
- **Animals give lifetime value.** CARE banks +1 unit per fed-and-cared day. A cared cow gives about 3 milk per cycle, and
  a cared sheep about 4 wool.
- **Sheep placed on day 11 get 5 wool harvests** (days 17, 20, 23, 26, 29), against 4 for a day-12 placement. This is
  VE1 in the 2945 Farm, and "v233x Early Yarn Commit" in Ahmed V50, which applies it in yarn-store towns.
- **Hires cost fib(n) per day.** Hires #12-#13 cost $144 and $233. Metav4 skips the tomato crew on days 19, 21 and 23,
  which are non-harvest days (+$501 per game in tomato towns).
- **Shed overflow at midnight destroys goods.** SHEDROOM (hours 21-23, margin 8) and tetsutani's "99-slot target" sell
  the excess before the midnight drop.
- **Idle-worker temporary wheat.** Pipe-16 / Dmitrii "Smaller Market Shock" / "One More Wheat" use idle opening
  workers to plant, water and harvest one temporary wheat crop on a tile that later becomes pasture: 2-3 free wheat.
- **Terminal handling.** A 7-turn rescue planner (Dmitrii, E182) simulates the last seven turns (712-718) to harvest and
  deliver every sellable unit and liquidate at the final step.

## 2. Market microstructure: where the edges are

The engine processes both players' market lists **slot by slot**, and same-index orders are quoted at the same
inventory. Premium goods (strawberry, milk, wool, melon) crash to the $1 floor after about 60-160 surplus units, so the
first seller takes the price. Almost every public improvement since mid-September is a market-layer trick built on this.

1. **Turn-0 wheat round trip.** Ahmed V45: `BUY 70 WHEAT` in slot 0, then `SELL 70` in slot 1. Against the market alone
   it is neutral. Against tape rivals whose second wheat buy is in slot 1, it raises their price, so they plant one melon
   fewer (about -$1.2k for them). The counter-moves:
   - Pipe-7 cut the size.
   - goodpjw2008 used 10 units plus a **step-1 squeeze**: `BUY 90` in slot 0 and `SELL 90` in slot 1, placed before the
     rival's `BUY 5` feed order in slot 1, then its own `BUY 5` in slot 2. This took ~$50 from V4x rivals before step 17.
   - tetsutani downshifts from 30 to 8 units when the public cash balances mirror each other.
   - Rayk C94 "Feed5-first" puts the feed buy in slot 0.
   - Dmitrii "BUY 5 + 1 seed" does no round trip at all.

   The current top cluster opens with one of three patterns:
   - `BUY 20 / SELL 15 / BUY_SEED 1` (Metav4 lineage)
   - `BUY 8 / SELL 3` (Dmitrii 7-Turn Rescue, tetsutani)
   - `BUY 5 + seed` (cha22 / Shepherd / Farmer John)

   Big round trips (V45's 70, sdy623's 50) did badly against the modern cluster in our check (-5k to -7k).
2. **Sale racing / front-running (RACE, RACEPX, RACEGATE, PREDICT).**
   - Rival sales are observable: `rival_sold = Δinventory + town_draw − own_sold`.
   - A **replay library** of rival sale streams predicts when the rival will dump a premium good. Metav4 rebuilt it from
     1,200 top-30 replays (222k events). Ahmed V53 added sale streams from Pipe16 and More-Wheat. Dmitrii keeps up to 3
     near-best-matching trajectories.
   - The agent then **advances already-planned sales** by up to 40 turns (41 in Ahmed V55; 42-43 regressed).
   - RACEPX/RACEGATE forbid racing into a book that already trades below base.
   - Holding premium goods for a peak loses between -$1.2k and -$3.2k.
3. **Queue hygiene / order book.** These change execution order only, not quantities.
   - lynnsakurai V2 queue closure, Ahmed V48 "Clear the Queue", tetsutani fixed-sell closure and Shepherd's Ledger "hole
     closure": drop SELL slots that cannot execute and pull executable sells into earlier slots. This was worth about
     $200 per game. Pipe-15 measured 27-3 in favour of lynnsakurai V2 against the 2945 Farm, purely from ordering.
   - shiiin9 "layer D" is an exact lockstep order book that permutes the market list to maximise modelled revenue given
     the rival queue. It beat V48 100-0.
   - The v44y lockstep SELL re-order (from turn 216) and ORDERPRI2 put exposed products in slot 1.
   - Ahmed V57 fixed a causal bug: the optimiser had moved HIRE/BUY ahead of the sales that fund them.
   - haodou092 V74 re-applies the sale-order optimiser to contiguous SELL blocks of 2-6 orders.
4. **Clone and mirror detection.** Public tile-similarity (>=90%) and cash-response probes (Ahmed r37/r44, lynnsakurai
   V5) identify a near-mirror opponent. Only then do the agents turn on same-turn premium pre-emption (Ahmed V44,
   andrewsokolovsky "slot sniper") or mirror-aware sale permutations (boatlee V13/V16 shift part of the next SELL one
   turn earlier and repay it the following turn).
5. **Shop routing.** The first two shops (revealed around steps 72 and 144) select 1 of 13 tapes (yhay81 0909). The
   2965 Master Engine has 41 routes (13 classic + 28 EXP240). Kaito v58 adds checkpoints at 72, 96, 144 and 360.
   tetsutani Shape-the-Shop has checkpoints at 226, 360 and 433 with prefix-compatible tails.
6. **Frontier (not solved in public).** Metav4's analysis shows that top-10 private teams **abandon the tape around days
   12-18** and use adaptive planners. That is worth +$5k to +$7k per game against the tape. haideptry notes that
   private top agents run about 10 tomato tiles from day 12. leoprovorov "God's Mode" tries to steer the hidden-seed
   shop draws through RNG consumption: interventions changed the next shop 76.6% of the time, but no proven score gain.
7. **Evaluation noise.** jaxa623/sdy623 had the same bytes under two entries and they settled 90 rating points apart
   (2857 vs 2786). The ladder noise floor is therefore about 90 points, and most posted A/B deltas are below it.

## 3. Lineage family tree

```
aurax7 Reactive Router (sale timing, shed projection)  --->  yhay81 Shop Router 0908 / 0909 (13 tapes, step-144 router, DIG repair)
                                                                    |                         \
                                                                    |                          nusrati 2715.6 (0908 mosaic), mzcao7 LGBM, crystalbaby 13-route
Thomas Tschinkel Public State Router --+                            v
                                       +--> Ahmed Berat Özer V25 (Shop0908 base) -> v28 -> v31 -> v34 -> V35 -> V36 -> V37 -> V38 -> V39 -> V41 -> V43
                                              (+prvsiyan V219 tomato / V233 sheep, Dmitrii stock reservations & terminal rescue, tetsutani room/repair, lucifer19)
                                                   |                                                                |
                                                   |   V43 -> V44 (clone-gated same-turn pre-emption) -> V45 (turn-0 70-wheat round trip) -> V46 -> V47 (+seyitkaangunes layers) -> V48 (clear queue)
                                                   |     |  \-> Nathan Jacob Pipe-4 (C9 opening), Pipe-5, Pipe-7 (small round trip), Pipe-8;  sdy623 2780/2802;  seyitkaangunes 2820;  goodpjw2008 squeeze 2749
                                                   |     |  \-> alperen (V46/V48 sale advance), shiiin9 "Beat V48" (layer D + O-B), zihengedie, hosen42 (V47)
                                                   |     V48 -> V49 (+2945 Farm econ layers) -> V50 (day-11 yarn commit) -> V51/V52 (lean flock) -> V53 (opening signature)
                                                   |
                                                   +--> Thomas Tschinkel v9/4 "The 2945 Farm" (V39/V40 + RACE/PREDICT/COURIER/CARROT/HERD + V100.24 grafts) [2944.7]
                                                            |-> Dmitrii "A Smaller Market Shock" (idle-worker temp wheat)
                                                            |-> Nathan Jacob Pipe-15 (HybridOpening + lynnsakurai queue closure)
                                                            +-> Thomas "Metav4 v13" (tomato-crew skip, CARROT2, fert d14, SHEDROOM, PREDICT from 1,200 replays)
                                                                   |-> Nathan Jacob Pipe-16 (idle workers) -> Pipe-18 (six layers)
                                                                   |      |-> Dmitrii "One More Wheat" (3 temp wheat) ;  haideptry "2950 Peak" (= Pipe16)
                                                                   |      +-> Ahmed V54 (productive wheat & patient sale) -> V55 (41-turn horizon) -> V56 (seed/fert caps) -> V57 (funding order)
                                                                   |                 +-> shiiin9 "Order Book" (V55 + exact order book) -> arsgorynich Order Book v3
                                                                   |                         +-> Dmitrii "More Wheat, Smarter Sales" -> "7-Turn Rescue" (= statma Herd-Safe) -> arsgorynich/leoprovorov Herd-Safe v3 (4-turn forecast), arsgorynich V40 Challenger
                                                                   |                         +-> leoprovorov MarketShock-M1-WR1K (Seven-Turn Rescue base)
                                                                   +-> tetsutani Step100x "Demand-Preserving Turn Sale Timing" (fixed-sell closure passes)  [= guruprasaathas "TOP 2 v4"]
                                                                          +-> haideptry "Shepherd's Ledger" (+ hole closure) = flexonafft Multi-Route;  cha22 (abhinav0370) = guruprasaathas Master Engine V3;  lynnsakurai Farmer John (wheat seller)
                                                                          +-> haideptry "2965 Master Hybrid Engine" (41 routes) -> prvsiyan "Moon Counts Melons" / "Soil Remembers Rain" (final SELL-block reorder)
                                                                          +-> hanifnoerrofiq Pioneers (More Wheat controller + demand-preserving sequencing), wzhengbiao v15stack / hybu, yasutakababa late-purchase

Independent / older route families (Aug; 10 cows + 4 sheep, 20 melons; open with 5 hires + animals on step 0; lose ~30-50k to the modern cluster):
  Kaito Fukami v20/v21/v48/v58  -> web3cainiao v21 (2837.3), ravi123 v21.1, hboyang KiteX, llccqq624 relay/ensemble
  E283 "BL-MDgogo-10C4S" public-replay consensus -> reyhanksatria best-market (B0, "3025.0"), foysal B1 (+Rayk C94 Feed5-first)
  boatlee V13-R3 / V16-RC2 / V29-R1 (clean-room); Rayk Kretzschmar C-series (C92 2836.8); salemali7 HarvestForge-X; stevenleehans X567
  avioon Apex V7 (C++ .so, Three-Day Shop Router) -> flexonafft Adaptive Farm Intelligence;  tetsutani Shape-the-Shop (Sep 6); y3uanm router
  xuantianfengwu (CN) capacity-release / day-3 cow / terminal logistics; hesoponyo BC+PPO transformer; djamilabenchikh GNN+DQN on boatlee V16
```

## 4. The ~20 strongest or most distinct agents

For each agent: the directory, the claimed score and score-list rank where known, our quick margin against Metav4
(sum of two games), and the idea it adds.

1. **cha22 agent** (`abhinav0370__cha22-agent`). Byte-identical to `guruprasaathas111__kaggriculture-master-engine-v3`.
   `lynnsakurai__farmer-john-and-the-wheat-seller` and `haideptry__the-shepherds-ledger-herd-safe-sovereign` played
   identical games in our check.
   - Local margin against Metav4: **+4.1k**, the best in our check.
   - A route replayer on the 2945/tetsutani chassis, with the modern 8-cow/9-sheep route.
   - Opens with `BUY 5 WHEAT + BUY_SEED WHEAT 1`: no round trip, so it is immune to the squeeze wars.
   - Adds demand-preserving sale timing and SELL "hole closure".
   - The author claims 1,337-31 (97.7%) against 57 public agents.
2. **prvsiyan Frontier "Moon Counts Melons"** (`prvsiyan__kaggriculture-frontier-the-moon-counts-melons`, rank 18,
   +3.7k; "Soil Remembers Rain" is the same file).
   - haideptry's 41-route Master Engine, plus a final wrapper that reorders contiguous SELL blocks after all other
     layers.
   - prvsiyan also contributed V219 (tomato project), V226A (feed repair that forecasts wheat pickups) and V233
     (financed 6-sheep SE paddock).
3. **Dmitrii "More Wheat, Smarter Sales"** (`dmitriigluzdov__kaggriculture-more-wheat-smarter-sales`, +3.5k;
   `arsgorynich__kaggriculture-v40-challenger` adds a weed-build repair).
   - Built on shiiin9's Order Book, V55, V56 and Metav4.
   - The rival-sale predictor keeps up to 3 historical trajectories within 1 point of the best match. If any of them
     predicts a big milk, wool or strawberry sale within 2 turns, the agent advances its own planned sale.
4. **Pioneers of Kaggle Town (checkpoint 1 and candidate 2)** (`hanifnoerrofiq__*`, +3.5k).
   - The More-Wheat controller with demand-preserving (tetsutani) market-order sequencing.
   - The Pipe-16 HybridOpening temporary-wheat workers.
5. **Herd-Safe v3, four-turn forecast** (`arsgorynich__herd-safe-v3-experimental-risk-aware-feed`, same file as
   leoprovorov four-turn-forecast, +3.5k).
   - Dmitrii's Herd-Safe Sale Window v2, with the rival-sale predictor looking 4 turns ahead instead of 2.
   - The extra horizon is used only when at least 3 matching past events exist and precision over the last 240 turns is
     at least 70%.
6. **tetsutani "Demand-Preserving Turn Sale Timing", Step1009** (`tetsutani__demand-preserving-turn-sale-timing`,
   rank 33, 98 votes, +3.2k; copied verbatim as "TOP 2 Master Engine V4").
   - The most-iterated public "search head": repeated deterministic fixed-SELL reorder closure passes on the production
     chassis.
   - Opens with `BUY 8 / SELL 3`; an earlier version used the "22-wheat shock".
   - Earlier tetsutani releases: "Market-Smart Farming" (30-wheat opening pressure, down to 8 when public cash mirrors)
     and "Shape the Shop" (route replay with public-state checkpoints and a 99-item shed target).
7. **haideptry "2965 Master Hybrid Engine"** (`haideptry__the-2965-master-hybrid-engine`, title claims 2965, +2.9k).
   - 41 routes: 13 classic plus 28 EXP240 shop-specific routes triggered at step 144.
   - R42 step-0 opening guard, Gluzdov 7-turn closeout, V44Y sell-block optimiser.
   - Buys 8 cows + 6 sheep + 3 geese in our trace (a goose variant). Earned the most own-coins of any agent in that game
     ($98k).
8. **wzhengbiao v15stack / hybu** (`wzhengbiao__kaggriculture-v15stack-submit`, 62 votes, +3.0k).
   - A stack of Pipe-16 HybridOpening, the 2945 RACE layers and the Ahmed chassis, assembled by a script.
9. **Dmitrii "7-Turn Rescue"** (`dmitriigluzdov__kaggriculture-7-turn-rescue-historical-lb-2800`; the same file as
   `statma__kaggriculture-herd-safe-sale-window-submit`, rank 23; +2.4k).
   - Opening changed to `BUY 8 / SELL 3`, which protects early hire and feed cash.
   - Terminal planner for turns 712-718: 128 simulations, one pass, up to 8 proposals per worker; it harvests and
     delivers extra stock only when the result is physically dominant.
   - The statma race-ca20/ca25 variants tweak parameters of this file.
10. **Ahmed V53 "Opening Signature"** (`ahmedberatozer__kaggriculture-v53-opening-signature`, **rank 4**, +2.2k).
    - V52 plus an opening classifier (identifies the rival family from its step-0/1 orders).
    - Adds executed premium-sale streams from Pipe16 and More-Wheat to the PREDICT library.
    - V51/V52 "Lean Flock" (+1.9k/+1.7k) use single-hand sheep service on non-harvest days.
    - V50 "Early Yarn Commit" commits a day-11 six-sheep flock when there are 2 or more yarn stores (the 5th wool
      harvest).
11. **shiiin9 "Your Market List Is an Order Book"** (`shiiin9__your-market-list-is-an-order-book`; also
    `degnonguidi__best-agent-ranking`, and arsgorynich v3 is a derivative; +1.8k).
    - V55 byte-for-byte, plus an exact lockstep order-book model of the 10-slot market list (layer D).
    - Adds a priced gate on the late tomato investment and 4 re-measured constants.
    - The author reports 280-0 against 7 new public agents.
12. **Ahmed V55/V56/V57** (`ahmedberatozer__kaggriculture-v55-one-turn-market-race-edge`, **rank 5**; V56 and V57 have
    no rank).
    - V55 sets the premium-sale reservation horizon to 41 turns.
    - V56 adds a late seed budget (no seeds beyond the remaining planting opportunities) and a harvest-aware fertilizer
      cap.
    - V57 keeps purchases behind the sales that fund them.
    - Locally they sit around +0.9k to +1.4k against Metav4.
13. **Thomas Tschinkel "The Metav4 Farm" v13** (`thomastschinkel__the-metav4-farm-submission-v13`). This is the reference
    opponent in our margin column.
    - The best-documented chassis: see §1 and §2 for its layers.
    - The author claims 40-0 against the 2945 Farm.
    - Nearly every top public agent now builds on it.
14. **Nathan Jacob Pipe-16 / Pipe-18** (`nathanjacob__kaggriculture-pipe16-idle-workers`,
    `nathanjacob__kaggriculture-pipe18-six-layers`).
    - Pipe-16 is Metav4 plus idle opening workers growing temporary wheat: 50-0 against Metav4, but only +$118 per game.
    - Pipe-18 stacks six micro-layers: crop longevity, price guard, race horizon, E402 seed cap, E410 fertilizer guard and
      queue compaction.
15. **Thomas Tschinkel "The 2945 Farm" v9/4** (`thomastschinkel__the-2945-farm-96-vs-the-top-10-public-bots`,
    **2944.7**, 154 votes). Four other notebooks are byte copies.
    - Introduced RACE (reservation from step 192, horizon 40, margin 12), RACEPX, PREDICT, COURIER/OVERFLOW,
      CARROT (wheat to carrot swaps), HERD/COWSWAP, CAPHARV and SL2/VE1/VT1.
    - The best explainer of the reflex architecture.
16. **goodpjw2008 "Melon Threshold Squeeze"** (`goodpjw2008__kaggriculture-melon-threshold-squeeze-2749`, **2749**).
    - V45 with a 10-unit round trip plus the step-1 squeeze (§2.1) and the sdy623 sale-race layer.
    - Aimed at the V4x family specifically. It loses about 2.3k to Metav4 in our check.
17. **leoprovorov MarketShock-M1-WR1K** (`leoprovorov__a-song-of-ice-and-fire-fixed-flexible`, rank 35; Kaggricult-Man is
    **rank 8**). **The entrypoint is patched locally.**
    - Seven-Turn Rescue / Shop0909 lineage with one bounded market-timing intervention and a water-repair patch.
    - "Ice and Fire" is the best replay-level analysis of which top-agent actions are fixed and which are reactive.
18. **Ahmed V38/V39/V41** (**ranks 1, 2, 6**; `ahmedberatozer__kaggriculture-v38-smarter-feed-stronger-margins` etc.).
    - The older V-series chassis: 8 cows, 6 sheep, 3 geese, and a `BUY 13 + BUY 30 / SELL 30` opening.
    - It still holds the best public kernel scores because those submissions are old and rated early. Against the
      modern cluster it loses about 7-8k (two games).
19. **Older independent route families.** These are weak against the modern cluster, but they are worth including as
    diverse opponents:
    - Kaito v58 (`kaitofukami__238-...`): checkpoint best-response.
    - web3cainiao v21 (2837.3).
    - boatlee V13/V16/V29: order-safe premium controller and hysteresis.
    - E283 "BL-MDgogo-10C4S" (`reyhanksatria__best-market-agent-high-strategy`, and the foysal B1 variant said to derive
      from a "3025.0" agent). E283 uses 10 cows, 4 sheep and 20 melons. The E283 agents lost about 38-41k per game to
      Metav4 in our two-game check, and the others in this group lost about 18-60k. Their LB claims come from August.
20. **Outliers.**
    - hesoponyo BC+PPO transformer (slow, 387 s per game).
    - mzcao7 LightGBM/XGBoost over yhay81 routes (2476.8).
    - xuantianfengwu CN family (3 lands, a 4-6 cow opening, R10 melon sale race).
    - avioon Apex V7 (C++ .so).

## 5. Claimed ideas that look like real edges (ranked by how directly they transfer)

1. **Never bleed on turn 0-1.** Use a small or zero wheat round trip (`BUY 5-8`, optionally `SELL 3`) and put the feed
   buy in slot 0. This makes the agent immune to V45-style round trips and the step-1 squeeze. If the opponent is a
   funded-plan tape (V4x), a squeeze that costs them a day-0 melon is worth about $1.2k.
2. **Order-book-exact market lists:**
   - remove dead SELLs;
   - put exposed premium products first;
   - permute SELLs against the modelled rival queue (shiiin9 layer D);
   - never move purchases ahead of the sales that fund them (V57).
3. **Predictive front-running of premium sales** with a replay library of rival sale streams. Keep the horizon at 40-41
   turns, gate it on precision (Herd-Safe v3), and never race into a book that is already below base (RACEPX).
4. **Labour economics:** skip marginal hires on days with no harvest (Fibonacci pricing); use a single hand for sheep
   service on non-harvest days; skip care and feed on days 28-29.
5. **Day-11 sheep placement in yarn towns** gives a 5th wool harvest.
6. **Idle-worker temporary wheat** in the opening gives +2 to +3 wheat for free.
7. **Terminal 7-turn planner** and shed-overflow guard at hours 21-23.
8. **Unclaimed upside (per Metav4 / haideptry):** leave the tape around days 12-18 and switch to an adaptive planner, for
   example about 10 tomato tiles from day 12 and shop-demand-matched seeding. This is where the private top-10 gets +$5k
   to +$7k. No public agent does it.
