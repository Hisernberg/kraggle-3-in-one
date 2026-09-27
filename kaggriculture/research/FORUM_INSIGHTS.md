# Kaggriculture forum insights (all 215 topics, full comment trees, scraped 2026-09-27)

Raw dumps: `research/forum_raw/<topic_id>.json` (topic body + nested comments with authors) and `_topics_index.json`.
References are `#<topic_id>` (URL: kaggle.com/competitions/kaggriculture/discussion/<id>). Engine = kaggle-environments 1.32.7
(the installed version is the latest on PyPI; the hosts say the servers always run the latest published version with no overrides, #734562).
"[code]" marks a fact I checked against `kaggriculture.py`, not only a forum claim.

Key dates: submission deadline **Sep 30**. After it, games keep running **Oct 1 to about Oct 15**, then a single Bradley-Terry (BT) fit decides the final ranking.
Today is Sep 27, so only about 3 days are left for changes.

---------------------------------------------------------------------------------------------------

## 1. Engine facts, quirks and bugs found by participants

### 1.1 Turn and step structure
- **An agent acts 719 times (steps 0..718), not 720.** DONE fires when `step >= episodeSteps-2`. The state after step 718 is scored [code] (#742943, #737480, #732450).
  The end-of-day for day 29 never runs, because step 718 is day 29, hour 22. Anything still carried by a unit, or sitting in the shed, at the end is worth $0.
  The last SELL that counts is the one at step 718.
- **Order inside a step** [code]:
  1. All unit actions run: the farmer first, then hands in index order, then the other player.
  2. Market queues run.
  3. Town consumption runs.
  4. Plants decay.
  5. At hour 23, the end-of-day runs.
  Consequences:
  - A DROP and a SELL of the same goods can happen in the same turn, because the DROP runs first.
  - Sale proceeds cannot fund a PLANT in the same turn (#742329).
  - A seed bought with BUY_SEED is plantable only from the next turn. A bought animal lands in the shed, so it needs PICKUP, then a move, then PLACE.
- **A new hire acts from the next turn.** Hands are hired through the market, which runs after unit actions.
  The hands list is always empty at hour 0 (hands vanish overnight), so planning at hour 0 with `len(hands)` counts only 1 unit.
  This "hour-0 trap" hit one competitor 8 times (#742246). Top agents average about 7.9 units at hour 1 and 9.3 at hour 12.
- **`obs["step"]` is missing for seat 1 in serialized `env.steps` and in replay JSON.** Live `env.run()` calls receive it correctly (#737545).
  Always derive the turn as `t = day*24 + hour`. Replay tapes store the action for turn t at `steps[t+1][seat]["action"]`, which is an easy off-by-one (#737480, #740437).
- **The main function is the last callable in main.py.** Its argument count decides between `(obs)` and `(obs, config)` (#733535).
  A file reference or import failure crashes silently on the evaluator. One competitor lost about 1800 rating to this (#740847).
  Timing: 1 s per turn plus a 60 s overage bank; the first turn's load time counts against the bank.
  Both agents run in parallel, so the 1200 s runTimeout is safe (#739874, #737801).
  One competitor measured a deepcopy on the first call at 292-423 ms (#742856).
  An agent that throws on every turn usually returns PASS and ends with a bank near $3,000. Treat such a result as a crash, not an outcome (#742856).
- **Episodes are deterministic.** A seed plus both agents' actions reproduce a game bit-exactly, locally and on the ladder.
  The validation game is self-play on seed 0 (#742856, #731152).
  `competitions.EpisodeService/GetEpisode` returns the seed, so you can replay your own ladder games exactly.
  Downloaded replays carry `info.seed`; `configuration.seed` is often null (#739972).

### 1.2 Market order processing: the main place players interact
- **Up to 10 orders per turn; extras are silently dropped** (a BUY_LAND at position 11+ never executes, #742246).
- **Orders pair by list index across the two players** [code] (#742943, #739972):
  - For each index i, a HIRE or BUY_LAND is handled atomically first, player 0 before player 1.
  - Then both players' SELL/BUY orders at index i run **one unit at a time in lockstep**. Both units are quoted from the same pre-commit inventory, then both commit.
  - When one side's order at index i ends, the other side continues alone at index i.
  - Only after both index-i orders finish does index i+1 start.
  - **Consequences:**
    - Seat order does not matter (#731152).
    - Within a turn, **putting your contested sale at a lower index than the opponent's sale of the same good gets you the better prices**. Rank sell slots by price impact, most contested first (#734637).
    - HIRE orders at low indices push your sells to later indices.
    - Empty `[]` entries still count as positions (#742943).
    - A player whose SELL ran out of stock stops that order; later orders keep their positions.
- **Sell quotes use pre-sell inventory; buy quotes use post-buy inventory (inv−1)**, so a buy immediately followed by a sell nets zero.
  A sale at the $1 floor does not add inventory, so the floor is sticky and pays nothing (#732655).
  Only WHEAT and FERTILIZER can be bought.
  Buys, including BUY_ANIMAL, fail when the shed is full (≥100 non-seed items, animals included) [code].
- **Price** = `int(round(base ± amp·f(|inv−I0|)))`, floored at $1. Python `round` rounds halves to even [code].
  The formula matches the spec's anchor prices for all 9 goods (#741891, #742025).
  Premium goods fall off a cliff. Net units sold above I0 before the $1 floor [code]:

  | good | units to floor | notes |
  |---|---|---|
  | wool | 59 (sq) | price 177 at +20, 55 at +50 |
  | strawberry | 62 | 82 at +20 |
  | milk | 76 | 118 at +20 |
  | melon | 158 | 246 at +20 |
  | fertilizer | 493 | |
  | tomato | 529 | |
  | carrot | 842 | |
  | wheat, egg | never (log curve) | still about 76-78% of base at +2T |

  Dumping T units in one go of a premium good earns only about 1/3 of qty×quote, because 38-46% of the units go at $1 (#739972).
  The "never dump more than 20 premium units in one order" rule of thumb comes from this (#737568).
- **Market orders run before town consumption.** Consumption happens when `step%4==0` (shops) and `step%24==0` (town center, including step 0) [code].
  Selling at step ≡ 1 (mod 4) gets the freshly drained inventory, worth +0.2-1.4% per unit (#732655).
- **Town demand** (1.32.6+) [code] (#733431, #734412):
  - The town center eats 1 of every product (not fertilizer) per day, a flat rate.
  - Each shop instance eats 1 of each of its products every 4 turns; single-product shops (Yarn, Pet Cafe) eat 2x.
  - Shops unlock at the end of days 2, 5, 8, …, 23: 8 instances, drawn uniformly **with replacement**. That gives 132 shop-instance-days per season.
  - Expected season drain: wheat 525, strawberry 426, carrot 327, milk 327, tomato 228, egg 228, wool 228, **melon 30 (town center only; no shop buys melon)**, fertilizer **0**.
  - Medians: wheat 504, strawberry 396, milk 288, carrot 270, tomato/wool/egg 180.
  - **Wool has no buyer in 34% of seasons** (no Yarn Store). Carrot demand is 0 in 10% of seasons and about 684 at p95 (#734412 Georgy Mamarin, #738075).
  - The first shop is fixed by the end of step 71 and the second by step 143 (#742943).
  - The README once said demand "escalates 2x after day 10, 4x after day 20". That code no longer exists (#740847).
- **"Selling into the hole"**: the town drains inventory below I0, and sales into that scarcity pay far more.
  Season revenue sold into the hole vs dumped at I0: strawberry $100k vs $4k, milk $87k vs $6.4k, wool $54k vs $8k, melon $8.2k vs $7.4k (#734412).
  The hole is shared: selling harder costs you about $4k and costs the opponent about $11.8k (destbreso).
  About the $703k total per episode at median demand is theoretically available to both players combined; the best recorded game captured 48% of it.
- **The shop draw shares an RNG stream with weed spawning** [code] (#732613 yhay81, #737663, #740792).
  At end-of-day the engine does:
  1. `rng = Random((seed*1_000_003) ^ day)`.
  2. Weed rolls for player 0: one `rng.random()` per empty unlocked tile.
  3. Weed rolls for player 1.
  4. The shop draw.

  So **the town you get depends on both farms' empty-tile counts that evening.**
  - A fixed seed is a valid control only if your change does not alter tile occupancy.
  - When you drop your agent into a recorded game, the town usually re-rolls, and the opponent's tape then plays a world it never saw.
    One competitor measured 78% vs top-10 tapes in re-rolled games, against 40% when the town was preserved; only the second number is real (#740792).
  - You can recover the seed from weed and shop observations, but it takes 9-24 days of data and the opponent influences the draw anyway, so it is not useful (#739084, #739388).
    The hosts say the seed is not meant to be available to agents (#734743).
  - The hosts did not act on the proposal to split the RNG.
- **1.32.7 hinge change** (#735311, #734412):
  - Carrot, tomato and egg spike when town scarcity passes T. Firing rates: tomato 50-55%, carrot 26-28%, egg 22-26% of games.
  - **The unannounced part:** carrot's below_target moved from 0.2 to 1.0. That change, not the hinge, is what lifted carrot at median demand (about $41 → $60).
  - After the patch: carrot max-price median was 57 (p99 360); tomato p99 was 786; egg p99 was 167, so egg is barely affected. Melon is unchanged at max about 272.
  - The field moved to carrots within a day (6% → 44% of seats); nobody picked up tomato.

### 1.3 Crops
- **A seed must be watered on its planting day or it becomes a weed that night.** New plants start at `consecutive_unwatered=1`.
  After that a plant dies after 2 missed days in a row, so the minimum is watering every other day (#732450).
- **One-time crops** get +1 per watered day inside the bonus window [code]:
  - The window runs from age `ceil(max_yield_day/2)` to max_yield_day. Fertilizer makes it +2 per day.
  - Maximum yield: wheat 4, or 6 with fertilizer; carrot 3, or 4 with fertilizer.
  - Melon reaches 6 at age 10, or at age 8 with fertilizer. However, **HARVEST is only allowed at age ≥ first_yield_day (melon 10)**, so fertilizer does not make melon faster.
  - Decay starts at step `(planted_day+max_yield_day+1)*24` and removes 1 unit every other turn until the tile becomes a weed. Melon must be harvested by about day planted+13.
- **Ongoing crops** [code]:
  - Tomato produces at the end of days p+7..p+10; strawberry at the end of p+9, p+11, p+13, p+15. That is 4 productions each, then decay and a weed.
  - Each production gives +1, or **+2 if the plant is fertilized AND watered that day**.
  - Held units are capped at 4, so harvest often.
  - One fertilizer lasts days d..d+2, so on strawberry it covers two productions. **With 2 fertilizers a strawberry gives 8 units instead of 4** (derived from code).
    The forum only hints at this: yield-aware FERTILIZE only helps if a production falls within day..day+2 (#740847). Top agents' fertilizer use is consistent with it.
- **The PLANT check is per crop and all-or-nothing.** If units issue more PLANT orders for a crop than you have seeds, **none** are planted (#741907).
- DIG removes plants, weeds and empty coops or pastures only, not an occupied one (#732450). BUILD_* is free but needs an empty owned tile.
- Weeds spawn with probability 0.005 per empty unlocked tile per day.

### 1.4 Animals, CARE and fertilizer
- **FEED takes 1 wheat from the acting unit's own inventory, not from the shed** (#737731, #741907). Units must carry wheat.
  An animal escapes after 2 unfed days in a row; a newly placed animal survives its first day unfed.
  Missing FEED is the most common costly tape bug: 2 cows dead on day 1 cost $12-32k (#739273).
- **CARE** [code] (#732820, #734033, #740847):
  - At end of day, if the animal was fed AND cared that day, `pending_care_bonus` rises by 1. Caring an unfed animal does nothing.
  - On a production day, if the animal is fed, the bonus is added to that production.
  - Daily feed+care gives a steady state of 1+interval units per production:
    - sheep: 4 wool every 3 days (4.25x lifetime)
    - cow: 3 milk every 2 days (3.3x)
    - goose: 2 eggs per day (2.1x)
  - The first sheep production can reach 6 (max_held).
  - Everyone at the top does this (one public agent logs about 321 CARE actions).
- **Fertilizer**: each surviving animal makes 1 available per day, whether or not it was fed or cared.
  The flag is a boolean and does not accumulate. Fertilizer is sellable (#731953).
  - No town demand exists, so fertilizer only ever sells on the glut side and its price only falls (100 → about 1-20 by the end of top games, #737736). **Sell it immediately or use it.**
  - The fertilizer stream is worth roughly $2.9k per animal per season, more than the animal's named product in several analyses (#734412, #741907).
- **Placing an animal takes 4 steps**: BUY_ANIMAL, then PICKUP at a shed tile, then walk to an empty matching structure, then PLACE (#733496).

### 1.5 Units, land and shed
- **The shed is not a tile.** Its access tiles are (4,4), (5,4), (4,5) and (5,5). PICKUP and DROP work there even while the tile is locked.
  Locked tiles can be walked through; other tile actions no-op on them. The hand-stuck-on-SE bug was fixed Aug 3 (#731635, #732450).
- **Hires** cost fib(n) and reset daily. Cumulative cost per day [code]:

  | hands | cost per day |
  |---|---|
  | 8 | $54 |
  | 10 | $143 |
  | 11 | $232 |
  | 12 | $376 |
  | 13 | $609 |
  | 14 | $986 |

  - Hands spawn on the least-occupied shed tile (NWSE tiebreak) and vanish at end of day, dropping their inventory into the shed.
  - The farmer teleports back to (4,4) every morning. This "walking toll" is about 15 percentage points of unit-turns (#741907).
  - Public meta, about Sep 11: 8 cows + 4 sheep, 62 plants, about 14 hands, 264 hires per game, 0 deaths (#740847).
- **Walking, not labour, is the binding constraint.**
  - One agent spent 83% of unit-turns moving. Finishing all work on the current tile (FEED/CARE/HARVEST/COLLECT all run from the same square) before moving cut that to 55% and **tripled** its bank (#734412).
  - Giving each hand its own slice of tiles beats a shared job list by about $4.5k (#742449).
  - Tiles left bare at nightfall track the bank better than tiles per worker do.
- **Land** is bought in the fixed order NE $1k, SW $2k, SE $4k, so reaching SE costs $7k in total (#742449).
  Most agents used 3 quadrants early on. In late September the #1 on the leaderboard used SE under some conditions (#742449, #734308).
- **Shed cap is 100 items.** End-of-day overflow is destroyed. Stockpiling to wait out a price crash loses money (−$1.4k measured twice, #732655).

### 1.6 Display bugs (not engine bugs)
- Several "I had more coins but lost" reports were replay-viewer bugs showing the wrong match. The backend and the JSON were correct (#734684, #739699, #742341).
- The live Elo can lose one of two concurrent updates (a race). The final BT fit counts both games (#742165).
- The "Play in browser" demo differs from the engine: it blocks moves onto locked tiles and has no DROP (#733902).
- Kaggle notebook images ship kaggle-environments **1.29.3**, where fertilizer cannot be sold and the prices are old. **Pin 1.32.7** (#740650, #737459).

---------------------------------------------------------------------------------------------------

## 2. Strategies: what top players describe, and what works

### 2.1 Meta history
1. **Jul 30 to Aug 6.** One public notebook dominated: turns 1-51 were byte-identical across the top 13 instances, "81% win rate, ignores prices" (#732902, #733055).
   Early meta details: "plant 44 strawberries", LAST_CARE_DAY=27, liquidation on day 29. Seb's build was a standout.
2. **Aug 6 rebalance.** Town-center demand was cut and shops became draws with replacement. After that, the Kaito Fukami public lineage and its forks dominated (#735683).
3. **Late Aug to Sep: "tape" era.** Teams replay recorded 720-turn action lists from strong public episodes, with small market tweaks.
   - About 85% of the top plays tapes (#739273).
   - In one competitor's measurement, 96% of ladder opponents match a known tape on more than 60% of farm turns (#740437).
   - Public notebooks match each other on 96.5% of unit actions (#742329).
   - A new economy gets cloned by about 15 teams within 24 h (#737955), in waves that fade within days.
   - A round-robin of 14 public implementations on 96 seeds was nearly transitive: the newer beat the older in 86/91 pairs, and **the ordering tracks average final money** (#739273).
4. **Sep: routers and RL.**
   - "Tape + mechanism" agents (shop-router-0909, fieldbook) fork the route at t=72-144 based on the shop draw (#739179, #738727).
   - Genuinely adaptive or RL agents reached the top:
     - The Sep 1 "king" diverged from turn 34, went 40-0 with a +$13k median margin, then was withdrawn (#739179, #738563).
     - SpaTaro played unique lines in every game (#739273).
     - Sayaka Miki (#2 around Sep 17) uses RL (#741792).
     - Snorlax reached the gold zone with BC warm-up plus macro-level PPO over about 300k games (#741743).
   - Strong teams pull or hide their best agents to avoid being cloned (#739179, #743384).
   - **Non-transitivity appears among the strongest agents**:
     - B85 > Andrews2883 > Kaito v35 > B85 (#736439).
     - v7 shop-router beats hybrid2965 12/12 but loses to V53 0/12, and 0/6 to the 2945, herd-safe and wonderful-life lineages (#743231).
     - Avineesh found a 100% / 85% / 100% cycle among three strong public agents (#742856).

### 2.2 Economy: what produces money
- Reference banks (public v48-fast-routes and similar):
  - about $123-145k per game against active opponents, about $180k against a passive one (#742246)
  - strong-vs-strong games about $100k each (#743179, #742856)
  - a relaxed upper bound of about $195k per game (#742246)
  - the best fixed-vector search found only $55k, so the good executors are the hard part
- **Where the money is.** v48's gross sales per game: strawberry $53k, wool $31k, milk $31k, fertilizer $18k, melon $17k, wheat $13k, egg 0 (#742246).
  About 77% of a weak agent's gap to v48 was strawberry plus wool, a **production** gap rather than a timing gap.
- Early crop comparisons (single farmer, at base prices): melon has the best profit per action and per tile-day (#734033, #741907, #737731).
  But **no shop buys melon**, so its market is a fixed pot of about 150-200 units (#738075, #734412).
  All-melon openings go bankrupt (#737731). Wheat bootstraps cash from day 2.
- **Animals dominate per tile once CARE is included.** Cows and sheep take the first ~15 tiles in a greedy allocation (#734412).
  Get cows bought and placed on days 0-4: an 8-day first yield and 22+ producing days made this worth +$10k in one test (#737568).
  - A sheep herd's season value is about $39k with a Yarn Store and about $11k without.
  - A cow herd was the best single herd in 70% of towns (#732613).
  - Herd sweeps are sharp: 9 sheep beat 6 by +$7.3k while 10 was worse; 7 cows + 7 sheep and 0 cows + 14 sheep were much worse (#732655, early engine).
  - Geese were −$42k (early engine).
- Spend all $3,000 at turn 0; the best route spends $2,982 immediately (#737568). Idle cash compounds badly.
- Stop starting seed-to-harvest pipelines after about day 26 (+$782, 16/16 seeds). A calendar cap on late hiring hurt (−$233), so value late hires by the work they can still cash (#735119).
  Liquidate from day 28: removing sell reserves then recovered about $2.8k (#740847).

### 2.3 Selling and market play (where games between near-equal agents are decided)
- **At the top, games are decided by tiny margins.** Among opponents rated 2250+:
  - the median gap was $177 on banks of about $98k
  - 40% of games were decided by less than $100, and 78% by less than $1k (#742856)
  - one game was lost 103,148 to 103,147, costing −106 rating (#742083)
- **Sell early rather than patiently.** Against sell-on-sight opponents, trickling 3 units per turn cost $547 and handed the opponent $593 (#739273, Georgy).
  Other measurements:
  - "Selling earlier" as a global schedule shift cost −$80k, because goods had not arrived yet. The feasible sale schedule is a narrow band (#742856).
  - Splitting v7's endgame `SELL x 1000` into per-turn slices cost about −$1k (#743231).
  - Holding stock while prices are depressed cost −$2.3k (#742856).
  - The price path of top games: the premium window is roughly days 9-14. Premium goods peak then and end at 60-126; wheat, carrot and egg appreciate late (#737736).
- **Mirror matches are decided by sale timing.** Identical farms, money within a few coins through day 16, then wool sales on days 17-25 decided it.
  Wool went 226 → 31 → 1 when both dumped (#743231).
  - **Draws reveal a mirror opponent.** The standard exploit is to **advance your sales by one turn**, as long as the financing chain survives.
    Cloning teams do this systematically and it has become an arms race of counters (#740588, destbreso).
- A premium-seller layer on a fixed production plan (sell milk at ≥120, wool ≥140, strawberry ≥90, melon ≥60) worked best starting day 14 with batch 15, going 19-1 on held-out seeds.
  Too-small batches lose the timing edge; too-large batches crash the price (#737879).
- Opponent modelling:
  - Opponent inventory for carrot, tomato and egg can be inferred **exactly** from public state; milk and wool are noisy because of $1-floor sales (#737027).
  - Pinning the opponent's lineage by about step 12 was worth only about $38 per game to one agent. A cheat reading their future sells was worth about $224, which bounds the value of prediction (#740792).
  - hybrid2965 changes its opening based on the opponent's turn-0 market orders (#743231).
- Shops are extra drainage, not separate buyers. Read `town.unlocked_shops` together with inventory, prices and both players' production (#740823).

### 2.4 Things reported NOT to work
- Geese, tomato-only or carrot-only farms (pre-hinge), hoarding through crashes, and more hands beyond the need (fib cost) (#732655).
- Throttling or perturbing tapes: making an agent PASS on 20% of turns drops it from $180k to $312, because tapes are tightly coupled plan executors (#742246).
- Handing the endgame to an online controller cost −$7k (#742856). Guards copied from stronger agents that never fire are worth $0.
- BC on primitive actions: 99.4% per-frame accuracy, or DAgger at 99.67%, still gave near-zero money (#742246, #738079, #737937).
- Pure end-to-end or self-play PPO plateaus at about 20-90k (#734952, #738619, #741258, #740022).
- Self-play leagues can converge on doing nothing (#742246).
- MILP (HiGHS) inside the 1 s budget is too slow (#742005).

---------------------------------------------------------------------------------------------------

## 3. Ladder and rating dynamics
- The live rating is a mu/sigma (Elo-like) system like Orbit Wars. Outcomes are win, loss or tie only; the margin is ignored (#732708).
  - A new submission starts at 600.
  - K is about 220 flat for about 10 games, drops to 45-55 by game 20, and floors at about 8.5 by game 80 (#736219).
  - A win is worth about 100 points below 2300 and about 60 above, shrinking to under 10 (#739534).
  - **An early loss collapses K**. One submission lost its first game, won 51 straight, and still sat at about 1500 (#740606).
- **Game rate:**
  - About 15 games per hour for about 4-5 h (1-2 matches every ~4 min for the first ~80 games), then 1-3 per hour (#736219, #736314, #739534).
  - Two slow-matchmaking windows: Aug 20, while Pokemon compute was busy, and Sep 23-27, resolved Sep 27 (#743214).
  - The whole ladder plays about 140k episodes per day; the daily top-episode dump is about 0.5% of them (#736625).
- **Convergence** (811 submissions, #740282): median distance from the final level is 718 after 5 games, 338 after 10, 222 after 20 and 145 after 40.
  At about 5 h the median submission is still about 123 points away, and a quarter are more than 400 away.
- **Noise:**
  - Byte-identical copies ended 248-1400 apart (#734000, #738343).
  - A single early loss, against an under-rated fresh submission, left one copy about 300 lower (#734000).
  - A resubmitted old agent scores much lower because the field keeps improving (#736932, #737955).
  - Treat differences under about 50 points as noise, even at 200 games (#736219).
- **Top-10 matchmaking pool is narrow.** Leaders rarely meet the lower-ranked agents that could beat them (#739534).
  - An undefeated run reaches the top in about 8 h (#737955).
  - One former 6th-place agent fell to about 700th within 4 days as the field improved (#739534).
  - **Survive the climb:** an agent strong at 2250+ stalled at about 1800 because it went only 56.5% against sub-2000 opponents (#742856).
- **Seat effect is null**: 73% vs 71% (#740437), 44% vs 43% (#742856), +228±331 over 60 seeds (#742246).
  Swapping seats on the same seed gives nearly identical results, so **count seeds, not games, as samples** (#743231).
- Median average score of the daily top-episode dump has been 2,735-3,080 since Aug 4 (#731215).
- **Final evaluation rules (host statements):**
  - After Sep 30, games run for about 2 weeks; no midterm evaluation (#736187).
  - The **final BT fit uses all episodes ever played between submissions that are still active at the end**. An episode counts only if **both** agents are still active (#732931, #742571).
  - Your 2 most recent submissions are your final pair. The team is ranked by the better of the two, so **the second slot is a free hedge** (#739410).
  - **Ties count as half a win** (#739410).
  - The post-deadline play rate may rise, with no commitment (#739410). The hosts plan to write BT to the private leaderboard (#739788).
  - Game parameters will not change for the final evaluation (#737570).
  - **Contradicting forum lore:** there is no statement that ratings "reset to 600". The live rating is simply irrelevant to the final fit.
- **Submission management:**
  - Up to 5 submissions per day; only the latest 2 are active. **Each new submission retires the OLDER active one, so submit the keeper last.**
    One competitor retired his strong agent by one minute (#742856).
  - Submit final agents early enough to accumulate episodes against the final population.
  - Do not ship untested code on the last day, because an erroring agent plays nothing (#736219).
  - Re-rolling an identical bot buys only live-rating cosmetics.
  - Use 40-70 games and per-opponent-band win rates to judge a submission. The band where you win about 50% is your real level (#742856).
- **Local-evaluation lessons:**
  - Evaluating against starter, pass or random measures a different game: one agent made $121k locally vs $81k on the ladder.
  - A league of trading public agents on real ladder boards correlates with the ladder at about r=0.46-0.53 (#742856).
  - Weight lineages by how often you meet them in the target band, and add "walls": lineages you must not lose to (#743231).
  - The opponent mix by band: at 1500-1900, the a1-t31/hack family (24%) and utils-v1 (23%) dominate. At 2450-2850, a few large public lineages plus a private tail (#743231).
  - Pair everything: same seed and opponent, both seats, bootstrap CIs, per-opponent tables. Hold out a seed block; one change measured +2,301 on one block and −1,259 on another (#740792).
  - Fingerprint agents by a sha256 of the first 48 actions, against 2-3 different opponents (#743231).
- **Fast engines:**
  - Rust port (debmalyaroy/kaggriculture-simulation, byte-identical, about 550k steps/s) (#742943)
  - destbreso's C++ port, 2k-24k episodes/s (#737459)
  - nikital7's 4000x notebook (#733392)
  - kagsym, which calls the real interpreter on a duck-typed state, about 38k steps/s (#742246)
  - Datasets: kaggle/kaggriculture-episodes-index (daily top episodes), georgymamarin/kaggriculture-episodes (API crawl with an engine_version column), destbreso's community agents dataset, raykkretzschmar's reference agents (tiers 0-9).
- Episode views are limited to 3600 per 24 h; ListEpisodes can IP-block (#732114, #734135).

---------------------------------------------------------------------------------------------------

## 4. Host announcements and rule changes (chronological)

| Date (2026) | Who | What |
|---|---|---|
| Jul 31 | María Cruz #731587 | Final: games continue about 2 weeks after the deadline, then a single BT tournament decides the final leaderboard |
| Aug 3 | Bovard #731635/#731810 | Hand spawn on locked SE tile fixed. Limits: 100 MiB, no internet, CPU only, default config, 1 s/turn + 60 s bank |
| Aug 4 | Domino Weir #732450/#731953 | Docs aligned to the engine ("engine is source of truth"): CARE +1, fertilizer sellable, CARE not needed for fertilizer, DIG only empty structures, water on planting day, T uses a 24-day window on purpose |
| Aug 4 | Bovard #732708 | Rating system is the same as Orbit Wars; BT rescore at the end |
| Aug 5 | Domino #732450 | Market orders do not need a shed-adjacent unit |
| Aug 5-6 | Addison, Bovard #732931 | Final BT uses all episodes between active agents; both must be active |
| **Aug 6** | Bovard #733431 | **1.32.6 (PR 1394)**: town center 1x/day flat (was 2x/day with 2x/4x ramps); shops drawn with replacement. Live on the leaderboard Aug 7 |
| Aug 12 | Bovard #734562/#734743 | Servers run the latest published version, no overrides. The seed is not given to agents (it is in replays) |
| **Aug 15** | Bovard #735311 | **1.32.7 (PR 1399)**: hinge scarcity for egg/tomato/carrot (and carrot below_target 0.2→1.0, unannounced). "Should be the last change, excepting game-breaking bugs." Cutover Aug 15, 01:38-01:41 UTC |
| Aug 20 | Bovard #736187 | No midterm evaluation; BT scores are usually close to final ratings |
| Aug 26 | Addison #737431 | Entry deadline (join and accept rules) is separate from the submission deadline, Sep 30 |
| Aug 27 | Bovard #737885 | Scam emails asking for code are fake |
| Aug 28 | Addison #737788 | Anything freely and publicly available is fair use, including submitting public agents |
| Sep 1 | María Cruz #737570 | Game parameters will not change mid-competition or for the final |
| Sep 1 | Bovard #738837 | Using public replays to build a submission is "allowed and encouraged" |
| Sep 4 | Addison #739410 | Team is ranked by the better of 2 submissions; ties = half wins; post-deadline play rate not committed |
| Sep 7 | Bovard #739788/#739874 | Plan to write BT to the private leaderboard; agents run in parallel, so 1200 s runTimeout is safe |
| Sep 14 | Addison #741281 | Public notebook sharing closes **Sep 23 23:59 UTC**. Existing notebooks stayed updatable afterwards because of a bug (#743009) |
| Sep 19 | Bovard #742035 | Fake "bovarddd" scam accounts removed |
| Sep 21 | Bovard #742165 | The live-Elo race condition is irrelevant because BT counts all games |
| Sep 22 | Addison #742571 | BT covers every episode ever played between still-active submissions |

---------------------------------------------------------------------------------------------------

## 5. Ranked actionable ideas to beat the ~2900-3080 top (given about 3 days left)

1. **Optimize P(win) against the actual top lineages, not money.** Top games are decided by margins of about $100-1000 on ~$100k banks.
   Build a gauntlet from the strongest public and cloned lineages (hybrid2965, V53, 2945, herd-safe, wonderful-life, v48-fast-routes, shop-router-0909/v7, and recent top-dump tapes).
   Weight them by how often you meet them at 2450+, add walls, and include mid-band lineages so the agent survives the climb.
   Run paired seeds in both seats on 12 or more seeds, use held-out seeds, and report per-opponent win rates. (#742856, #743231, #740792)
2. **Win the within-turn and cross-turn sell race.** This is the one lever where a small change flips close games.
   - Put contested premium SELLs at **index 0-2**, ahead of HIRE, BUY and other orders.
   - Sell premium stock the turn it lands (DROP+SELL in the same turn works).
   - Time sales for step ≡ 1 (mod 4).
   - Detect mirror or tape opponents (a turn-0 fingerprint of their public actions, or state that tracks yours) and **sell each contested batch one turn before they do**, while protecting your cash-flow chain.
   - Keep wool, milk and strawberry batches small relative to the floor distances (59/76/62 net units) whenever both players are selling. (#740588, #743231, #739273, #734637)
3. **Take a top-class production backbone** (the strongest current tape or router economy) and **fix its mechanical failures**.
   Examples: missed FEED killing animals, blocked actions (repairing a blocked action was worth +$4.4k), carried inventory stranded at the end, shed overflow, idle tiles at nightfall.
   Repair beats redesign; one line changed on a top public agent was worth +147 on the ladder. (#739273, #742856, #741653)
4. **Use fertilizer on strawberry and tomato production days** (+2 per production instead of +1, so up to 8 strawberries per plant with 2 fertilizers), and sell the rest immediately.
   Fertilizer only loses value and has no town demand. (code; #740847, #737736)
5. **Adapt to the shop draw** (the first shop is known after step 71, the second after step 143).
   - Size or divert sheep and wool when no Yarn Store appears (34% of games).
   - Use the tomato and carrot hinge spikes when their shops stack up.
   - Route like the successful shop-routers and forks. (#732613, #738075, #739179, #735311)
6. **Tighten the endgame.**
   - Start no pipelines after about day 26.
   - Liquidate everything by **step 718**: the last action is day 29, hour 22, and there is no final end-of-day.
   - Leave nothing carried or in the shed; stop CARE and FEED only when they can no longer pay.
   - Put BUY_LAND and HIRE early in the order list so they are not truncated at 10. (#735119, #740847, #742943)
7. **Submission hygiene for the BT finish.**
   - Submit the final pair by about Sep 28-29 so both accumulate games against the final population, and submit the keeper **last**.
   - Use the second slot as a deliberately different hedge (for example an anti-mirror market variant); the team takes the better of the two.
   - Pre-screen for timeouts and crashes: avoid first-turn deepcopy, pin 1.32.7, and flag any bank near $3k as a crash. (#739410, #742856)
8. Lower priority or out of reach in 3 days:
   - Macro-level RL with BC warm-up (Snorlax, Sayaka Miki) is proven but needs weeks.
   - Seed recovery is not viable.
   - Opponent-inventory inference is exact for carrot, tomato and egg, but worth only tens of dollars per game unless used for sell timing (#737027, #740792).
