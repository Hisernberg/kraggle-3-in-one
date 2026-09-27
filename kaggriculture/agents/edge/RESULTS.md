# Market-edge layer on cha22: results

Base: `public_agents/abhinav0370__cha22-agent/main.py` (cha22). Every edge agent is the base source
with `agents/edge/edge_layer.py` appended by `agents/edge/build.py NAME BASE 'dict(...)'`.
`edge_agent` is the last callable in the file, and the Kaggle loader picks it (checked with
`kaggle_environments.agent.get_last_callable`). The layer only post-processes the `market` list.
Unit actions are untouched. Every edge step is wrapped in try/except, and any error falls back to
the base action.

## Best variant: `agents/edge/cha22_edge_arm/main.py` (built as `v_arm2`)

```
dict(sa=True, sa_front=True, sa_max=1, l2=True, l2_model='robust',
     sa_full_below=1.3, arm_k=2, arm_before=400)
```

1. **Sell-ahead by one unit (SA-1).** From step 216, for each premium item (STRAWBERRY, MELON,
   MILK, WOOL, EGG, TOMATO, CARROT), the layer sells 1 unit more than the base, taken from the stock
   the base keeps (projected shed after this turn's DROP/PICKUP). It never sells at the $1 floor.
   Against a cha22 mirror, our sales run 1 unit per turn ahead, so the mirror sells into a market
   we have already sold into. The base's patience is kept, so absolute revenue barely changes.
2. **Front placement that never evicts base orders.** A new SELL goes to index 0. If 10 orders are
   already queued, it takes an empty `[]` slot or is dropped. Pushing the base's 10th order (a HIRE)
   out of the processed window cost -35k in an early version.
3. **Robust level-2 ordering (L2).** The layer permutes our SELL orders over their own slots, at
   most 6 sells. Orders for items we also BUY_PRODUCT stay in place. Each permutation is scored with
   the exact lockstep simulator (`_v44y_factor_margin`) as our revenue minus the rival's. We keep the
   order with the best worst case over two rival models: (a) the base's own market list, which is
   what a cha22 clone submits, and (b) our final list. Without this, the 1-unit front insert pushed
   a 21-unit strawberry sale to index 1, behind the mirror's index 0 (-780 on seed 3003). The mirror
   record went from 8/12 to 12/12.
4. **Opponent-flow switch to full sell-ahead in crashed markets.** Each turn the layer infers the
   rival's sales from market inventory deltas: town drain is deterministic from the shop list, and
   our own fills are removed. A *pre-emption* is a turn where the rival sold item X while we held X
   and the base did not want to sell X. If at least 2 pre-emptions occur before step 400, the layer
   arms for the rest of the game. Once armed, any item whose price is below 1.3x base is sold
   entirely on sight, at the front of the queue. Tetsutani sells on sight after each tick; in
   crashed strawberry and wool markets, whoever sells first wins. Tetsutani reaches 3-5
   pre-emptions by step 400 on every probed seed. The v5x lineage, dmitrii, thomastschinkel and
   evgendvorkin reach 0-1. haideptry reached 2 on 2 of 6 probed seeds, which cost one game at -36.
   A cha22 mirror never reaches any.

Timing: the edge layer adds at most 10 ms per turn and about 0.15 s per game. The worst turn
measured for the whole agent under a shared CPU was 0.22 s, and those spikes come from the base (the
base alone measured 0.23 s in `rr_top`). The official `kaggle_environments` runner completes games
with status DONE.

## Measured results

Games are seat-symmetric (swapping seats gives identical money), so each (opponent, seed) pair is
played once with the hero in seat 0. The field is the 9 agents in `runs/field_top.txt`; the cha22
row counts its games against itself as ties. Cells are wins/games (mean margin).

| agent | vs cha22 mirror, seeds 3000-3011 | field 2000-2003 | field 2100-2103 | field 2200-2203 | vs tetsutani 2000-2007 | broad 40-agent field, seed 1000 |
|---|---|---|---|---|---|---|
| **base cha22** | ties | 31/36 (+1576) | 31/36 (+2915) | 32/36 (+1779) | 2/8 (-358) | 38/40 (2 mirror ties) |
| `y_m1r` (steps 1-3 only) | 12/12 (+708) | 32/36 (+1196) | 33/36 (+2783) | 34/36 (+1638) | 1/8 (-684) | 40/40 |
| `v_fb13r` (full sell-ahead below 1.3x base from step 216, no switch) | 12/12 (+606) | 31/36 (+975) | 35/36 (+2253) | 35/36 (+1560) | 8/8 (+275) | 36/40 |
| **`v_arm2` = `cha22_edge_arm`** | **12/12 (+708)** | **35/36 (+1302)** | **34/36 (+2877)** | **36/36 (+1725)** | **8/8 (+262)** | **40/40** |

Totals over the three top-field seed sets (108 games): base 94/108 (87%), `y_m1r` 99/108 (92%),
`v_fb13r` 101/108 (94%), **`v_arm2` 105/108 (97%)**.

Per-opponent notes for `v_arm2`:
- **cha22 mirror:** 12/12 on seeds 3000-3011, minimum margin +137. It also beat the two exact cha22
  copies in the broad field (flexonafft multi-route and guruprasaathas master-engine) that the base
  ties.
- **tetsutani:** 8/8 on seeds 2000-2007 and 2/4 on 2100-2103, where it lost by -29 and -713. It
  arms at step 393, and some damage is already done by then. That makes 14/16 over all tetsutani
  games. The base is 2/8 on seeds 2000-2007.
- **Its three losses in the 108-game field:** haideptry on 2003 (-36; a false arm) and tetsutani on
  2100 and 2103.

Fresh-seed status: the arming thresholds were chosen from probe games on seeds 2000-2003 and
2100-2101. Seed set 2200-2203, the broad field on seed 1000 and the mirror seeds 3000-3011 were
never used for tuning. They give 36/36, 40/40 and 12/12.

Other candidate files kept: `agents/edge/cha22_edge_m1r/main.py` (= `y_m1r`, steps 1-3 only, the
safest choice if tetsutani-type sellers are rare) and `agents/edge/cha22_edge_fb13r/main.py`
(= `v_fb13r`).

## What helped, what hurt (idea by idea)

Helped:
- **Sell-ahead of a mirror, 1 unit per turn (idea 2).** The mirror goes from all ties to all wins,
  at almost no cost in absolute revenue.
- **Order-slot priority (idea 1), done as a lockstep best response against a robust pair of rival
  models.** A fixed rule ("premium first") or a best response to our own list alone was worse:
  `y_m1` with only the self model won 8/12 mirror games, against 12/12 with the robust pair.
- **Opponent-flow inference (idea 3)** used as a switch. It is exact enough to count the rival's
  per-item units, and pre-emption counts separate on-sight sellers (tetsutani) from patient ones
  (the cha22/v5x lineage).
- **Full sell-ahead in crashed markets (price < 1.3x base), always at index 0.** It flips tetsutani.
  With `after_sells` placement the same idea scores 0-1/8 against tetsutani.

Hurt or neutral:
- **Full sell-ahead of all stock (`e_sa*`).** It wins the mirror but dumps strawberries: -1.9k of
  strawberry revenue against thomastschinkel-2945. It won 83-89% of the field, against 86% for the
  base.
- **Price floors or caps on the full sell-ahead** (`x_sr08`, `x_sr10`, `x_s1`, `x_ns`): 4/8 against
  tetsutani. Thresholds of 1.0x (`v_fb10r`) and 1.6x (`v_fb16r`) were worse than 1.3x.
- **Moving all SELLs before HIRE/BUY (`e_sf`):** 4/24 against the base.
- **Arming on late or high pre-emption counts** (`w_a3`, `w_a5`, `w_g12`, `v_ck4r`): 0-6/8 against
  tetsutani, because the switch comes too late.
- **Clone or position gates and cash-lead gates:** they cannot separate opponents. Every public
  agent in the field runs the same production tape with identical unit positions, and holding
  stock makes cash look behind in mid-game.
- **Never selling at or below $3 before the last day (idea 5, `z_fh3`):** neutral (-6 on the test
  game). Milk that is held just gets dumped later.
- **End-game liquidation (idea 4):** nothing to gain. The base already ends with an empty shed and
  empty unit inventories. Step-718 DROP+SELL is handled.
- **Plain level-2 reordering alone (`e_l2`):** 21/24 against the base, but only about +60 per game.

## Reproduce

```
python agents/edge/build.py NAME public_agents/abhinav0370__cha22-agent/main.py "dict(...)"
python agents/edge/gaunt.py --hero agents/edge/NAME/main.py --opps-file runs/field_top.txt --seeds 4 --seed0 2200 --workers 2 --out runs/x.jsonl
python agents/edge/summary.py abhinav0370__cha22-agent v_arm2      # table above (reads runs/edge_*.jsonl)
```
Raw results are in `runs/edge_mirror.jsonl`, `runs/edge_field2.jsonl`, `runs/edge_fresh.jsonl`,
`runs/edge_fresh3.jsonl`, `runs/edge_tet.jsonl` and `runs/edge_broad.jsonl`.
