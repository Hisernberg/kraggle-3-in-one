# Market-edge layer on cha22: results (work in progress)

Base: `public_agents/abhinav0370__cha22-agent/main.py` (cha22). Every edge agent is the base source plus
`agents/edge/edge_layer.py`, appended by `agents/edge/build.py NAME BASE 'dict(...)'`. The wrapper
`edge_agent` is the last callable in the file. It post-processes only the `market` list.

## Current best (first candidate): `agents/edge/cha22_edge_m1r/main.py` (= `y_m1r`)

Config: `sa=True, sa_max=1, sa_front=True, l2=True, l2_model='robust'`.

1. **Sell-ahead by one unit (SA-1).** Each turn from step 216, for each premium item
   (STRAWBERRY, MELON, MILK, WOOL, EGG, TOMATO, CARROT), the layer sells 1 unit more than the base
   out of the stock the base keeps (projected shed after this turn's DROP/PICKUP). It never sells at
   the $1 floor. Against a mirror this puts our cumulative sales 1 unit per turn ahead, so the mirror
   sells into a market we have already hit. It does not change the base's patience, so absolute
   revenue is kept.
2. **Front placement without evicting base orders.** A new SELL goes to index 0. If the queue already
   has 10 orders, it takes an empty `[]` placeholder slot or is skipped. It never pushes a base
   HIRE/BUY past slot 10. Pushing one out was a -35k bug.
3. **Robust level-2 ordering (L2).** The layer permutes our SELL orders over their slots (at most 6
   sells; BUY_PRODUCT items stay fixed) with the exact lockstep simulator (`_v44y_factor_margin`).
   It maximises the minimum of our revenue minus the rival's under two rival models: (a) the base's
   own orders, which a cha22 clone submits, and (b) our final orders. Without this, the 1-unit
   front insert pushed a 21-unit strawberry sale to index 1 behind the mirror's index 0 (-780 on
   seed 3003).

Timing: max turn 0.14 s locally (base 0.11 s), about 1.1 s agent time per game.
