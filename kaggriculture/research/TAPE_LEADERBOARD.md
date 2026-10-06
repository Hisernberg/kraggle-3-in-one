# Tape leaderboard (robustness screen of recorded top-player action tapes)

Generated 2026-09-27 by the tape-mining run. Agents are `agents/tapeplay/raw_<episode>_<player>` built with `build_tapeplay.py ... --switch-day 30`; tapes in `tapes/`, catalog in `runs/tape_catalog.csv`.

## Protocol
* **Stage 1** (`runs/tapescreen2_s1.jsonl`): vs `haideptry__the-2965-master-hybrid-engine` only, seeds 5000-5008, both seats = 18 games.
  Run sequentially: seeds 5000-5002 for all 448 mined tapes, seeds 5003-5008 only for the 149 that won at least one of those 3 worlds.
  The 15 re-screened older tapes also got the full 9 seeds.
* **Stage 2** (`runs/tapescreen2_s2.jsonl`): vs the 9 agents in `runs/field_top.txt`, seeds 6000-6003, both seats = 72 games.
  Mined tapes with stage 1 >= 10/18 (28 tapes) plus the top 15 older tapes from `runs/tapescreen.jsonl` (43 tapes in total).
* **Stage 3** (`runs/tapescreen2_s3.jsonl`): the top 19 stage-2 tapes vs haideptry, seeds 7000-7015, both seats = 32 games (16 more worlds).

**Why stage 1 changed: the world decides the result.** On one seed a tape usually beats all 9 field agents in both seats (18/18)
or loses to all of them (0/18). Within a seed the field agents are almost interchangeable: haideptry's per-(tape, seed) result correlates
0.96 with the sum over the rest of the field. So `--seeds 1` vs the full field measures one world 18 times, and 15/18 on one seed
only says the tape suits that world. Stage 1 spends the same 18 games on 9 worlds against a single representative opponent.
Nothing reached 15/18 under this stage 1 (best was 14/18), so the stage-2 cut was set at 10/18.
Stage 2 by itself covers only 4 worlds and is still coarse (win% moves in steps of about 25 points per world),
so the **29-world** columns (stage 1 + 2 + 3) are the better measure of robustness.

## Stage-2 ranking (72 games vs field_top, seeds 6000-6003)

| # | tape | team (LB rank) | orig. opponent | orig. result | S2 win% | S2 avg margin | S2 worlds | S1 (/18) | S3 (/32) | all-games win% | worlds won (all) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `raw_114017715_1` | Unknown Mother-Goose (7) | Just A game on your lips | W 109444-103887 | 77.8 | +3164 | 3/4 | 12 | 22 | 73.8 | 20/29 |
| 2 | `raw_114013601_0` | M & M & P & Q (4) | Yizhou | W 118732-112456 | 75.0 | -3652 | 3/4 | 10 | 14 | 63.9 | 15/29 |
| 3 | `raw_113944054_0` | DECEM (6) | Anton Tikhonov | W 122061-111481 | 72.2 | +4106 | 3/4 | 14 | 14 | 65.6 | 17/29 |
| 4 | `raw_113888303_0` | DSM (1) | M & M & P & Q | W 107024-105957 | 69.4 | +3237 | 3/4 | 12 | 14 | 62.3 | 16/29 |
| 5 | `raw_113959718_1` | DECEM (6) | Anton Tikhonov | W 106439-93936 | 63.9 | +979 | 3/4 | 8 | 12 | 54.1 | 13/29 |
| 6 | `raw_114009306_1` | Vadim Vasilenko (3) | Unknown Mother-Goose | W 102629-97961 | 61.1 | -516 | 2/4 | 10 | 18 | 59.0 | 16/29 |
| 7 | `raw_113939686_0` | Unknown Mother-Goose (7) | Vadim Vasilenko | L 115377-116756 | 55.6 | -2273 | 2/4 | 6 | 14 | 49.2 | 12/29 |
| 8 | `raw_113970426_1` | DECEM (6) | Boey | L 84522-86882 | 55.6 | -5877 | 2/4 | 8 | 18 | 54.1 | 15/29 |
| 9 | `raw_113970425_0` | DSM (1) | Boey | W 112354-109355 | 52.8 | -3951 | 2/4 | 16 | 22 | 62.3 | 21/29 |
| 10 | `raw_113985942_0` | Boey (2) | DSM | L 116271-116442 | 50.0 | -11217 | 2/4 | 10 | 8 | 44.3 | 11/29 |
| 11 | `raw_113973950_1` | KawattaTaido (8) | mtmr_s1 | L 98753-100798 | 50.0 | -12304 | 2/4 | 6 | 8 | 41.0 | 9/29 |
| 12 | `raw_113911253_0` | Boey (2) | DSM | W 110643-109822 | 50.0 | -15199 | 2/4 | 12 | 10 | 47.5 | 13/29 |
| 13 | `raw_114001736_0` | Boey (2) | DSM | L 101334-103317 | 50.0 | -15826 | 2/4 | 10 | 10 | 45.9 | 12/29 |
| 14 | `raw_114013386_0` | Unknown Mother-Goose (7) | Just A game on your lips | L 90339-92547 | 44.4 | -7816 | 2/4 | 10 | 16 | 47.5 | 15/29 |
| 15 | `raw_113963514_0` | DSM (1) | Vadim Vasilenko | W 104495-102687 | 44.4 | -7817 | 2/4 | 10 | 15 | 46.7 | 15/29 |
| 16 | `raw_114029949_1` | DECEM (6) | M & M & P & Q | L 107814-108090 | 44.4 | -8540 | 2/4 | 12 | 20 | 52.5 | 18/29 |
| 17 | `raw_113985949_1` | Vadim Vasilenko (3) | KawattaTaido | W 97240-91050 | 41.7 | -10380 | 2/4 | 8 | 13 | 41.8 | 13/29 |
| 18 | `raw_113908167_0` | Unknown Mother-Goose (7) | Yizhou | W 107038-100807 | 41.7 | -12360 | 2/4 | 12 | 18 | 49.2 | 17/29 |
| 19 | `raw_114024569_1` | Boey (2) | Vadim Vasilenko | W 120158-116817 | 41.7 | -13306 | 2/4 | 10 | 10 | 41.0 | 12/29 |
| 20 | `raw_113957215_0` | DSM (1) | Vadim Vasilenko | W 129665-119104 | 38.9 | -14166 | 2/4 | 6 | - | 37.8 | 5/13 |
| 21 | `raw_113909332_1` | Unknown Mother-Goose (7) | Anton Tikhonov | W 133949-129600 | 36.1 | -1039 | 1/4 | 10 | - | 40.0 | 6/13 |
| 22 | `raw_114009429_0` | Unknown Mother-Goose (7) | TheEggman | W 116206-108637 | 33.3 | -6013 | 1/4 | 12 | - | 40.0 | 7/13 |
| 23 | `raw_114001731_1` | DSM (1) | Boey | W 108286-105165 | 30.6 | -4874 | 1/4 | 10 | - | 35.6 | 6/13 |
| 24 | `raw_114017662_0` | Vadim Vasilenko (3) | M & M & P & Q | W 118186-115531 | 30.6 | -9345 | 1/4 | 10 | - | 35.6 | 6/13 |
| 25 | `raw_113857202_1` | Boey (2) | DSM | L 127673-130052 | 25.0 | -6765 | 1/4 | 10 | - | 31.1 | 6/13 |
| 26 | `raw_113917708_1` | Boey (2) | Vadim Vasilenko | L 137708-137783 | 25.0 | -7445 | 1/4 | 10 | - | 31.1 | 6/13 |
| 27 | `raw_113976853_0` | KawattaTaido (8) | Just A game on your lips | W 107026-106409 | 25.0 | -8326 | 1/4 | 8 | - | 28.9 | 5/13 |
| 28 | `raw_114013220_1` | DECEM (6) | mtmr_s1 | W 94525-79276 | 22.2 | -12385 | 1/4 | 12 | - | 31.1 | 7/13 |
| 29 | `raw_113993850_1` | Vadim Vasilenko (3) | DSM | L 113038-114728 | 22.2 | -12864 | 1/4 | 12 | - | 31.1 | 7/13 |
| 30 | `raw_113946289_0` | DECEM (6) | Vadim Vasilenko | W 90393-89080 | 22.2 | -21010 | 1/4 | 10 | - | 28.9 | 6/13 |
| 31 | `raw_114031216_0` | DECEM (6) | Just A game on your lips | W 82542-72065 | 22.2 | -30633 | 1/4 | 10 | - | 28.9 | 6/13 |
| 32 | `raw_113963510_0` | DSM (1) | M & M & P & Q | W 119138-115602 | 16.7 | -18024 | 0/4 | 4 | - | 17.8 | 2/13 |
| 33 | `raw_113963514_1` | Vadim Vasilenko (3) | DSM | L 102687-104495 | 8.3 | -15175 | 0/4 | 8 | - | 15.6 | 4/13 |
| 34 | `raw_113944967_0` | Vadim Vasilenko (3) | mtmr_s1 | W 99318-84375 | 8.3 | -17788 | 0/4 | 10 | - | 17.8 | 5/13 |
| 35 | `raw_114025413_0` | DECEM (6) | TheEggman | W 106978-95800 | 5.6 | -18884 | 0/4 | 14 | - | 20.0 | 7/13 |
| 36 | `raw_114009672_1` | DECEM (6) | M & M & P & Q | L 111635-116186 | 5.6 | -20083 | 0/4 | 12 | - | 17.8 | 6/13 |
| 37 | `raw_113978089_1` | Vadim Vasilenko (3) | Boey | W 102252-98133 | 5.6 | -22746 | 0/4 | 6 | - | 11.1 | 3/13 |
| 38 | `raw_113955909_0` | Majkel1337 (5) | Vadim Vasilenko | L 110288-113265 | 0.0 | -10631 | 0/4 | 10 | - | 11.1 | 5/13 |
| 39 | `raw_113875580_1` | M & M & P & Q (4) | Boey | W 117408-116764 | 0.0 | -11919 | 0/4 | 14 | - | 15.6 | 7/13 |
| 40 | `raw_113917717_0` | Majkel1337 (5) | DSM | L 110037-110393 | 0.0 | -15823 | 0/4 | 12 | - | 13.3 | 6/13 |
| 41 | `raw_113962528_1` | Unknown Mother-Goose (7) | Vadim Vasilenko | L 98395-107004 | 0.0 | -22310 | 0/4 | 0 | - | 0.0 | 0/13 |
| 42 | `raw_114017765_0` | Vadim Vasilenko (3) | M & M & P & Q | L 109979-110323 | 0.0 | -25766 | 0/4 | 10 | - | 11.1 | 5/13 |
| 43 | `raw_113955912_0` | DECEM (6) | Unknown Mother-Goose | W 93604-91031 | 0.0 | -31612 | 0/4 | 10 | - | 11.1 | 5/13 |

## Most robust (finalists ranked by worlds won across all 29 seeds)

| # | tape | team | worlds won /29 | all-games win% (122 g) | all-games avg margin | S2 win% |
|---|---|---|---|---|---|---|
| 1 | `raw_113970425_0` | DSM | 21/29 | 62.3 | -852 | 52.8 |
| 2 | `raw_114017715_1` | Unknown Mother-Goose | 20/29 | 73.8 | +2830 | 77.8 |
| 3 | `raw_114029949_1` | DECEM | 18/29 | 52.5 | -4568 | 44.4 |
| 4 | `raw_113944054_0` | DECEM | 17/29 | 65.6 | +2465 | 72.2 |
| 5 | `raw_113908167_0` | Unknown Mother-Goose | 17/29 | 49.2 | -6733 | 41.7 |
| 6 | `raw_113888303_0` | DSM | 16/29 | 62.3 | +1432 | 69.4 |
| 7 | `raw_114009306_1` | Vadim Vasilenko | 16/29 | 59.0 | -277 | 61.1 |
| 8 | `raw_114013601_0` | M & M & P & Q | 15/29 | 63.9 | -3642 | 75.0 |
| 9 | `raw_113970426_1` | DECEM | 15/29 | 54.1 | -2163 | 55.6 |
| 10 | `raw_114013386_0` | Unknown Mother-Goose | 15/29 | 47.5 | -5250 | 44.4 |
| 11 | `raw_113963514_0` | DSM | 15/29 | 46.7 | -6917 | 44.4 |
| 12 | `raw_113959718_1` | DECEM | 13/29 | 54.1 | -5519 | 63.9 |
| 13 | `raw_113911253_0` | Boey | 13/29 | 47.5 | -13122 | 50.0 |
| 14 | `raw_113985949_1` | Vadim Vasilenko | 13/29 | 41.8 | -7895 | 41.7 |
| 15 | `raw_113939686_0` | Unknown Mother-Goose | 12/29 | 49.2 | -3768 | 55.6 |
| 16 | `raw_114001736_0` | Boey | 12/29 | 45.9 | -14490 | 50.0 |
| 17 | `raw_114024569_1` | Boey | 12/29 | 41.0 | -12076 | 41.7 |
| 18 | `raw_113985942_0` | Boey | 11/29 | 44.3 | -11174 | 50.0 |
| 19 | `raw_113973950_1` | KawattaTaido | 9/29 | 41.0 | -13105 | 50.0 |

## Which teams' tapes travel best

Phase A = the first 3 stage-1 worlds (seeds 5000-5002 vs haideptry), run on all 448 mined tapes. It is the one unbiased per-team comparison.

| team (LB rank) | mined tapes | phase-A game win% | tapes winning >=1 of 3 worlds | stage-2 tapes (of 43) | best 29-world result |
|---|---|---|---|---|---|
| Boey (2) | 27 | 37.0 | 21 | 6 | `raw_113911253_0` 13/29 |
| DECEM (6) | 27 | 32.1 | 18 | 10 | `raw_114029949_1` 18/29 |
| KawattaTaido (8) | 20 | 28.3 | 13 | 2 | `raw_113973950_1` 9/29 |
| Vadim Vasilenko (3) | 45 | 23.0 | 26 | 8 | `raw_114009306_1` 16/29 |
| Unknown Mother-Goose (7) | 30 | 22.2 | 15 | 7 | `raw_114017715_1` 20/29 |
| DSM (1) | 46 | 16.7 | 20 | 6 | `raw_113970425_0` 21/29 |
| M & M & P & Q (4) | 46 | 15.2 | 16 | 2 | `raw_114013601_0` 15/29 |
| TheEggman (14) | 23 | 11.6 | 6 | 0 | - |
| Fourth Quadrant (9) | 14 | 9.5 | 4 | 0 | - |
| Majkel1337 (5) | 36 | 7.4 | 7 | 2 | - |
| mtmr_s1 (15) | 27 | 1.2 | 1 | 0 | - |
| Anton Tikhonov (11) | 27 | 1.2 | 1 | 0 | - |
| Yizhou (13) | 28 | 1.2 | 1 | 0 | - |
| Azat Akhtyamov (10) | 23 | 0.0 | 0 | 0 | - |
| Just A game on your lips (12) | 29 | 0.0 | 0 | 0 | - |

**Takeaways**
* **DECEM, Unknown Mother-Goose, DSM and Boey** travel best. Their tapes hold up in random worlds most often, and together they make
  up 15 of the 19 finalists. DECEM and Boey have the highest base rates (32-37% phase-A game win% across all their tapes). DSM and
  Unknown Mother-Goose have fewer good tapes, but their best ones are the most robust overall: `raw_113970425_0` won 21/29 worlds and
  `raw_114017715_1` won 20/29.
* **Vadim Vasilenko and KawattaTaido** are mid-pack: about 23-28% phase-A game win%, with a few decent tapes.
* **M & M & P & Q** has one good tape (`raw_114013601_0`, 75% in stage 2) but a low base rate (15%).
* **Majkel1337, TheEggman, Fourth Quadrant, mtmr_s1, Anton Tikhonov, Yizhou, Azat Akhtyamov and Just A game on your lips** barely
  work open-loop (0-12% phase-A game win%). These players probably adapt to the live state, so their recorded actions do not
  transfer to another world.
* Tapes from games the player won transfer somewhat better (17.6% vs 10.9% phase-A game win%), but the original score margin is a
  weak predictor.
* **No tape is robust in every world.** The best win about 70% of random worlds. The earlier 53/54 for `raw_113970425_0` came from
  lucky seeds: here it won 2 of the 4 stage-2 worlds (52.8%) but 21/29 worlds overall. Always compare tapes on many seeds, not many games.
* Stage-1 score is only a loose predictor of stage 2. For example, `raw_113875580_1` (M&M&P&Q) scored 14/18 in stage 1 and 0% in stage 2.
  Rank by the 29-world column.

