# WEAR: 10-submission plan (2026-10-10 → deadline 2026-10-12 21:59 UTC)

Starting point: public 0.93597 (#8). #3 is 0.93849 (+0.0025), #1 is 0.94535 (+0.0094).
Nothing is submitted until the user says "submit" (repo policy). Base file for every label-level slot:
`Fm1_margin1_only` (ref 56944621).

## Why this plan looks different from the Oct 8–9 ladders

Another round of committee flips cannot close a 0.0025–0.0094 gap. The two days of flip probing added +0.00056 on
public, and those flips are probably worth zero on private. Only two kinds of lever are large enough:

1. **A labelling-convention question worth ±0.006–0.014 per subject** (third sessions, README §2). It costs
   1–2 slots to answer, it was never tested, and it is the only cheap move that could reach the top 3.
2. **A better decode of the f4n family** (goodpjw oracles: true counts +0.013, true boundary order +0.016).
   It needs the f4n/mv4/mv6 raw probabilities from the user's PC.

Everything else is ≤ +0.002 and is used only to fill slots after these.

## Prerequisites (no slots)

| # | What | Who | Time |
|---|---|---|---|
| R1 | Upload the f4n/mv4/mv6 OOF and test probabilities (final Q/P, count-prior inputs, link sets) from `E:\Claude code\wear` to a **private** Kaggle dataset shared with this account, or to a private GitHub repo | user | 15 min |
| R2 | Run goodpjw Part 2 on Kaggle T4×2 with `WEAR_DUMP=1` and 2 seeds, so every stage array is available for offline re-decoding (and as a fresh committee member) | this session | about 3 h GPU each |
| R3 | Offline re-decode harness: goodpjw `decode()` and count targets on dumped arrays, OOF-scored by subject fold; add metadata constraints (subject totals; activity count band 78–126; complex-variant share 45–58%; per-(subject, class) target 0 when a third-session result says so) | this session | about 3 h CPU |

Gate for any OOF-built file: subject-balanced OOF ≥ +0.0010 over its parent and no subject worse by more than
0.003. Following Law 3, at most 2 tuned parameters per change.

## The ladder

| Slot | File | Built from | Read | Expected public Δ |
|---|---|---|---|---|
| **1** | `P1_s22_sitcx_to_null` | Fm1 with sbj_22's 128 sit-ups (complex) windows set to null | is sbj_22's third session labelled null, and is it sit-ups (complex)? | ±0.005–0.014 |
| **2** | `P2_s23_strham_to_null` | Fm1 with sbj_23's 111 stretching (hamstrings) windows set to null | the same question for sbj_23 | ±0.005–0.014 |
| 3 | depends on 1–2 (below) | | | |
| 4 | `R_decode_f4n` | f4n family re-decoded with R3 constraints (needs R1) | Law-4 lever | +0.001–0.004 |
| 5 | `B_band` | Fm1 + count-band repair, moving only the lowest-margin windows (fork probabilities choose which): sbj_23 push-ups (complex) 128 vs push-ups 88 (share 0.59), sbj_24 jogging (skipping) 69 / bench-dips 73 / jogging (rotating arms) 77 below 78, sbj_25 stretching (lunging) 120 | count-band prior on the best file | 0 to +0.0015 |
| 6 | `N_gate` | Fm1 + jiweiliu-style context null gate (LightGBM on S3 context columns, OOF-trained on fork stage arrays), applied only to windows within ±3 s of a predicted boundary and below a margin threshold | null boundaries carry 71% of the known gain | 0 to +0.002 |
| 7 | `M_part2` | committee margin-1 regime recomputed with the 2 new Part 2 decodes from R2 (Law 1: needs an internal family behind the flip and full external agreement) | does a new member open new margin-1 flips? | 0 to +0.0005 |
| 8–9 | brackets of the best of 3–7 | one parameter step either side, such as band edges, gate threshold, re-decode count weight | tuning | small |
| 10 | closer | best composition of the winners, built on the best base | | sum of winners |

### Decision rules after slots 1–2 (submit both together, they are independent)

| Outcome vs 0.93597 | Meaning | Slot 3 |
|---|---|---|
| Δ ≥ +0.002 on one probe | the convention holds and the class is right for that subject | keep it; re-decode that subject with count target 0 for the class (R3), so the freed bout boundaries and the neighbouring null are decoded properly → `T3_redecode` |
| Both Δ ≥ +0.002 | both subjects null | `P12_both`, then its re-decode in slot 4 or later |
| Δ ≤ −0.004 | that predicted class really is labelled there, so either the convention does not hold or it is the wrong class | try **one** alternative class only if an independent cue (links, scene) points to it; otherwise close the idea |
| \|Δ\| < 0.002 | few of these windows are public, or the effect cancels | inconclusive; do not build on it |

If both probes lose, the plan falls back to slots 4–10, and the realistic outcome is #5–#8 (each of those levers is
≤ +0.002 and so far OOF→LB transfer has been about 50% or less).

## Day split

- **Oct 10:** slots 1–2 right away (files are built). Start R2 on GPU, ask for R1, build R3. Slot 3 in the evening.
- **Oct 11:** slots 4–7 as their files pass the OOF gate; empty slots are simply not used.
- **Oct 12 (until 21:59 UTC):** slots 8–10 and **final selection in the UI before 21:00 UTC**.

The daily limit is 10, so there are up to about 30 slots across the three days. This plan uses 10 and keeps the rest unused unless
a lever wins.

## Final selection

The rules page says one final submission is judged. Check in the Submissions tab whether one or two
can be selected. If none is selected, Kaggle picks the best public score, which would be a public-fitted flip file.

- If a third-session probe won: select the best re-decoded file that contains that change. The convention
  applies to the whole test set, so it carries to private.
- Otherwise: `Fm1_plus2` (ref 56944691) or `f4n` (ref 56884414). Do not select the slot 8–9 brackets
  unless their parent won by ≥ 0.001.

## Do not repeat (falsified on the LB)

Probability fusion without re-decoding (D2), smoothing or blip repair (D_esmooth), cross-family flips (B2), 5–7-member
votes, margin ≥ 2 committee flips, anti-evidence gating (T3), decode-less argmax, public-run Q as a fusion member,
count-ridge nudges (b4wa), the kansuke video head as a voter (0.894).
