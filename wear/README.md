# WEAR @HASCA 2026 (`3rd-wear-dataset-challenge-hasca-2026`)

Research and 10-submission plan as of 2026-10-10. Deadline **2026-10-12 21:59 UTC**. Team **Koushik Rudra**,
public **#8, 0.93597** (`Fm1_margin1_only`, ref 56944621). Leaders: 0.94535, 0.94355, 0.93849 (#3).

Plan: [`PLAN.md`](PLAN.md). Scripts: `scripts/` (no data and no tokens in this repo).

## 1. Prize eligibility (read first)

The Description, Prizes and Rules pages all say that prizes go to the three best private-LB teams
**as of the technical-report deadline (July 5, 2026)**, and only to teams that submitted a peer-reviewed
HASCA technical report. On 2026-09-26 the organiser (Marius Bock) wrote in the thread "Why Is the Competition
Still Open…" that the later Kaggle audience "does not qualify for prizes". The team's first submission
was 2026-09-01, so **no rank on this board can win the €300/150/75**. What remains is the Kaggle
ranking itself and a forum write-up or arXiv preprint, which the organiser offered to feature at the Shanghai workshop.

## 2. Data facts that matter

- Test: 12,234 one-second windows, one random limb each (about 25% per limb), plus 15 VideoMAEv2 frames
  (array is `(N, 768, 15)`, frames 8–22 of each second).
- Per subject: 22 → 5197, 23 → 3128, 24 → 1995, 25 → 1914 windows. That is every second of every
  session in `participant_meta_data.txt` (exact for 23/24/25; sbj_22 is 180 s longer than its listed sessions).
- Every subject performs all 18 activities once, 80–110 s each in train. In test that is roughly
  380 windows per activity class, so one corrected window is worth about +0.00014 macro F1 (on the whole set).
- **One-activity third sessions.** sbj_22 (2:57, location 13, a different day) and sbj_23 (1:47) each have a third session
  with one activity. In train, sbj_2 has the same layout: 17 activities labelled, and its 112 s third session
  holds the 18th (jogging (rotating arms)) but is **labelled null**. sbj_10's 1,419 s third session is null too.
  If the test labels follow the same convention, the current file predicts about 100–130 windows per
  affected subject as an activity where the truth is null, worth about ±0.006–0.014 each. This was never tested in
  170 submissions. Which class sits in each third session is *not* established. The best candidates are
  sbj_22 → sit-ups (complex) (128 predicted, the subject's maximum) and sbj_23 → stretching (hamstrings);
  video clustering alone could not confirm either.
- Video-only successor linking is weak: the true next second is the top-1 match in 20–29% of train
  windows. Ordering comes from accelerometer links (goodpjw L3: 65–68% exact).

## 3. What the 170 previous submissions established

| Phase | Best public | Lesson |
|---|---|---|
| 09-18..22 per-window models + calibration | 0.715 | decode-less models cap at about 0.71 |
| 09-23..27 own chain/graph decoder (mrf4) + votes | 0.889 | timeline decoding is worth +0.15 |
| 09-28..10-03 Hanbat graph + our links, then goodpjw Learned Links + Counts | 0.927 | learned links + count prior are the core |
| 10-04..07 4-run chain-link fusion + log-count prior (`f4n_clog`, ref 56884414) | **0.93541** | best honest model |
| 10-08..09 committee flips (F, Fm1, B_null4, W_a/W_b…) | 0.93597 | +0.00056 from 8 single-window flips; public-only evidence |

Recorded laws (from the deep-research repo, docs 09–12):
1. Probability fusion without re-decoding loses about 0.01 (D2 0.92539).
2. Smoothing or blip removal loses (D_esmooth −0.0166).
3. OOF gains from stacks with many knobs did not carry to the LB (G stack: OOF 0.9357 → LB 0.93308).
4. Per-window public deltas add up exactly.
5. 71% of the flip gain came from null↔exercise boundaries.

The flip gains are public-LB fitting; expect about 0 of the +0.00056 on private.

## 4. Assets available to this cloud session

- All 170 scored submission files (`scripts/dl_subs.py`), including Fm1, f4n, mv4, mv6 and the committee members.
- Team kernel outputs (readable with this token): `koushikrudra/wear-good-fork` and `-fork2`. These hold goodpjw-pipeline
  OOF and test final probabilities, stage arrays (window, S3, T experts, QA/QB, 8 link matchings, L2 links), and labels/folds.
  Also `wear-hanbat-gpu`, `wear-hanbat-l2oof`, `wear-fusion-full`, `wear-uec-k1` and `wear-prep-windows`.
  OOF macro F1 of fork1 final probabilities is 0.923. Errors: 2,063 null→act, 2,266 act→null, 1,030 act→act.
- **Not available here:** the raw probabilities of the f4n/mv4/mv6 family. They live only on the Windows PC
  (`E:\Claude code\wear`). The one remaining large lever (a better decode on that family) needs them.
- Public code: goodpjw Part 2 (`WEAR_DUMP=1` dumps every stage), kansukehabano video-graph (count band 78–126,
  complex-variant share 45–58%, top-30-PC removal), jiweiliu context null gate.

## 5. Security

Tokens were pasted in chat, and the **public** repo `Hisernberg/wear-hasca-2026-deep-research` hard-codes a Kaggle
token in `code/download_own_subs.py`, `code/kaggle_probe.py` and `code/submit_candidate.py`. Rotate the Kaggle,
GitHub and Hugging Face tokens.
