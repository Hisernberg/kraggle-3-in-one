# RSNA Knee: results of 2026-10-09 and the plan to Oct 22

## 1. What happened today

| Submission | Recipe | Public LB |
|---|---|---|
| 14:42 fork of `heliosli/rsna-knee-abnormality-detection` (submitted by the user) | Helios compact bundle | **error**: dataset retired by its owner at 13:50 UTC |
| K1 (`rsna-knee-apex-hardened` v1) | R384 mirror × CNXT, Apex V3 per-target rank weights | **0.950** |
| K2 (v2) | K1 + C224 0.3 inside the CoAtNet side | **0.950** |
| K3 (v3) | K2 + MaVIT at rank weight 0.12 | 0.949 |
| K4 (v4) | K2 + MaVIT at rank weight 0.20 | 0.949 |

**Leaderboard after today** (5,610 teams, measured):
- Best moved from 0.942 (rank ~1954) to **0.950 (rank 759)**.
- 455+ teams are tied at 0.950 and ties go to the earliest submission, so a new 0.950 adds nothing.

| Target | Score needed | Teams at or above |
|---|---|---|
| Top-10 | 0.962 | 11 |
| Top-50 | 0.957 | 55 |
| Top-100 | 0.954 | 107 |
| Top-200 | 0.952 | 186 |
| Top-300 | 0.951 | 301 |

## 2. What the four scores tell us

1. **The hardened notebook works on the hidden set.** All four versions scored, with no crash. The 1,300-study rerun with every leg active fits comfortably in the time limit, so this is now our safe base for every later experiment.
2. **C224 is neutral (0.950 → 0.950).** It comes from the same team, the same labels and the same data as R384, only at a different crop. As predicted, it adds no diversity.
3. **MaVIT is slightly harmful at both 0.12 and 0.20 (−0.001).**
   - On the 3 public studies, its rank correlation with R384 was only 0.75, and 0.46 with CNXT, so it is diverse.
   - But it is also weaker. Like every other public second model, it costs more than it adds. Drop it.
4. **Conclusion:** no public checkpoint left can push the score past 0.950.
   - The only move with public evidence of going further is an **independently trained model on our own labels**. Helios's independent ConvNeXt bundle at 0.3 rank weight on top of R384 gave **+0.002** (0.951–0.952) before it was withdrawn.
   - Teams at 0.957–0.964 all train their own model families.

## 3. Plan to Oct 22

Nothing in this plan is submitted without the user's explicit instruction (`SUBMISSION_LOG.md`).

### 3.1 Own model: the main lever

| Item | Choice | Why |
|---|---|---|
| Architecture | 2.5D ConvNeXt-T (repo `configs/arm_a_convnext_tiny_320.yaml`), 224–320 px, 140 mm crop, slices sorted by patient position, per-finding query attention | Top teams report ConvNeXt-T 5-fold at 0.959 and EffNetV2-S at 0.956; bigger encoders gain nothing (topic 735154); sorting slices alone was worth +0.028 (745759) |
| Labels | Soft report labels (`scripts/label_reports_llm.py`, an allowed LLM labeller, topic 733965) averaged with the best public table (stevenleehans v4, 0.893 on gold-58). Fill silent synovitis from the effusion field (+0.11 AUC on that column, 733932). Treat "small effusion" and "small Baker's" as 0 (radiologist rule, 733826) | Label quality is the biggest controllable factor |
| Folds | 5-fold over all 4,407 studies; gold-58 used **only** as a guard, never for weights | Weights tuned on gold-58 overfit |
| Training | fp16 (never bf16 on T4, 744230), EMA, cosine, ~10 epochs | Per the repo design |
| Round 2 | Pseudo-label round: blend OOF predictions 50/50 into the soft labels and train a student on a different backbone or resolution (EffNetV2-S 320) | Reported by several 0.95+ teams |

GPU budget, at 30 h per week shared with the account's other projects:

| When | GPU budget | Work |
|---|---|---|
| Week of Oct 10 | ~15 h | 5 ConvNeXt-T folds (2 folds in parallel per T4×2 session) |
| Week of Oct 17 | ~12 h | student round + final inference runs |

### 3.2 Blending

- Add the own model as a new leg in `kaggle/apex_hardened`. The structure is already there: a separate process, a timeout, and weight 0 on failure.
- Weights:
  - start with a global rank weight of 0.3 on top of K1 (the Helios-style evidence);
  - then try per-target weights only where OOF supports them.
- Do not add MaVIT, C224 or the 0.943 DINO stack.

### 3.3 Candidate submissions (only on instruction)

| Date | Candidate | Expected |
|---|---|---|
| Oct 12–13 | K1 + own ConvNeXt-T at 0.3 (global) | 0.951–0.953 |
| Oct 13–14 | K1 + own model at 0.2 / 0.4, and per-target OOF weights | direction finding |
| Oct 18–20 | K1 + own teacher + student | 0.952–0.955 |
| Oct 21–22 | final picks | – |

### 3.4 Final selection (2 picks)

1. **Pick 1:** the best own-model blend, if it scores ≥ 0.952 public and its OOF supports it. Otherwise K1 (0.950).
2. **Pick 2:** a structurally different submission: K1 or K2, if Pick 1 is a blend.

### 3.5 Other levers and deadlines

- **Team merge by Oct 15.** Merging with a team that already has its own 0.95x model is the highest-value single move; several teams are recruiting.
- **OAI ruling.** Every 0.950 recipe, ours included, uses nartaa's OAI-trained weights. If the host rules against OAI-trained weights (topics 747455, 747660), the fallback is the own model alone. That is one more reason to build it now.
- **Efficiency prize.** Both final picks also count for the efficiency leaderboard. The current notebook takes ~1.5–2 h, which is not competitive there and is not a goal.

### 3.6 Expected outcome

- **0.951–0.955:** likely, if the own model trains as reported.
- **Top-50 (0.957):** roughly 15–25 %.
- **Top-10 (0.962):** under 5 % without a merge or OAI-scale data.
