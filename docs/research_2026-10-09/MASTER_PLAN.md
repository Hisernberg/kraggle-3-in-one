# Master plan, 2026-10-09: Knee, CASMI, Filament, TrafficFlow

Consolidated from six research agents (2026-10-09 15:00–16:00 UTC). Account `kragglenote2forwork`.
Nothing was submitted. Detailed reports:

| Competition | Reports |
|---|---|
| RSNA Knee | [`KNEE_NOTEBOOKS.md`](KNEE_NOTEBOOKS.md) (39 notebooks, code-diff lineage), [`KNEE_DISCUSSIONS.md`](KNEE_DISCUSSIONS.md) (173 topics) |
| CASMI 2026 | [`CASMI_NOTEBOOKS.md`](CASMI_NOTEBOOKS.md) (85 notebooks), [`CASMI_DISCUSSIONS.md`](CASMI_DISCUSSIONS.md) (76 topics, EDA) |
| Solar Filament | [`../../cobalt_heron/docs/RESEARCH_2026-10-09.md`](../../cobalt_heron/docs/RESEARCH_2026-10-09.md) |
| TrafficFlow | [`../../trafficflow/docs/RESEARCH_2026-10-09.md`](../../trafficflow/docs/RESEARCH_2026-10-09.md) |

Evidence tags in the reports: **[M]** measured (API, code, logs), **[A]/[C]** claimed by an author, **[E]** estimate.

## The team

| Persona | Role |
|---|---|
| **Lead strategist** (main session) | Sets scope, reconciles conflicts between agents, owns the plan and the submit decisions |
| **Dr. Patella: Lineage Hunter** | Knee notebook forensics: code diffs, sources, crash risks, blend arithmetic |
| **Dr. Patella: Forum Oracle** | Knee rules, metric, labels, top-team disclosures, prior RSNA art |
| **Prof. Fragment: Lineage Hunter** | CASMI notebook forensics: lineage, leak/licence flags, today's 5 builds |
| **Prof. Fragment: Forum Oracle** | CASMI metric, rules, leak thread, EDA, MS/MS literature |
| **Helio** | Filament: leak analysis, honest frontier, rubric-driven plan |
| **Wardrop** | TrafficFlow: board movement, per-task headroom, recovery plan |

## State on 2026-10-09

| | Our best public | Rank | Realistic target | Top-10 needs | Deadline |
|---|---|---|---|---|---|
| RSNA Knee | 0.942 | 1954 / 5586 | 0.950–0.951 today; 0.951–0.955 with own model | 0.962 | Oct 22 (merge Oct 15) |
| CASMI 2026 | 0.328 | ~1550 / 2899 | 0.43–0.45 today; 0.46+ with own work | ~0.448–0.455 | Dec 14 (merge Dec 7) |
| Filament | 0.55 (**copied leak file, never select**) / honest 0.37 | 54 | honest LB ≈ 0.42, rubric-judged | (≥ 0.60 = leak territory) | Nov 15 (form) |
| TrafficFlow | **0.87249** | 53 raw / 47 post-rebuild | 0.877–0.883 public + private line | 0.890 post-rebuild | Nov 7 (registration Oct 25) |

---

## 1. RSNA Knee: today's 4 submissions

**Facts behind the plan:**
- **The 14:42 submission (fork of `heliosli/rsna-knee-abnormality-detection`) failed** [M].
  - Error: `RuntimeError: Cannot resolve rsna-knee-compact-v1`.
  - heliosli retired that dataset at 13:50 UTC.
  - The whole Helios family (claimed 0.951–0.954: heliosli blend-gold, versia-7, matterhorn v2, rabari dual-engine, and others) can no longer run.
  - Do **not** fork anything that references `rsna-knee-compact-v1`.
- **The live public frontier is the "Apex" recipe at 0.950** [A, several authors]:
  - nartaa's CoAtNet R384 (OAI external data, new labels, SWA, anatomical mirror TTA; 0.949 alone);
  - goodpjw2008's ConvNeXt reader, blended per finding with rank weights.
  - 455 teams are tied at 0.950. Ties are broken by submission time, so a new 0.950 lands near rank ~720.
- **Board cut-offs** [M]: top-10 0.962, top-50 0.957, top-100 0.954, top-200 0.951.
  - Top-10 is not reachable from public assets. Every public "second model" sits 0.016–0.020 below R384. Only an independently trained family has added more than +0.001.

**Base for all four:** `ranjeet258/rsna-knee-apex-b-under-150`, hardened as follows:
- set `check=True` → `check=False` on the pip install and the decoder import;
- wrap every extra leg in try/except, with a fallback to R384;
- write rows in `sample_submission` order;
- pin the original environment (py3.12), or add a decoder smoke test on train DICOMs;
- keep T4×2 with internet off.

| Slot | Candidate | Recipe | Expected | Risk |
|---|---|---|---|---|
| K1 | **Apex exact** (bank it) | R384 mirror + CNXT 3 folds, per-target rank weights {Baker's .35, Contusion .30, MedOA .25, ACL .25, MedMen .20, MCL .20, LatMen/Fx/LatOA/PFOA .08, Effusion .04, Synovitis .03}, fallback R384 | 0.950 (floor 0.949) | low |
| K2 | Apex + C224 | inside the CoAtNet side: `rank(0.7·R384 + 0.3·C224)`, then K1 routing | 0.950 (0.949–0.951) | low |
| K3 | Apex + C224 + MaVIT | third leg `hengck23` MaVIT-288 at 0.12 rank weight; all-zero rows treated as missing; falls back to K2 | 0.948–0.951 | medium (first measurement of an independent family) |
| K4 | **Hold**, or a non-OAI hedge | keep for an infrastructure retry; otherwise a variant without OAI-trained weights (the OAI ruling is open: topics 747455, 747660) | – | – |

**How to submit the four:**
- Build one notebook that writes `submission.csv` plus `sub_k1..k3.csv`, then submit with `kaggle competitions submit -c rsna-knee-abnormality-detection -k <user>/<slug> -v <N> -f <file> -m ...`.
- `-f` with a non-default file is **unverified for this competition**: test it on K1 first. If it is refused, use a `VARIANT` constant and save one version per variant (about 5–10 GPU-min per commit run).
- The weekly GPU quota resets **Sat 2026-10-10 00:00 UTC**.

**Path to Oct 22**:
- **Oct 10–11:** train our own 5-fold ConvNeXt-T 2.5D teacher (224–320 px, 140 mm crop, slices sorted by position). Labels: report soft labels with OOF blended 50/50; fill silent synovitis from effusion (+0.11 on that column, 733932).
- **Oct 12–13:** pseudo-label round; a student on a different backbone or resolution.
- **By Oct 15:** consider a **team merge** with a team that already has its own 0.95x model. The disclosure thread named this as the highest-value single move; merge deadline is Oct 15.
- **Oct 17–19:** final student. Rehearse full inference on 1,300 training studies to catch crashes. Never use bf16 on a T4 (744230).
- **Blending:** equal-rank blends with Apex only; validate on OOF vs report labels, with gold-58 as a guard only.
- **Finals:**
  - Pick 1: best OOF-supported blend (or Apex exact unless beaten by ≥ 0.002).
  - Pick 2: a structurally different submission, free of OAI-trained weights unless the host clears OAI.
- **Expected outcome:** 0.951–0.955 (medal zone likely). Top-50 is about 15–25 %; top-10 is under 5 %.

---

## 2. CASMI 2026: today's 5 submissions

**Facts behind the plan:**
- **Code competition:**
  - The visible `test.parquet` is a 400-molecule placeholder.
  - The hidden rerun uses about 400 timsTOF molecules / about 1,500 spectra.
  - Limits: ≤ 9 h, no internet, output `submission.csv`.
- **Metric:** MRR@25 on the InChIKey first block after RemoveStereochemistry plus tautomer canonicalisation (RDKit 2026.03.3; the Oct 7 change is retroactive).
  - No partial credit.
  - Always fill 25 distinct keys.
  - Generated structures go at rank 1 only when calibrated above the DB top-1; otherwise at ranks 16–25.
- **Classes:** 1 = public spectra (~16–25 %), 2 = in PubChem/COCONUT (~45 %), 3 = novel (~35–39 %). A retrieval-only ceiling of about 0.61.
- **Leak thread (745715):** hidden IDs do not overlap the visible ones, so the leak is harmless. But **imranarif v44** and **nursrijan forensics v12** hard-code the visible IDs and are expected to error on the rerun. Do not use them.
- **Licences:**
  - Forbidden: NIST, METLIN, vendor libraries.
  - Avoid: rajrajak's MassBank (CC-BY-NC-SA) and the FRIGID weights (CC BY-NC).
  - Unresolved: `ahmedberatozer/*` datasets ("Other"), which every frontier notebook uses.
  - Penalty: the host can remove non-compliant teams from the LB (745841).
- **Public frontier:** one lineage at 0.433 [C]:
  - lineage: ahmed v4n → seyit → lszlst → bob v18 → obstacledeveloper V1.1 → haideptry v32;
  - ingredients: library search, analog propagation, fingerprint model, PubChem popularity, ICEBERG/GLACIER, LightGBM rankers.
- **Missing ingredient:** the **PubChem JOIN** from the huseyin branch (+0.011 [C]).
- **Noise:** the public LB covers about 130 molecules, so differences below about 0.016 are noise.

| Slot | Source | Change | Expected | Risk |
|---|---|---|---|---|
| C1 | `haideptry/0-350-casmi-2026-v32-ensemble` | none; pin its py3.12 image; attach the same 16 datasets | 0.43–0.44 | low (ICE/GL failures only warn) |
| C2 | C1 + huseyin `pc_join.py` | PubChem K=50; join when `library_max < 0.7`; inject with fallback; clear caches every 50 molecules; truncate PC lists back to 25; keep promotion (recipe: `CASMI_NOTEBOOKS.md` §6 S2) | **0.44–0.45** | moderate engineering (1–2 h), +30–60 min runtime |
| C3 | `huseyinemreaksoy/casmi26-v4n-fusion-pubchem-on-public-0-421` | add `'GL_MH_ONLY': True, 'POST_ICE_LAM': 0.0, 'FILL_25': True` to its `CFG.update` | 0.425–0.44 | trivial; the second family, the best private hedge |
| C4 | `amanatar/casmi26-v18` | none (CPU, no GPU quota) | 0.42–0.44 | close to the 9 h limit |
| C5 | `flexonafft/casmi26-public-baseline-adaptation-experiments`, or C2 without promotion if C2 is ready | none | 0.42–0.45 | high variance |

**Timing:**
- The scoring rerun takes about 3.5–4.5 h on T4, so start every submission well before 23:59 UTC; they queue in parallel.
- Commit runs on the placeholder take about 20 min each.
- Expected after today: public 0.43–0.45, moving rank from ~1550 to roughly 50–150.

**9-week path to top-10** (the bar is likely 0.48–0.50 by December):
1. **Weeks 1–2:** a three-mask offline validation harness; purge held-out structures from every library by metric key; use the 250 `enveda-np-examples` as a hidden-like set (more NH4/Na/formate adducts, different collision-energy ladder, 52 % with both polarities). Cache pipeline stages; tune JOIN, promotion and popularity offline.
2. **Weeks 2–4:** rebuild the core on licence-clean data:
   - retrain the fingerprint model on identity-disjoint folds with timsTOF weighting, adding the Enveda-180 Zenodo data;
   - retrain the ranker as LambdaRank cut at 25, with GLACIER/ICEBERG scores as features.
3. **Weeks 3–6:** class 2 isomer discrimination. 95 % of the class-2 gap is same-formula isomers ranked above the truth (742055). Tools: a cross-encoder reranker, CE-aware features, timsTOF-tuned GLACIER/ICEBERG/CFM-ID.
4. **Weeks 3–6:** class 3, the biggest upside (+0.02–0.05). A fingerprint+formula-conditioned generator retrained from MIT/Apache code, plus biotransformation edits of library analogs, inserted behind a confidence gate.
5. **Week 8:** stress-test on hidden-like inputs (unseen adducts, masses 157–1,159 Da) with runtime under 6 h; licence audit. Finals: one max-public and one different-family submission.

---

## 3. Solar Filament: plan (no submissions today)

- **Do not chase the board above 0.55.** The 0.55 cluster (59 teams, including this account's 09-10 entry) is one frozen CSV.
  - It comes from a private YOLOv8l-seg model, pasted base85-encoded into the public `lamhuy8904` notebook.
  - It very probably used the banned MAGFiLO test annotations. A full leak scores about 0.66.
  - The host will clean the LB and judges by rubric.
  - **Never select the 0.55 submission as a final, and never include it in the form package.**
- **Honest frontier:** OOF 0.46–0.49 → LB 0.40–0.41. Ours is OOF 0.42–0.44 → LB 0.37, exactly on the honest line.
- **Prize:** decided by a rubric: 70 % quantitative (PQ plus Dice, IoU and M:N distributions), 30 % qualitative. Requires a 4-page report, a public repo and a Google form by Nov 15.
- **Experiments, in order:**
  1. Decision layer (threshold forest + LightGBM hurdle scorer + exact DP at τ ≈ 0.26): +0.02–0.04 OOF, CPU only.
  2. GPU 3-architecture × 5-fold dense ensemble: +0.02–0.03.
  3. Native-resolution crop refiner: +0.01.
  4. Flat field + YOLO evidence: +0.005.
  5. Rubric hygiene.
- **Submission gate:** submit only when paired OOF Δ ≥ +0.005; the LB cannot resolve less than about 0.015.
- **Dates:** freeze Nov 4, report Nov 5–10, form Nov 12.

## 4. TrafficFlow: plan (no submissions today)

- **Corrections:**
  - Our best is **0.87249** (H11P, 09-28), not 0.86838.
  - The top of the raw board is pre-rebuild leak-era rows. Post-rebuild #1 is KTK 0.90303, #10 is 0.89037.
  - Nothing changed on the organizer side. The field doubled, leaders kept climbing, and we did not submit for 11 days.
- **Headroom:** almost all of it is in Task 2 (queue).

  | Task | Our score | Headroom |
  |---|---|---|
  | Queue (S_queue) | 0.788 | 0.064 weighted |
  | Task 1+3 | – | ~0.004–0.008 |
  | ODME | – | closed |

- **Lost work:** the newest code is on branch `claude/focused-allen-uzq8gr` @ 84503a8. The 29 Sep chain (H12P–H16P) was never submitted and must be rebuilt.
- **Next 5:** H15b (month bias) → H12P′ (blackout fullPD) → H13P′ → H14P′ (P4 mix) → H16P′ (v7+v11 ongoing).
- **Bigger bets:**
  - April adaptation of Task 2 (private-only gain);
  - onset label denoising;
  - a whole-day U-Net Task 1 member;
  - a corridor CNN for onset.
- **Expected:** public 0.877–0.883, plus 0.003–0.008 from the private line. The realistic goal is private top-10 among registered, reproducible teams.
- **Award requirements:** registration e-mail by **Oct 25** (to trafficflowbench@gmail.com, subject "Traffic Flow Bench Registration"); reproducible package + PDF by Nov 10.

---

## Actions only the user can take

1. **Rotate credentials.** Revoke the GitHub token and the Hugging Face token pasted in chat; rotate the Kaggle token when this work is done.
2. **TrafficFlow registration e-mail by Oct 25.** Without it there is no award, whatever the rank.
3. **Knee team merge decision by Oct 15**, if you want to merge with a team that has its own 0.95x model.
4. **Filament:** confirm with teammate koushikrudra that the 0.55 entry is never selected as a final.
5. **Approve today's submissions:** 4 knee (K1–K3, K4 held) and 5 CASMI (C1–C5). Each submission consumes a slot and GPU time, so the agents waited for your go-ahead.

## Calendar

| Date | Knee | CASMI | Filament | Traffic |
|---|---|---|---|---|
| Oct 9 (today) | K1–K3 (+K4 hold) | C1–C5 | rebuild OOF bank | E0 restore |
| Oct 10 (GPU reset) | own teacher training | build validation harness | forest + scorer + DP; GPU R34 | rebuild 29 Sep chain |
| Oct 15 | **merge deadline** | – | – | – |
| Oct 17 (GPU reset) | final student | ranker retrain | UNet++ / ensemble | E3–E5 |
| Oct 22 | **final deadline** | – | – | – |
| Oct 25 | – | – | – | **registration e-mail** |
| Nov 4 | – | – | **freeze** | freeze Nov 1–4 |
| Nov 7 | – | – | – | **final deadline 06:55** |
| Nov 10 | – | – | report/repo | **package + PDF** |
| Nov 12–15 | – | – | **form + deadline** | – |
| Dec 7 / 14 | – | **merge / final** | – | – |
