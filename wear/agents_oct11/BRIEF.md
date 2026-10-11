# Shared brief for WEAR bout-repair research agents (read-only on Kaggle: NEVER submit anything)

Competition: Kaggle 3rd-wear-dataset-challenge-hasca-2026. 19 classes (0 null, 1 jog, 2 jog-rot-arms, 3 jog-skip, 4 jog-side,
5 jog-butt, 6 str-triceps, 7 str-lunging, 8 str-shoulders, 9 str-hamstrings, 10 str-lumbar, 11 push-ups, 12 push-ups-cx,
13 sit-ups, 14 sit-ups-cx, 15 burpees, 16 lunges, 17 lunges-cx, 18 bench-dips). Metric macro F1. Test = 12,234 one-second
windows of 4 subjects (22-25), shuffled; each window carries ONE random limb (left/right arm/leg) accel (50x3) + 15 VideoMAE frames.
Each subject performs each of the 18 activities in one bout (~80-110 s, sometimes in a few sets), rest is null.
Current best public file: /home/user/wear_work/own_subs/sub_57035427.csv ("L11", public 0.93756). We need ~+0.0024 public.

KEY LESSON: the only big win (+0.00116) was a BOUT-LEVEL repair: in sbj_24 the jog-butt bout had 28 arm / 54 leg windows
(limbs are random, so true bouts are ~50/50; train truth 0.495+-0.055), its arm windows look like plain jogging to every
window model, and 11 jog-labelled arm windows whose link-neighbours (fork link matchings) mostly sat in the jog-butt bout
(neighbour share >= 0.36) were moved to jog-butt -> all 11 were right (a 12th-15th with share 0.25-0.33 were all wrong).
Single-window flips and generic rules are mostly noise. Every proposed change MUST be validated out-of-fold (OOF) on train.

Files (all under /home/user/wear_work; python3 with numpy/pandas/sklearn/scipy available; use `sys.path.insert(0,'/home/user/wear_work')`):
- Test meta: /home/user/wear_data/test/test_meta_data.csv (id, sbj_id, sensor_location). Test video: /home/user/wear_data/test/test_videomae_data.npy shape (N,768,15); per-window mean video: test_vmean.npy (N,768).
- Test inertial: /home/user/wear_data/test/test_inertial_data.npy (N,50,3).
- Fork pipeline (goodpjw-style, team runs) outputs: kout/wear-good-fork{,2}/final_probabilities.npz (oof (69326,19), test (12234,19)) = decoded probabilities.
  kout/wear-good-fork/keep3/stage.npz: oof_y, oof_sbj, oof_fold, oof_logp/test_logp (window-model log-probs), tab_S3/T, QA/QB etc.
  kout/wear-good-fork{,2}/keep3/links.npz: oof_succ/test_succ (8 Hungarian successor matchings each, -1 = none), scores.
  OOF tiles are in true time order per recording.
- OOF tile metadata: kout/wear-hanbat-gpu/keep/sim_meta.npz: rec, start, sensor (0 right_arm,1 right_leg,2 left_leg,3 left_arm), y, sbj, fold (same order as fork OOF).
  kout/wear-hanbat-gpu/keep/oof_emb.npy / test_emb.npy: video embeddings used by the pipeline (check shapes). feat_oof/feat_test npz: imu (261 feats), vpca (96), vmot.
- Other window models on test: kout/wear-fusion-full/test_full_s{0,1,2}.npy (probs), kout/wear-uec-k1/test_cnn8.npy, test_xcepvid.npy, kout/wear-hanbat-gpu/keep/test_logp_raw_imu_s{0,1}.npy (IMU-only logp), P_test.npy.
- Helper modules: limbfix.py / limbrepair.py (link-neighbour utilities: build_nb, nb_labels), meta.py (loads many things), band.py.
- Raw train data if needed: /home/user/wear_data/train/inertial_feat/*.csv, train/videomae_feat/*.npy.

How to validate: simulate your rule on fork OOF predictions (argmax of final_probabilities oof) using OOF links/meta, measure
(a) precision of moved windows (fraction whose new label equals oof_y) and (b) macro-F1 change. A rule is only useful if precision
is clearly > 0.6 and dF1 > 0 on BOTH fork and fork2 OOF. Then apply the rule to the test file L11 and list proposed changes.

Output: write your candidate file(s) to /home/user/wear_work/agents/<yourname>/ as CSV (id,target_feature), plus a REPORT.md with:
the rule, OOF precision and dF1 on fork and fork2, the exact test changes (window id, subject, from->to), and your honest confidence.
Do NOT call `kaggle competitions submit`. Do NOT modify files outside your agents/<yourname>/ folder.
