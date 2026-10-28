# Account B run log (one line per action, UTC)

- 2026-10-28 11:35 UTC  branch acctb-daily created from claude/dgx-umud-traffic-solar; plan in ACCOUNT_B_DAILY.md; scheduled tasks B1-B4 created.
- 2026-10-28 11:37 UTC  B4 start. Subs: NTL best 0.559, SB best 0.01702, Quantum none. LB tops: NTL 0.728, Quantum 26.14, SB 0.00514. 0 subs today in all three.
- 2026-10-28 11:40 UTC  Quantum: pushed kragglenote2forwork/qcc-public-best-repro-acctb v1 = credited fork of somejoint 22.63 notebook (runs official verifiers). Expected ~22.6; kill if verifier mismatch.
- 2026-10-28 11:45 UTC  NTL: fork of romantamrazov who-am-i prepared (ntl/k_whoami) but push refused: weekly GPU quota 30h reached. Retry after quota reset.
- 2026-10-28 11:48 UTC  SB: pushed kragglenote2forwork/sb-ztbest-fork-acctb v1 = credited fork of malyshevdanil probe-zt-best (public 0.01420). Expected ~0.0142 vs our 0.01702; kill if >0.017.
- 2026-10-28 11:53 UTC  Quantum: kernel v1 complete, official verifiers local GM 22.62971 (matches public). Submitted ref 57735662 (1/5 today). Score pending at 12:04.
- 2026-10-28 11:55 UTC  SB: kernel v1 complete, submission.parquet produced. Submitted ref 57735695 (1/5 today). Score pending (expected ~0.0142).
- 2026-10-28 12:04 UTC  B4 end. Next run: record both scores; Quantum next = beat public 22.63 on one task (chem lambda/T/Q or invsqrt qubits) verified locally; SB next = one-change ablation on zt-best (e.g. economic target 0.6->0.8 K); NTL waits for GPU quota reset.
- 2026-10-28 12:30 UTC  B1 start. UMUD: 5/5 used today (d29, best 0.19888 rank 3; top 0.18297/0.18704). TFB: 3/5 used today (N1 0.87759, N3 0.87969, Q1 0.88400 = new best; top 0.92817). Lessons recorded.
- 2026-10-28 12:35 UTC  B1 scout: no new public notebook beats our level (UMUD public best 0.3389; TFB public best 0.87141). To read: plarmuseau 'eleven anomalies of the data' (TFB, 10-27).
- 2026-10-28 12:36 UTC  B1 TFB: no submission. This cloud container (2 CPU, 7 GB) has no L1/Q1 artefacts (tfb-work backup = J2, 10-22); rebuilding needs 12 GB data + hours of T1/T2 chain. Any file built here would be J2-based (0.88203) -> no evidence-backed change on the best possible. Action needed: DGX session to run `trafficflow.loop backup` with Q1 zip + L1 state/queue so cloud runs can build on Q1.
- 2026-10-28 12:37 UTC  B1 end. Next: UMUD slots reopen 00:00 UTC (need a new measurement signal, not spread/weight); TFB 2 slots left today only if Q1 artefacts become available.
