# DEPENDENCIES.md: Push Order, Gates and Conflict Rules

Read with `README.md`. Both files live in the **repo root**, are committed by M1 as the very first commit, and are edited **only by M1**. Other members get them by pulling; the agent reads them from disk (PROMPT 0 tells it to). A **gate** is a milestone that one member must push (merged into `develop`) before others may build on it.

## 1. Gate list (in the order they must land)

| Gate | Owner | What lands | Needs first | Target date | Unblocks |
|---|---|---|---|---|---|
| **G0** | M1 | Scaffold, `types.py`, `interfaces.py`, `registry.py`, config, owners, CI, rules, `/gate-check` | none | Wed Oct 28 | **Everyone** |
| **G1** | M1 | `simulator.py` + `validate.py` | G0 | Fri Oct 30 | G3, G4, G8 |
| **G2** | M2 | `objectives.py` (f1, f2) | G0 | Fri Oct 30 | G3, G5, G6 |
| **G3** | M2 | `baselines.py` (sort_skill, fifo, greedy_nearest) | G1, G2 | Sun Nov 1 | G8, G10, G11 |
| **G7** | M4 | `pareto.py` + `plots.py` | G0 | Sun Nov 1 | G8, G12 |
| **G4** | M3 | `encoding.py` + `repair.py` | G0, G1 | Mon Nov 2 | G5, G6 |
| **G5** | M3 | `nsga2.py` | G2, G4 | Fri Nov 6 | G11, G12 |
| **G6** | M3 | `weighted.py` (SA + GA) | G2, G4 | Sat Nov 7 | G11, G12 |
| **G8** | M4 | `run_experiments.py` (baselines-only OK) | G1, G3, G7 | Fri Nov 6 | G12, G13 |
| **G9** | M2 | f3 + `metrics.py` + `benchmarks.py` | G3 | Sun Nov 8 | G12 |
| **G10** | M1 | `rolling_sim.py` + scenario presets | G3 | Mon Nov 9 | G12 (relaxation validation) |
| **G11** | M3 | `hybrid.py` + tuned config file | G3, G5, G6 | Wed Nov 11 | G12 |
| **G12** | M4 | Final experiments + `relaxation.py` + figures | G8, G9, G10, G11 | Fri Nov 13 | G13, G14 |
| **G13** | M2 | `conflict_study.py` + `stats.py` | G12 | Sat Nov 14 | G14 |
| **G14** | M4 | `docs/report.md` + slides outline | G12, G13 | Sun Nov 15 | G15 |
| **G15** | M1 | Final QA, `REPRODUCE.md`, tag `v1.0` | G14 | Mon Nov 16 | done |

## 2. Dependency graph

```
G0 (M1) ─┬─► G1 (M1) ─┬─► G3 (M2) ─┬─► G8 (M4) ──────────────────┐
         │            │            ├─► G10 (M1) ──────────────────┤
         │            │            ├─► G9 (M2) ───────────────────┤
         │            └─► G4 (M3) ─┤                              ├─► G12 (M4) ─► G13 (M2) ─► G14 (M4) ─► G15 (M1)
         ├─► G2 (M2) ─┬─► G3       ├─► G5 (M3) ─┐                 │
         │            └────────────┴─► G6 (M3) ─┴─► G11 (M3) ─────┘
         └─► G7 (M4) ─► G8
```

## 3. Who waits for whom

| Member | Cannot push before | Why | While waiting (parallel-safe work) |
|---|---|---|---|
| **M1** | nothing for G0/G1. G10 waits for **M2's G3** | The rolling sim needs a real baseline matcher | Use a dummy "match nothing" Matcher; write scenario presets and docs |
| **M2** | G2 waits for **M1's G0**. G3 waits for **M1's G1** | Baselines need the simulator + validator | Write objectives against the contract with hand-built fixtures |
| **M3** | G4 waits for **M1's G1**. G5/G6 wait for **M2's G2**. G11 waits for **M2's G3** | Repair needs `validate`; fitness needs `objectives`; seeding needs the baselines | Draft operators and the NSGA-II skeleton locally using the contract; do not push until the gate is on `develop` |
| **M4** | G7 waits for **M1's G0**. G8 waits for **M1's G1** and **M2's G3**. G12 waits for **M3's G11** | The runner needs instances and baselines; final runs need all algorithms | Build Pareto/plot utilities on synthetic arrays; make the runner skip missing algorithms |

**The two critical-path chains:**
1. `G0 → G1 → G3 → G11 → G12 → G14 → G15`. Any slip here slips the project.
2. `G0 → G2 → G5/G6 → G11`. M2's objectives are needed by M3 early. **M2 should finish G2 first in Week 1, before baselines.**

## 4. How a member consumes a gate

1. Gate owner announces "G_n merged into develop" in the team chat.
2. Dependent member runs:
   ```
   git fetch origin
   git checkout <your-branch>
   git rebase origin/develop
   pytest -q
   ```
3. Only then start the dependent part of the prompt. If tests fail on the rebased branch, tell the gate owner; do **not** patch their file.

## 5. Branches, PRs and merge order

- `main` = release only (M1 tags at G15). `develop` = integration branch.
- Each member works on `m<N>/g<gate>-<short-name>` (e.g. `m2/g3-baselines`).
- Before opening a PR run `/gate-check` in Antigravity (owner check, tests, rebase). Open the PR into `develop`.
- **Merge rule:** a PR is merged only if CI is green and all gates it depends on (table in section 1) are already on `develop`. M1 is the merge captain and merges in gate order. If two PRs are ready at once, merge the lower gate number first.
- Squash-merge, one gate per PR. Never force-push `develop`.

## 6. Conflict prevention rules

1. **One owner per file** (`config/owners.yaml`). `scripts/check_owner.py` fails the check if you touch someone else's file.
2. **Shared files are frozen:** `types.py`, `interfaces.py`, `registry.py`, `default.yaml`, `requirements.txt`. All packages are pre-installed in G0. If you need another package, put it in the PR description and M1 adds it.
3. **No shared edit points:** algorithms self-register via `@register`; the registry lazily imports modules, so adding an algorithm never edits a shared file. Tuned parameters go in your own file (`config/algorithms_tuned.yaml`).
4. **Contract changes:** write the request in `docs/CONTRACT_CHANGES.md` as a new section (`## Request by Mx, date`: what, why, impact). Only M1 edits the frozen files, after agreeing with all affected members. Append-only file → no merge conflicts if each member adds at the end; resolve by keeping both entries.
5. **Generated output** (`results/`) is written only by M4's runner. Everyone else reads it.
6. **Tests live in your own test files** (named in README section 1). Never put your tests into someone else's file.
7. **Never commit a file you did not author** just to "make it work". If a dependency is broken, report it to the owner.

## 7. Handling blockers (decision table)

| Situation | Action |
|---|---|
| Gate is late by more than 1 day | The owner posts an ETA; the dependent member continues with fixtures, then rebases. M1 may re-order merges. |
| Gate's tests fail after rebase | The consumer reports the failing test + traceback to the owner; the owner fixes in a new PR (`fix/g<n>`). |
| You need a field/function that doesn't exist | `docs/CONTRACT_CHANGES.md`, then wait for M1's approval. |
| Rebase conflict in a file you own | Keep your version of your own file; if the conflict is in someone else's file, you edited it by mistake: `git checkout origin/develop -- <file>`. |
| Two members both think they own a task | `config/owners.yaml` decides. M1 updates it only after both agree. |

## 8. Gate acceptance checks (the owner must pass these before announcing)

| Gate | Check |
|---|---|
| G0 | `pytest tests/test_types.py tests/test_registry.py`; `python scripts/check_owner.py 1`; CI green; all empty module files exist |
| G1 | Same seed gives identical instances; each constraint has a passing and a failing test |
| G2 | f1/f2 match hand-computed fixture values; `evaluate` is vectorized (1,000 evaluations in under 1 s on a medium instance) |
| G3 | All three baselines return feasible solutions on light/medium/heavy for 5 seeds |
| G4 | ≥ 99% of repaired random genomes feasible; operators keep permutations valid |
| G5 | Deterministic per seed; front non-empty; all front solutions feasible |
| G6 | Weight sweep returns monotone-ish trade-off (w=1 lowest f1, w=0 lowest f2) |
| G7 | Pareto/hypervolume tests on synthetic arrays pass (including a known-hypervolume case) |
| G8 | `python -m mm.run_experiments --quick` runs with baselines only and writes CSVs |
| G9 | Metrics tests pass; `results/baselines.csv` produced |
| G10 | Dummy matcher makes the queue grow; FIFO keeps it bounded in the "light" scenario |
| G11 | Hybrid runs on all scenarios; walkthrough states whether it beat plain NSGA-II (with numbers) |
| G12 | Full run completes; every figure + the relaxation rule table exist in `results/` |
| G13 | Stats tests run on `results/*.csv`; findings added to `docs/objectives.md` |
| G14 | Each number in the report traces to a `results/` file |
| G15 | Clean-clone install + full test suite + `--quick` run pass; tag `v1.0` created |

## 9. Status tracker (**M1 only** edits this file, after merging each gate)

Members do NOT tick boxes themselves (four people editing one file causes merge conflicts). Announce a landed gate in the team chat; M1 merges the PR and updates the box.

- [ ] G0  - [ ] G1  - [ ] G2  - [ ] G3  - [ ] G4  - [ ] G5  - [ ] G6  - [ ] G7
- [ ] G8  - [ ] G9  - [ ] G10 - [ ] G11 - [ ] G12 - [ ] G13 - [ ] G14 - [ ] G15

## 10. Daily 5-minute sync (suggested)

Each member answers: (1) Which gate did I land? (2) Which gate am I waiting on? (3) Do I need a contract change? Post answers in the team chat; M1 updates section 9.
