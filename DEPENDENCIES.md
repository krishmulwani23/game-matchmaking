# DEPENDENCIES.md: Push Order, Checkpoints ("Gates") and Conflict Rules

Read together with `README.md`. Both files live in the **repo root**, are committed first by Krish, and are edited **only by Krish (M1)**.

**What is a gate?** A checkpoint: a piece of work one member must finish and get merged into `develop` before another member may start the part that depends on it. "G1" is just the label of checkpoint number 1.

**Team:** Krish = M1, Sarvesh = M2, Vedant = M3, RAM = M4. Remote = `origin`. Integration branch = `develop`.

## 1. Gate list (in the order they must land)

| Gate | Owner | What lands | Needs first | Target date | Unblocks |
|---|---|---|---|---|---|
| **G0** | Krish | Scaffold, `types.py`, `interfaces.py`, `registry.py`, config, owners, CI, rules, `/gate-check` | none | Wed Oct 28 | **Everyone** |
| **G1** | Krish | `simulator.py` + `validate.py` | G0 | Fri Oct 30 | G3, G4, G8 |
| **G2** | Sarvesh | `objectives.py` (f1, f2) | G0 | Fri Oct 30 | G3, G5, G6 |
| **G3** | Sarvesh | `baselines.py` (sort_skill, fifo, greedy_nearest) | G1, G2 | Sun Nov 1 | G8, G10, G11 |
| **G7** | RAM | `pareto.py` + `plots.py` | G0 | Sun Nov 1 | G8, G12 |
| **G4** | Vedant | `encoding.py` + `repair.py` | G0, G1 | Mon Nov 2 | G5, G6 |
| **G5** | Vedant | `nsga2.py` | G2, G4 | Fri Nov 6 | G11, G12 |
| **G6** | Vedant | `weighted.py` (SA + GA) | G2, G4 | Sat Nov 7 | G11, G12 |
| **G8** | RAM | `run_experiments.py` (baselines-only is fine) | G1, G3, G7 | Fri Nov 6 | G12, G13 |
| **G9** | Sarvesh | f3 + `metrics.py` + `benchmarks.py` | G3 | Sun Nov 8 | G12 |
| **G10** | Krish | `rolling_sim.py` + scenario presets | G3 | Mon Nov 9 | G12 |
| **G11** | Vedant | `hybrid.py` + `config/algorithms_tuned.yaml` | G3, G5, G6 | Wed Nov 11 | G12 |
| **G12** | RAM | Final experiments + `relaxation.py` + figures | G8, G9, G10, G11 | Fri Nov 13 | G13, G14 |
| **G13** | Sarvesh | `conflict_study.py` + `stats.py` | G12 | Sat Nov 14 | G14 |
| **G14** | RAM | `docs/report.md` + slides outline | G12, G13 | Sun Nov 15 | G15 |
| **G15** | Krish | Final QA, `REPRODUCE.md`, tag `v1.0` | G14 | Mon Nov 16 | done |

## 2. Dependency graph

```
G0 (Krish) ─┬─► G1 (Krish) ─┬─► G3 (Sarvesh) ─┬─► G8 (RAM) ─────────────────┐
            │               │                 ├─► G10 (Krish) ───────────────┤
            │               │                 ├─► G9 (Sarvesh) ──────────────┤
            │               └─► G4 (Vedant) ──┤                              ├─► G12 (RAM) ─► G13 (Sarvesh) ─► G14 (RAM) ─► G15 (Krish)
            ├─► G2 (Sarvesh) ─┬─► G3          ├─► G5 (Vedant) ─┐             │
            │                 └───────────────┴─► G6 (Vedant) ─┴─► G11 (Vedant) ─┘
            └─► G7 (RAM) ─► G8
```

## 3. Who waits for whom

| Member | Cannot push before | Why | While waiting (safe work) |
|---|---|---|---|
| **Krish** | G10 waits for **Sarvesh's G3** | The rolling simulation needs a real baseline matcher | Use a dummy "match nothing" matcher; write scenario presets and docs |
| **Sarvesh** | G2 waits for **Krish's G0**; G3 waits for **Krish's G1** | Baselines need the simulator and validator | Write objectives against the contract with hand-built fixtures |
| **Vedant** | G4 waits for **Krish's G1**; G5/G6 wait for **Sarvesh's G2**; G11 waits for **Sarvesh's G3** | Repair needs `validate`; fitness needs `objectives`; seeding needs the baselines | Draft operators and the NSGA-II skeleton locally from the contract; do not push until the gate is merged |
| **RAM** | G7 waits for **Krish's G0**; G8 waits for **Krish's G1** and **Sarvesh's G3**; G12 waits for **Vedant's G11** | The runner needs instances and baselines; final runs need all algorithms | Build Pareto/plot utilities on synthetic arrays; make the runner skip missing algorithms |

**Critical path (any slip here slips the project):** `G0 → G1 → G3 → G11 → G12 → G14 → G15`.
Second chain: `G0 → G2 → G5/G6 → G11`. **Sarvesh should do G2 (objectives) before G3 (baselines) in Week 1**, because Vedant needs objectives early.

## 4. How to start a task that depends on a gate

1. The gate owner announces "G_n merged into develop" in the team chat. Krish merges the PR only after CI is green.
2. You run:
   ```
   git fetch origin
   git checkout develop
   git pull origin develop
   git checkout -b mN/gX-name
   pytest -q
   ```
3. Start a **new** Antigravity conversation (so the agent does not use stale context) and send your prompt. The agent's first step is a sync check that confirms the needed files contain real code.
4. If `pytest -q` fails right after the pull, tell the gate owner. Do not patch their file.

## 5. Branches, Pull Requests, merge order

- `main` is for releases only (Krish tags `v1.0` at G15). `develop` is the integration branch.
- Branch name: `m<number>/g<gate>-<short-name>` (e.g. `m2/g3-baselines`). One gate per Pull Request.
- Before opening the PR: `python scripts/check_owner.py <your number>`, `git fetch origin`, `git rebase origin/develop`, `pytest -q`, `git push -u origin <branch>`.
- **Merge rule:** Krish merges a PR only if CI is green and every gate it depends on (section 1) is already on `develop`. If two PRs are ready, the lower gate number goes first. Squash-merge. Nobody force-pushes `develop`.
- The agent never runs git commands itself; the member does (see README section 6).

## 6. Conflict prevention rules

1. **One owner per file** (`config/owners.yaml`; `scripts/check_owner.py` fails if you touch someone else's file). Krish's G0 commit is the only exception (`--gate G0`).
2. **Frozen files:** `types.py`, `interfaces.py`, `registry.py`, `config/default.yaml`, `requirements.txt`. All packages are installed from G0. If you need another package, say so in the PR description and Krish adds it.
3. **No shared edit points:** algorithms self-register via `@register`, and the registry lazily imports modules, so adding an algorithm never edits a shared file. Tuned parameters go into your own file (`config/algorithms_tuned.yaml`).
4. **Contract changes:** append a section to `docs/CONTRACT_CHANGES.md` (`## Request by <name>, <date>`: what, why, impact). Only Krish edits the frozen files, after agreement with everyone affected. If two people append at once and Git conflicts, keep both entries.
5. **Generated output** is written by one owner only: RAM → `results/raw`, `results/fronts`, `results/figures`; Sarvesh → `results/baselines`, `results/conflict`. Others only read.
6. **Tests go in your own test files** (names in README section 1).
7. **Never commit a file you did not author** just to make something work. Report broken dependencies to the owner.

## 7. Blockers: what to do

| Situation | Action |
|---|---|
| Gate is more than 1 day late | The owner posts an ETA; the waiting member continues with fixtures, then rebases. Krish may re-order merges. |
| Tests fail after pulling a gate | Report the failing test and traceback to the owner; the owner fixes it in a new PR (`m<N>/fix-g<n>`). |
| You need a field or function that does not exist | Write it in `docs/CONTRACT_CHANGES.md` and wait for Krish. |
| Rebase conflict in a file you own | Keep your version. |
| Rebase conflict in a file you do not own | You changed it by mistake: `git checkout origin/develop -- <file>`. |
| Two people think they own the same task | `config/owners.yaml` decides; Krish updates it only after both agree. |

## 8. Gate acceptance checks (owner must pass before announcing)

| Gate | Check |
|---|---|
| G0 | `pytest tests/test_types.py tests/test_registry.py`; `python scripts/check_owner.py 1 --gate G0`; CI green; every module file exists with its `OWNER:` header |
| G1 | Same seed gives identical instances; each constraint has a passing and a failing test |
| G2 | f1/f2 match hand-computed fixtures; `evaluate` is vectorized (1,000 evaluations in under 1 s on a medium instance) |
| G3 | All three baselines return feasible solutions on light/medium/heavy for 5 seeds |
| G4 | At least 99% of repaired random genomes are feasible; operators keep permutations valid |
| G5 | Deterministic per seed; front non-empty; all front solutions feasible |
| G6 | Weight sweep behaves as expected (w=1 gives the lowest f1, w=0 the lowest f2) |
| G7 | Pareto/hypervolume tests pass on synthetic arrays, including a known-hypervolume case |
| G8 | `python -m mm.run_experiments --quick` runs with baselines only and writes CSVs |
| G9 | Metrics tests pass; `results/baselines/baselines.csv` exists |
| G10 | Dummy matcher makes the queue grow; FIFO keeps it bounded in the light scenario |
| G11 | Hybrid runs on all scenarios; walkthrough states with numbers whether it beat plain NSGA-II |
| G12 | Full run completes; all figures and the relaxation-rule table exist in `results/` |
| G13 | Stats tests run on `results/raw/*.csv`; findings added to `docs/objectives.md` |
| G14 | Every number in the report traces to a `results/` file |
| G15 | Clean-clone install, full test suite and `--quick` run all pass; tag `v1.0` created |

## 9. Status tracker (**only Krish edits this**, after merging each gate)

Members do not tick boxes themselves (four people editing one file causes merge conflicts). Announce a landed gate in the team chat; Krish merges the PR and updates the box.

- [ ] G0  - [ ] G1  - [ ] G2  - [ ] G3  - [ ] G4  - [ ] G5  - [ ] G6  - [ ] G7
- [ ] G8  - [ ] G9  - [ ] G10 - [ ] G11 - [ ] G12 - [ ] G13 - [ ] G14 - [ ] G15

## 10. Daily 5-minute sync (suggested)

Each member posts in the chat: (1) Which gate did I land? (2) Which gate am I waiting on? (3) Do I need a contract change? Krish updates section 9.
