---
trigger: always_on
description: Project contract, working rules, ownership table, and dependencies
---

# Project Contract & Rules for matchmaking-opt

## 3. Shared contract (single source of truth)

**Data types (`types.py`, frozen after G0):**
- `Player(id:int, skill:float, wait:float, arrival:float, role:int|None=None, ping:float|None=None)`. `role` and `ping` exist from day 1 so nobody has to edit `types.py` later.
- `Instance(players:list[Player], k:int, dt:float, max_wait:float, gap0:float, gap_alpha:float, seed:int)`. A match has **2k players** (k per team). `dt` is the round length (the extra wait added to a player left unmatched).
- `Solution(match:np.ndarray[int], team:np.ndarray[int])`, both shape `(N,)`. `match[i] = -1` means player i stays in the queue; `match[i] = m >= 0` means player i is in match m. `team[i] in {0,1}` matters only when matched.

**Objectives (`objectives.py`, all minimized; definitions locked here):**
- `f1_imbalance`: mean over formed matches of `|mean skill of team0 - mean skill of team1|` (optionally plus within-team spread, behind a config flag).
- `f2_wait`: `sum over unmatched i of (wait_i + dt)`: the total waiting of players left in the queue, including the extra round.
- `f3_mismatch`: optional role/ping mismatch; returns 0 if the data has no role/ping.
- `evaluate(inst, sol) -> np.ndarray` returns `[f1, f2]`, or `[f1, f2, f3]` if `cfg.use_f3`.

**Constraints (`validate.py`):**
- Each match has exactly 2k players, k per team; each player is in at most one slot.
- If `wait_i + dt > max_wait`, player i **must** be matched.
- For every pair in a match: `|skill_i - skill_j| <= gap0 + gap_alpha * max(wait_i, wait_j)`.
- If ping is modelled, the ping gap in a match must stay under the configured limit.
- `validate(inst, sol) -> list[str]` returns violations (empty = feasible); `is_feasible(inst, sol) -> bool`.

**Interfaces (`interfaces.py`, `registry.py`):**
- `Matcher` protocol: `name: str` and `solve(inst, **kw) -> list[Solution]`. Single-solution baselines return a list of length 1; multi-objective algorithms return their non-dominated set.
- `@register("name")` registers a Matcher; `registry.get(name)` and `registry.list_names()` find them. Adding an algorithm never edits a shared file.
- `simulator.make_instance(n_players, k, seed, scenario) -> Instance`, with `skill ~ Normal(1500, 300)` (Elo-like) and Poisson arrivals. Data is **synthetic** and must be labelled so everywhere.

**Conventions:** Python 3.11; numpy, pandas, scipy, matplotlib, pymoo, pytest, pyyaml, seaborn; type hints; explicit seeds; every module has tests.

## 4. Working rules for the agent (these are also copied into `.agent/rules/project-contract.md`)

1. **Sync check first.** Run `git log --oneline -15` and `git status`. For every module the task depends on (see `DEPENDENCIES.md`), open the file and confirm it has real code, not just the `OWNER:` header. List which checkpoints are actually on this branch. If one is missing, tell the user and do only the parts that do not need it.
2. **Edit only files owned by the user's member number** (`config/owners.yaml`). Never touch other members' files.
3. **Never change** signatures or fields in `types.py`, `interfaces.py`, `registry.py`. If a change is needed, stop and write a request in `docs/CONTRACT_CHANGES.md`.
4. **Plan first.** Produce an Implementation Plan artifact, then implement.
5. **Tests.** Write pytest tests for everything and run them, showing the output.
6. **Missing dependency?** Use small hand-built fixtures based on the contract. Never create or edit the missing module.
7. **No git writes.** Do NOT run `git add`, `commit`, `push`, `merge` or `rebase`. Finish with a Walkthrough artifact: files changed, test results, and the exact git commands for the user to run.

---

## Ownership Table (README section 1)

| Member | Name | Role | Files they own (only edit these) |
|---|---|---|---|
| **M1** | **Krish** | Foundation, simulator, integration, CI | `README.md`, `DEPENDENCIES.md`, `requirements.txt`, `pyproject.toml`, `.gitignore`, `config/default.yaml`, `config/owners.yaml`, `scripts/`, `.github/`, `.agent/`, `src/mm/{types,interfaces,registry,config,simulator,validate,rolling_sim}.py`, `tests/__init__.py`, `tests/test_{types,registry,simulator,validate,rolling}*.py`, `docs/data.md`, `docs/REPRODUCE.md` |
| **M2** | **Sarvesh** | Objectives, baselines, metrics, conflict study | `src/mm/{objectives,baselines,metrics,benchmarks,stats,conflict_study}.py`, `tests/test_{objectives,baselines,metrics,stats}*.py`, `docs/objectives.md`, `results/baselines/`, `results/conflict/` |
| **M3** | **Vedant** | Optimization algorithms | `src/mm/{encoding,repair,nsga2,weighted,hybrid}.py`, `tests/test_{encoding,repair,nsga2,weighted,hybrid}*.py`, `docs/algorithms.md`, `config/algorithms_tuned.yaml` |
| **M4** | **RAM** | Pareto analysis, experiments, relaxation rule, report | `src/mm/{pareto,plots,run_experiments,relaxation}.py`, `tests/test_{pareto,plots,relaxation}*.py`, `results/raw/`, `results/fronts/`, `results/figures/`, `docs/report.md`, `docs/slides_outline.md` |

Everyone may **append** to `docs/CONTRACT_CHANGES.md` (add a new section at the end, never edit others' sections).
**Golden rule:** never edit a file owned by someone else. Need a change? Write it in `docs/CONTRACT_CHANGES.md`.

---

## Gate List (DEPENDENCIES section 1)

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

---

## Who Waits for Whom (DEPENDENCIES section 3)

| Member | Cannot push before | Why | While waiting (safe work) |
|---|---|---|---|
| **Krish** | G10 waits for **Sarvesh's G3** | The rolling simulation needs a real baseline matcher | Use a dummy "match nothing" matcher; write scenario presets and docs |
| **Sarvesh** | G2 waits for **Krish's G0**; G3 waits for **Krish's G1** | Baselines need the simulator and validator | Write objectives against the contract with hand-built fixtures |
| **Vedant** | G4 waits for **Krish's G1**; G5/G6 wait for **Sarvesh's G2**; G11 waits for **Sarvesh's G3** | Repair needs `validate`; fitness needs `objectives`; seeding needs the baselines | Draft operators and the NSGA-II skeleton locally from the contract; do not push until the gate is merged |
| **RAM** | G7 waits for **Krish's G0**; G8 waits for **Krish's G1** and **Sarvesh's G3**; G12 waits for **Vedant's G11** | The runner needs instances and baselines; final runs need all algorithms | Build Pareto/plot utilities on synthetic arrays; make the runner skip missing algorithms |
