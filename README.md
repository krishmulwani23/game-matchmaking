# Matchmaking Optimization: Team README (4 Members, 3 Weeks)

**Project:** Multi-Objective Optimization of Queue-Based Multiplayer Matchmaking (Proposal 3).
At each matchmaking round, choose which queued players go into matches and which keep waiting. Minimize **skill imbalance** and **waiting time** (they conflict). Deliver a **Pareto front** and a **rule for relaxing the skill gap as waiting increases**.

**Timeline:** Week 1 = Oct 27 – Nov 2 | Week 2 = Nov 3 – Nov 9 | Week 3 = Nov 10 – Nov 16
**Remote:** `origin` | **Integration branch:** `develop` | **Release branch:** `main`

**Word "gate":** not a technical term. It is a checkpoint, e.g. "G1 = Krish's simulator is finished and merged into `develop`". Another member may only start work that depends on it after it is merged. The full list is in `DEPENDENCIES.md`.

**How to read this file:** find your name in section 1, do your "START HERE" in section 2, then work through your own prompts (sections 7-10) in order. Every prompt is complete: copy the whole code block into Antigravity and send. You do not need to paste anything else.

---

## 1. Team

| Member | Name | Role | Files they own (only edit these) |
|---|---|---|---|
| **M1** | **Krish** | Foundation, simulator, integration, CI | `README.md`, `DEPENDENCIES.md`, `requirements.txt`, `pyproject.toml`, `.gitignore`, `config/default.yaml`, `config/owners.yaml`, `scripts/`, `.github/`, `.agent/`, `src/mm/{types,interfaces,registry,config,simulator,validate,rolling_sim}.py`, `tests/__init__.py`, `tests/test_{types,registry,simulator,validate,rolling}*.py`, `docs/data.md`, `docs/REPRODUCE.md` |
| **M2** | **Sarvesh** | Objectives, baselines, metrics, conflict study | `src/mm/{objectives,baselines,metrics,benchmarks,stats,conflict_study}.py`, `tests/test_{objectives,baselines,metrics,stats}*.py`, `docs/objectives.md`, `results/baselines/`, `results/conflict/` |
| **M3** | **Vedant** | Optimization algorithms | `src/mm/{encoding,repair,nsga2,weighted,hybrid}.py`, `tests/test_{encoding,repair,nsga2,weighted,hybrid}*.py`, `docs/algorithms.md`, `config/algorithms_tuned.yaml` |
| **M4** | **RAM** | Pareto analysis, experiments, relaxation rule, report | `src/mm/{pareto,plots,run_experiments,relaxation}.py`, `tests/test_{pareto,plots,relaxation}*.py`, `results/raw/`, `results/fronts/`, `results/figures/`, `docs/report.md`, `docs/slides_outline.md` |

Everyone may **append** to `docs/CONTRACT_CHANGES.md` (add a new section at the end, never edit others' sections).
**Golden rule:** never edit a file owned by someone else. Need a change? Write it in `docs/CONTRACT_CHANGES.md`.

## 2. START HERE

### Krish (M1): do this once, by hand, before any prompt (about 10 minutes)
1. Create the GitHub repo `matchmaking-opt` and add Sarvesh, Vedant and RAM as collaborators.
2. On your machine:
   ```
   git clone <repo-url> matchmaking-opt
   cd matchmaking-opt
   git checkout -b develop
   # copy README.md and DEPENDENCIES.md into this folder (the repo root)
   git add README.md DEPENDENCIES.md
   git commit -m "Add project README and dependencies plan"
   git push -u origin develop
   ```
3. Tell the team in the chat: "Repo is ready, clone it and read README.md and DEPENDENCIES.md. Wait for G0."
4. Open the **repo root folder** in Antigravity (not a subfolder). Start a new conversation in **Planning mode** and send prompt **M1-W1-A**. When it finishes, follow section 6 to push and merge, then announce **G0 merged**.
5. In Antigravity, open Customizations → Rules and make sure `project-contract` is enabled (always on). The agent cannot toggle this itself.

### Sarvesh, Vedant, RAM (M2, M3, M4)
1. `git clone <repo-url>` and `git checkout develop`. Read `README.md` and `DEPENDENCIES.md`.
2. Open the **repo root folder** in Antigravity.
3. **Wait until Krish announces "G0 merged".** Then run `git fetch origin && git checkout develop && git pull origin develop`, create your branch (section 6), start a **new** Antigravity conversation in Planning mode and send your Week 1 prompt.
4. Before each later prompt, check `DEPENDENCIES.md` section 3 for which checkpoints it needs, make sure they are merged, pull, and start a **new** conversation.

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

## 5. Antigravity routine (every prompt)

1. New conversation, **Planning mode**, workspace = repo root.
2. Paste the complete prompt block. Read the **Implementation Plan** before approving: it must only touch your own files.
3. When it finishes, run the tests yourself: `pytest -q`.
4. Run `python scripts/check_owner.py <your number>` (for example `python scripts/check_owner.py 2` for Sarvesh). It must print PASS. (The `/gate-check` workflow in Antigravity does this plus a rebase.)
5. Push with the commands in section 6.

## 6. Git cheat sheet (remote is `origin`)

```
# start of each checkpoint (replace N and name)
git fetch origin
git checkout develop
git pull origin develop
git checkout -b mN/gX-name              # e.g. m2/g3-baselines

# after the agent finishes and tests pass
python scripts/check_owner.py N
git add -A
git commit -m "gX: short description"
git fetch origin
git rebase origin/develop
pytest -q
git push -u origin mN/gX-name
# then open a Pull Request into develop on GitHub; Krish merges when CI is green
```

If the rebase shows a conflict in a file you do not own, run `git checkout origin/develop -- <file>` for that file; you edited it by mistake.

---

## 7. Krish (M1): Foundation, Simulator, Integration

### WEEK 1

**M1-W1-A: Checkpoint G0 (blocks everyone; do this first)**
```
I am Krish (Member M1) on the repo "matchmaking-opt" (remote: origin). Before anything else,
read README.md and DEPENDENCIES.md in the repo root, and follow README section 4 "Working
rules for the agent" (sync check, own files only, plan first, tests, no git commands).
README.md and DEPENDENCIES.md are already committed on develop; do not modify them.

Task: build the project scaffold and the shared contract (Checkpoint G0).
1. Create the directory layout (src/mm/, tests/, config/, scripts/, docs/, results/,
   .agent/rules/, .agent/workflows/, .github/workflows/). Create an empty module file for EVERY
   member's modules listed in README section 1, each starting with the comment
   "# OWNER: Mx - do not edit unless you are Mx". Create docs/CONTRACT_CHANGES.md with a short
   header explaining it is append-only.
2. Implement src/mm/types.py (Player, Instance, Solution) EXACTLY as in README section 3,
   with role/ping optional. Add helpers Solution.empty(n) and Solution.n_matches().
3. Implement src/mm/interfaces.py (Matcher Protocol) and src/mm/registry.py (@register
   decorator, get(name), list_names()). The registry lazily imports mm.baselines, mm.nsga2,
   mm.weighted and mm.hybrid inside try/except ImportError so missing or empty modules never
   break it.
4. Write config/default.yaml (k, dt, max_wait, gap0, gap_alpha, queue sizes for light/medium/
   heavy, seeds, algorithms list, use_f3 flag) and src/mm/config.py that loads it into a
   dataclass.
5. Write requirements.txt: numpy pandas scipy matplotlib pymoo pytest pyyaml seaborn.
6. Write pyproject.toml (package `mm` under src/, python >=3.11) and configure pytest with
   pythonpath = ["src"] so `import mm` works without installing. Add tests/__init__.py and a
   .gitignore (venv, __pycache__, .pytest_cache, .env; do NOT ignore results/).
7. Write config/owners.yaml mapping members 1-4 to the exact paths in README section 1 (plus
   a "shared_append_only" entry for docs/CONTRACT_CHANGES.md), and scripts/check_owner.py
   MEMBER_NUMBER that fails (non-zero exit, prints offending files) if
   `git diff --name-only origin/develop` touches files the member does not own. It must also
   accept `--gate G0`, which lets member 1 touch all paths (scaffold commit only). For every
   other case the strict check applies.
8. Create .agent/rules/project-contract.md containing: (a) README section 3 verbatim,
   (b) README section 4 verbatim, (c) the ownership table from README section 1, (d) the
   checkpoint list (DEPENDENCIES section 1) and the "who waits for whom" table (DEPENDENCIES
   section 3). Add the frontmatter needed to make it an always-on workspace rule.
9. Create .agent/workflows/gate-check.md: a workflow that asks for the member number, runs
   scripts/check_owner.py, runs pytest on that member's tests, runs `git fetch origin` then
   reports whether the branch is behind origin/develop, and prints PASS or FAIL. (Do not run
   rebase for me; just tell me the command.)
10. Create .github/workflows/ci.yml running `pytest -q` on every pull request into develop.
11. Add tests/test_types.py and tests/test_registry.py (registry must survive missing modules).
12. In the walkthrough, give me exact commands to: create a venv, pip install -r
    requirements.txt, run pytest, run `python scripts/check_owner.py 1 --gate G0`, and the git
    commands to commit on branch m1/g0-scaffold and push to origin. Do not run them yourself.
```

**M1-W1-B: Checkpoint G1 (needs G0 merged)**
```
I am Krish (Member M1). Read README.md, DEPENDENCIES.md and .agent/rules/project-contract.md,
and follow README section 4 (sync check, own files only, plan first, tests, no git commands).

Task: implement the synthetic data simulator and the constraint validator (Checkpoint G1).
1. src/mm/simulator.py: make_instance(n_players, k, seed, scenario) with skill ~ Normal(1500,
   300), exponential per-player waits, Poisson arrivals, optional roles (tank/dps/support) and
   lognormal ping. Scenarios "light", "medium", "heavy" (about 24 / 100 / 400 players). Label
   everything SYNTHETIC in docstrings and in docs/data.md.
2. src/mm/validate.py: validate(inst, sol) -> list[str] and is_feasible(inst, sol) -> bool
   implementing every constraint in README section 3.
3. Tests: same seed gives identical instances; one feasible hand-built solution; one violating
   solution per constraint.
End with the walkthrough and the git commands (branch m1/g1-simulator).
```

### WEEK 2

**M1-W2-A: Checkpoint G10 (needs G3 baselines; until then test with a dummy matcher)**
```
I am Krish (Member M1). Read README.md, DEPENDENCIES.md and .agent/rules/project-contract.md,
and follow README section 4.

Task: src/mm/rolling_sim.py, a multi-round queue simulation (Checkpoint G10). Each round:
new players arrive (Poisson), a Matcher from the registry solves the current queue, matched
players leave, unmatched players get wait += dt. Return per-round logs (queue length, mean/max
wait, mean imbalance, matches formed). Add scenario presets to config/default.yaml (light,
medium, heavy, bursty arrivals). Tests: a dummy matcher that matches nothing (queue must grow)
and, if mm.baselines has real code on this branch, the "fifo" baseline (queue must stay
bounded in the light scenario). Branch name: m1/g10-rolling-sim.
```

**M1-W2-B**
```
I am Krish (Member M1). Read README.md, DEPENDENCIES.md and .agent/rules/project-contract.md,
and follow README section 4.

Task: hardening. 1) Extend .github/workflows/ci.yml with a linter (ruff) and a coverage
report. 2) Add a test that runs make_instance twice per scenario and checks identical output.
3) Write docs/data.md: how skill, arrivals, roles and ping are generated, a clear SYNTHETIC
disclaimer, and a note that real traces from the papers were not used (check dataset licences
before relying on any public data). Branch name: m1/w2-hardening.
```

### WEEK 3

**M1-W3: Checkpoint G15 (runs last; needs G14)**
```
I am Krish (Member M1). Read README.md, DEPENDENCIES.md and .agent/rules/project-contract.md,
and follow README section 4.

Task: final integration QA (Checkpoint G15). 1) Clean-clone check: describe the exact commands
for a fresh venv, install, and the full test suite, and run what you can in this workspace.
2) Run `python -m mm.run_experiments --config config/default.yaml --quick` and confirm it
completes. 3) Write docs/REPRODUCE.md with the exact command for every figure and table in
docs/report.md. 4) Fix lint problems in M1's files only; list problems in other members' files
as text for me to forward. 5) Prepare release notes and the `git tag v1.0` command. Branch
name: m1/g15-final.
```

---

## 8. Sarvesh (M2): Objectives, Baselines, Metrics, Conflict Study

### WEEK 1

**M2-W1-A: Checkpoint G2 (needs G0 merged; do this one first)**
```
I am Sarvesh (Member M2). Read README.md, DEPENDENCIES.md and
.agent/rules/project-contract.md, and follow README section 4 (sync check, own files only,
plan first, tests, no git commands).

Task: src/mm/objectives.py (Checkpoint G2). Implement f1_imbalance, f2_wait, f3_mismatch
(stub returning 0 when role/ping are missing) and evaluate(inst, sol) EXACTLY as defined in
README section 3. Vectorize with numpy: the genetic algorithms will call this tens of
thousands of times. Write docs/objectives.md with the maths, one worked example, and the
"why the objectives conflict" argument (strictly fair matches need similar-skill players who may
not be in the queue, so players wait longer). Tests: hand-built fixtures with known f1/f2
values (no simulator needed), the all-unmatched solution, and a perfectly balanced match.
Branch name: m2/g2-objectives.
```

**M2-W1-B: Checkpoint G3 (needs G1 and G2 merged)**
```
I am Sarvesh (Member M2). Read README.md, DEPENDENCIES.md and
.agent/rules/project-contract.md, and follow README section 4.

Task: src/mm/baselines.py (Checkpoint G3) with three Matchers registered via @register:
- "sort_skill": sort by skill, group sequentially into matches of 2k, split teams with a snake
  draft, leave the remainder in the queue.
- "fifo": take the longest-waiting players first, then balance the teams.
- "greedy_nearest": start from the longest-waiting unmatched player, greedily add the
  nearest-skill players.
All baselines must respect the hard constraints (force-match players near max_wait, respect
the relaxed skill gap) and return feasible solutions. Tests: validate.is_feasible on
simulator.make_instance for light/medium/heavy with 5 seeds. In the walkthrough, report the
f1/f2 of each baseline on the medium scenario. Branch name: m2/g3-baselines.
```

### WEEK 2

**M2-W2-A: Checkpoint G9 (needs G3)**
```
I am Sarvesh (Member M2). Read README.md, DEPENDENCIES.md and
.agent/rules/project-contract.md, and follow README section 4.

Task (Checkpoint G9): 1) Implement f3_mismatch properly (role coverage per team + ping
spread) behind cfg.use_f3. 2) src/mm/metrics.py: mean/median/p95/max wait of matched players,
queue length after the round, match rate, mean/max imbalance, a runtime wrapper, feasibility
rate. 3) src/mm/benchmarks.py: run all registered baselines on a list of (scenario, seed)
pairs and write a tidy DataFrame to results/baselines/baselines.csv. Tests for every metric on
fixtures. Branch name: m2/g9-metrics.
```

### WEEK 3

**M2-W3: Checkpoint G13 (needs G12: final results from RAM)**
```
I am Sarvesh (Member M2). Read README.md, DEPENDENCIES.md and
.agent/rules/project-contract.md, and follow README section 4.

Task (Checkpoint G13): 1) src/mm/conflict_study.py: for the three scenarios, sweep max_wait,
gap_alpha, k and queue size and show (a table plus plot data written to results/conflict/)
how baselines behave: sorting cannot trade fairness against waiting, which is the "why
optimization is required" argument. 2) src/mm/stats.py: Wilcoxon signed-rank test and effect
size between algorithms across seeds, reading results/raw/*.csv written by RAM (read only; do
not edit those files). 3) Add 5-8 bullet points with numbers under "Empirical conflict
evidence" in docs/objectives.md. Branch name: m2/g13-conflict-stats.
```

---

## 9. Vedant (M3): Optimization Algorithms

### WEEK 1

**M3-W1: Checkpoint G4 (needs G0 and G1 merged; you may start from the spec but push only after G1)**
```
I am Vedant (Member M3). Read README.md, DEPENDENCIES.md and
.agent/rules/project-contract.md, and follow README section 4 (sync check, own files only,
plan first, tests, no git commands).

Task: src/mm/encoding.py and src/mm/repair.py (Checkpoint G4).
1. Encoding: a genome representing a grouping, e.g. a permutation of player indices where
   consecutive blocks of 2k form matches (first k = team 0, next k = team 1) and the rest stay
   unmatched; define how the number of matches is chosen. Document the choice in
   docs/algorithms.md.
2. decode(inst, genome) -> Solution.
3. repair(inst, solution) -> Solution: force-include players near max_wait, swap out pairs that
   violate the relaxed skill gap, drop incomplete matches. After repair, validate.is_feasible
   must hold for at least 99% of random genomes (test this).
4. random_feasible(inst, rng), plus mutation and crossover operators (swap, insert, order
   crossover) that preserve valid permutations.
Tests: decode round-trip, repair feasibility rate, operators keep permutations valid.
Branch name: m3/g4-encoding.
```

### WEEK 2

**M3-W2-A: Checkpoint G5 (needs G2 and G4)**
```
I am Vedant (Member M3). Read README.md, DEPENDENCIES.md and
.agent/rules/project-contract.md, and follow README section 4.

Task: src/mm/nsga2.py (Checkpoint G5), registered as "nsga2" with @register. Use pymoo (a
custom Problem with the permutation operators from encoding.py) or a clean hand-written
NSGA-II. Objectives come from objectives.evaluate. Handle constraints via repair. Return the
non-dominated set as list[Solution]; population size, generations and seed come from config.
Tests: on a small instance the front must be non-empty and all-feasible, must beat or equal each
baseline in at least one objective, and a run must be deterministic for a given seed.
Branch name: m3/g5-nsga2.
```

**M3-W2-B: Checkpoint G6 (needs G2 and G4; can run in parallel with G5)**
```
I am Vedant (Member M3). Read README.md, DEPENDENCIES.md and
.agent/rules/project-contract.md, and follow README section 4.

Task: src/mm/weighted.py (Checkpoint G6) with two registered matchers, "sa_weighted"
(Simulated Annealing) and "ga_weighted" (single-objective GA). Both minimize
w*f1_norm + (1-w)*f2_norm, with normalisation constants taken from the baselines on the same
instance. Provide sweep_weights(inst, weights=np.linspace(0, 1, 11)) that runs the algorithm
once per weight and returns the collected solutions. Tests: w=1 favours minimum imbalance, w=0
favours minimum wait; deterministic per seed. Branch name: m3/g6-weighted.
```

### WEEK 3

**M3-W3: Checkpoint G11 (needs G3, G5, G6)**
```
I am Vedant (Member M3). Read README.md, DEPENDENCIES.md and
.agent/rules/project-contract.md, and follow README section 4.

Task: src/mm/hybrid.py (Checkpoint G11), registered as "hybrid_greedy_ls". 1) Seed the
initial population with the output of the greedy_nearest and fifo baselines (via
registry.get). 2) Add swap-based local search (swap two players across teams/matches if it
improves a scalarised objective and stays feasible) as a memetic step or as a post-pass on the
front. 3) Add a flag so RAM can run hybrid vs plain NSGA-II. 4) In docs/algorithms.md, write an
honest note: "A hybrid is justified only if experiments show a limitation of plain NSGA-II"; state
the limitation you observed with numbers, or say none was observed. 5) Tune population,
generations and mutation rate with a small grid and write the final values to
config/algorithms_tuned.yaml only (do not edit config/default.yaml). Branch name: m3/g11-hybrid.
```

---

## 10. RAM (M4): Pareto Analysis, Experiments, Relaxation Rule, Report

### WEEK 1

**M4-W1: Checkpoint G7 (needs only G0)**
```
I am RAM (Member M4). Read README.md, DEPENDENCIES.md and
.agent/rules/project-contract.md, and follow README section 4 (sync check, own files only,
plan first, tests, no git commands).

Task: src/mm/pareto.py and src/mm/plots.py (Checkpoint G7); neither may depend on the
algorithms. pareto.py: non_dominated(F) for an (n, m) array of minimization objectives,
hypervolume(F, ref) for 2 objectives (with a generic fallback for 3), a spacing/spread metric,
and representative_points(F) returning three labelled operating points: "very_fair" (min f1),
"fast" (min f2) and "balanced" (knee point, e.g. furthest from the line joining the extremes).
plots.py: pareto_plot(fronts: dict[name, F]), tradeoff_curve, and a boxplot of wait vs
imbalance across seeds. Test everything on synthetic arrays, including a known-hypervolume
case. Branch name: m4/g7-pareto.
```

### WEEK 2

**M4-W2: Checkpoint G8 (needs G1, G3, G7; algorithms are picked up automatically as they get merged)**
```
I am RAM (Member M4). Read README.md, DEPENDENCIES.md and
.agent/rules/project-contract.md, and follow README section 4.

Task: src/mm/run_experiments.py (Checkpoint G8). CLI options: --config, --quick,
--scenarios, --seeds, --algorithms. For every (scenario, seed, algorithm): build the instance
with simulator.make_instance, call registry.get(name).solve(inst), evaluate every returned
solution with objectives.evaluate, check feasibility with validate.is_feasible (record the
rate), measure runtime, and write tidy CSVs to results/raw/ and each front to results/fronts/.
The runner must skip algorithms not available yet (registry.list_names() lists only importable
ones) and print which were skipped. --quick means 2 seeds and the light scenario. Test the
runner with baselines only, and produce a first baseline-only figure set in results/figures/.
Branch name: m4/g8-runner.
```

### WEEK 3

**M4-W3-A: Checkpoint G12 (needs G8, G9, G10, G11)**
```
I am RAM (Member M4). Read README.md, DEPENDENCIES.md and
.agent/rules/project-contract.md, and follow README section 4.

Task (Checkpoint G12): 1) Run the full experiment: all scenarios, at least 10 seeds,
algorithms sort_skill, fifo, greedy_nearest, sa_weighted (weight sweep), ga_weighted, nsga2,
hybrid_greedy_ls. 2) Compute hypervolume and spread per algorithm and the three representative
operating points (very fair / balanced / fast) per scenario. 3) src/mm/relaxation.py: derive the
suggested rule for how much to relax the skill gap as waiting increases, e.g. fit gap_alpha (or
a piecewise rule) from the Pareto fronts, output a table and a one-line rule such as
`gap(w) = gap0 + alpha*w`, and validate it by re-running mm.rolling_sim with the suggested rule
vs a fixed gap. 4) Write final figures to results/figures/. Branch name: m4/g12-final-runs.
```

**M4-W3-B: Checkpoint G14 (needs G12 and G13)**
```
I am RAM (Member M4). Read README.md, DEPENDENCIES.md and
.agent/rules/project-contract.md, and follow README section 4.

Task (Checkpoint G14): write docs/report.md and docs/slides_outline.md from the actual files in
results/ and the docs of Sarvesh and Vedant (read only). Sections: Problem statement, Decision
variables, Objectives and constraints, Data (clearly SYNTHETIC), Why optimization is required,
Why the objectives conflict, Baselines, Algorithms (NSGA-II vs weighted SA/GA, hybrid), Results
(Pareto fronts, hypervolume, operating points), Suggested relaxation rule, Limitations,
Reproducibility. Every number must come from a file in results/, and the file must be named
next to the number. Do not invent results; write "not run" where something is missing.
Branch name: m4/g14-report.
```

---

## 11. Weekly summary

| Week | Krish (M1) | Sarvesh (M2) | Vedant (M3) | RAM (M4) |
|---|---|---|---|---|
| **1** (Oct 27–Nov 2) | G0 scaffold, G1 simulator + validator | G2 objectives, G3 baselines | G4 encoding + repair | G7 Pareto + plots |
| **2** (Nov 3–9) | G10 rolling sim, hardening | G9 f3 + metrics + benchmarks | G5 NSGA-II, G6 SA/GA | G8 experiment runner |
| **3** (Nov 10–16) | G15 final QA + tag | G13 conflict study + stats | G11 hybrid + tuning | G12 final runs + relaxation rule, G14 report |

## 12. Definition of done (every prompt)
- [ ] `pytest -q` passes
- [ ] `python scripts/check_owner.py <your number>` prints PASS
- [ ] Only your own files changed
- [ ] Branch rebased on `origin/develop`, pushed, Pull Request opened into `develop`
- [ ] Announced in the team chat so dependents can pull
