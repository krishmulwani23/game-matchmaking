# Matchmaking Optimization: Team README (4 Members, 3 Weeks)

**Project:** Multi-Objective Optimization of Queue-Based Multiplayer Matchmaking (Proposal 3)
**Goal:** At each matchmaking round, choose which queued players to place in matches and which to keep waiting. Minimize **skill imbalance** and **waiting time**, which conflict. Deliver a **Pareto front** plus a **rule for relaxing the skill gap as waiting increases**.
**Timeline:** Week 1 = Oct 27 – Nov 2 | Week 2 = Nov 3 – Nov 9 | Week 3 = Nov 10 – Nov 16
**Read next:** `DEPENDENCIES.md` (who must push before whom). **Do not skip it.**

---

## 1. Roles at a glance

| Member | Role | Owns (only edit these) |
|---|---|---|
| **M1** | Foundation, simulator, integration, CI | `requirements.txt`, `config/`, `scripts/`, `.github/`, `.agent/`, `src/mm/{types,interfaces,registry,config,simulator,validate,rolling_sim}.py`, `tests/test_{types,simulator,validate,rolling}*.py`, `docs/data.md` |
| **M2** | Objectives, baselines, metrics, conflict study | `src/mm/{objectives,baselines,metrics,benchmarks,stats,conflict_study}.py`, `tests/test_{objectives,baselines,metrics,stats}*.py`, `docs/objectives.md` |
| **M3** | Optimization algorithms | `src/mm/{encoding,repair,nsga2,weighted,hybrid}.py`, `tests/test_{encoding,repair,nsga2,weighted,hybrid}*.py`, `docs/algorithms.md` |
| **M4** | Pareto analysis, experiments, relaxation rule, report | `src/mm/{pareto,plots,run_experiments,relaxation}.py`, `tests/test_{pareto,plots,relaxation}*.py`, `results/`, `docs/report.md`, `docs/slides_outline.md` |

**Golden rule:** nobody edits another member's files. If you need a change, raise it in `docs/CONTRACT_CHANGES.md` (see DEPENDENCIES.md).

## 2. Repo layout (created by M1 in Gate G0)

```
matchmaking-opt/
  README.md  DEPENDENCIES.md  requirements.txt
  .agent/rules/project-contract.md      # Antigravity workspace rule (the shared contract below)
  .agent/workflows/gate-check.md        # /gate-check workflow
  .github/workflows/ci.yml
  config/default.yaml  config/owners.yaml
  scripts/check_owner.py
  src/mm/   (modules listed above)
  tests/    docs/    results/
```

## 3. Shared contract (single source of truth; M1 copies this into `.agent/rules/project-contract.md`)

**Data types (`types.py`, frozen after G0):**
- `Player(id:int, skill:float, wait:float, arrival:float, role:int|None=None, ping:float|None=None)`. `role` and `ping` exist from day 1 so no one edits `types.py` later.
- `Instance(players:list[Player], k:int, dt:float, max_wait:float, gap0:float, gap_alpha:float, seed:int)`. Matches have **2k players**, with k per team. `dt` is the round length (extra wait added to a player left unmatched).
- `Solution(match:np.ndarray[int], team:np.ndarray[int])`, both of shape `(N,)`. `match[i] = -1` means player i stays in the queue; `match[i] = m >= 0` means player i is in match m. `team[i] in {0,1}` is meaningful only when matched.

**Objectives (`objectives.py`, minimized; the definitions are locked here):**
- `f1_imbalance(inst, sol)`: mean over formed matches of `|mean skill team0 - mean skill team1|` (optionally plus within-team spread, controlled by a config flag).
- `f2_wait(inst, sol)`: `sum over unmatched i of (wait_i + dt)`. It is the total waiting of players left in the queue, including the extra round of waiting.
- `f3_mismatch(inst, sol)`: optional role/ping mismatch. It returns 0 if the data lacks role/ping.
- `evaluate(inst, sol) -> np.ndarray` returns `[f1, f2]`, or `[f1, f2, f3]` if `cfg.use_f3`.

**Constraints (`validate.py`):**
- Each match has exactly 2k players, k per team. Each player is in at most one slot.
- If `wait_i + dt > max_wait`, player i **must** be matched.
- For every pair in a match: `|skill_i - skill_j| <= gap0 + gap_alpha * max(wait_i, wait_j)`.
- If ping is modelled, the ping gap must stay under the configured limit.
- `validate(inst, sol) -> list[str]` returns violations; an empty list means feasible.

**Interfaces (`interfaces.py`):**
- `class Matcher(Protocol): name: str; def solve(self, inst: Instance, **kw) -> list[Solution]` returns a list. Single-solution baselines return a list of length 1. Multi-objective algorithms return their non-dominated set.
- `@register("name")` (in `registry.py`) registers a Matcher factory. The runner finds algorithms by name. No shared file is edited when adding an algorithm.
- `simulator.make_instance(n_players, k, seed, scenario) -> Instance`, with `skill ~ Normal(1500, 300)` (Elo-like) and Poisson arrivals. Data is **clearly labelled synthetic**.

**Global conventions:** Python 3.11, numpy/pandas/scipy/matplotlib/pymoo/pytest/pyyaml, type hints, seeds always explicit, no randomness without `seed`. Every module ships with tests.

---

## 4. How to use Antigravity for this project (all members)

0. `README.md` and `DEPENDENCIES.md` live in the **repo root**, committed by M1 as the very first commit (before G0). Everyone clones/pulls to get them; do not attach them to chats. **M1 is the only person who edits these two files.** Always open the **repo root** (not a subfolder like `src/`) as the Antigravity workspace so the agent can find them. After every `git pull`/rebase, start a **new** Antigravity conversation so the agent does not work from stale context.
1. Open the repo folder in Antigravity. Open **Agent Manager** and create **one agent/conversation per prompt** below. Do not reuse one giant conversation across weeks.
2. Use **Planning mode**. Review the **Implementation Plan** artifact before approving. Check that it touches only your owned files.
3. When the agent finishes, read the **Walkthrough** artifact and run the tests yourself in the terminal.
4. Run `/gate-check` (workflow created by M1) before every push.
5. **You** run `git push` and open the PR, not the agent. See `DEPENDENCIES.md` for the order.

### PROMPT 0: paste this at the start of EVERY Antigravity session (replace `X`)

```
You are Member MX on the repo "matchmaking-opt". Before doing anything, read
.agent/rules/project-contract.md, README.md and DEPENDENCIES.md.
Step 0 (sync check, do this before planning): run `git log --oneline -15` and `git status`.
For every module my task depends on (see DEPENDENCIES.md), open the file and confirm it
contains real code, not just the OWNER header. List which gates are actually on this branch.
If a required gate is missing, tell me and work only on the parts that do not need it.
Rules:
1. Edit ONLY files owned by MX (see config/owners.yaml). Never touch other members' files.
2. Never change signatures or fields in types.py / interfaces.py / registry.py. If you need a
   change, STOP and write the request into docs/CONTRACT_CHANGES.md instead.
3. Use Planning mode: produce an Implementation Plan artifact first, then implement.
4. Every module needs pytest tests. Run them and show the output.
5. If a dependency (another member's module) is not on the branch yet, code against the
   contract with small hand-built fixtures; do not create or edit the missing module.
6. Finish with a Walkthrough artifact: files changed, test results, and the exact git
   commands for me to run. Do NOT run git push yourself.
```

---

## 5. Member 1: Foundation, Simulator, Integration

### WEEK 1

**M1-W1-A (Day 1 → Gate G0, blocks everyone. Do this first.)**
```
[Paste PROMPT 0 with X=1]
Task: create the project scaffold and the shared contract.
0. FIRST COMMIT (before anything else): make sure README.md and DEPENDENCIES.md are in the repo
   root, on the `develop` branch, and tell me to push that commit so the team can clone it.
   Do not modify their content.
1. Create the directory layout from README section 2, including empty module files for ALL
   members' modules, each with a header comment "OWNER: Mx - do not edit unless you are Mx".
2. Implement src/mm/types.py (Player, Instance, Solution) EXACTLY as in README section 3.
   Include role/ping as optional fields. Add helpers: Solution.empty(n), Solution.n_matches().
3. Implement src/mm/interfaces.py (Matcher Protocol) and src/mm/registry.py (@register
   decorator + get(name) + list_names()). The registry must lazily import mm.baselines,
   mm.nsga2, mm.weighted, mm.hybrid inside try/except ImportError so missing modules never break it.
4. Write config/default.yaml (k, dt, max_wait, gap0, gap_alpha, queue sizes, seeds, algorithms
   list, use_f3 flag) and src/mm/config.py to load it into a dataclass.
5. Write requirements.txt with: numpy pandas scipy matplotlib pymoo pytest pyyaml seaborn.
6. Write config/owners.yaml mapping each member to the exact paths in README section 1, and
   scripts/check_owner.py MEMBER that fails if `git diff --name-only origin/develop` touches
   files not owned by that member.
7. Create .agent/rules/project-contract.md (set it as an always-on workspace rule in
   Antigravity: Customizations -> Rules) containing: (a) README section 3 verbatim, (b) the
   gate list table (section 1) and the "who waits for whom" table (section 3) from
   DEPENDENCIES.md, (c) the ownership table from README section 1, and (d) the sync-check
   step from PROMPT 0. This way the agent always knows the push order even if it does not
   open the full files.
8. Create .agent/workflows/gate-check.md: a workflow that asks for the member id, runs
   scripts/check_owner.py, runs pytest on that member's tests, runs `git pull --rebase origin
   develop`, and prints PASS/FAIL.
9. Create .github/workflows/ci.yml running pytest on every PR.
10. Add tests/test_types.py and tests/test_registry.py.
```

**M1-W1-B (Days 2-4 → Gate G1)**
```
[Paste PROMPT 0 with X=1]
Task: implement the synthetic data simulator and constraint validator.
1. src/mm/simulator.py: make_instance(n_players, k, seed, scenario) with skill ~ Normal(1500, 300),
   per-player wait drawn from an exponential distribution, Poisson arrivals, optional roles
   (e.g. tank/dps/support) and ping (lognormal). Scenarios: "light", "medium", "heavy" queue load
   (e.g. 24 / 100 / 400 players). Everything labelled SYNTHETIC in docstrings and docs/data.md.
2. src/mm/validate.py: validate(inst, sol) -> list[str] implementing every constraint in README
   section 3 (match sizes, equal teams, at most one slot, forced matching of players near
   max_wait, wait-relaxed skill gap, optional ping gap). Also is_feasible(inst, sol) -> bool.
3. Tests: reproducibility under the same seed, a hand-built feasible solution, and one
   violating solution per constraint.
Ping Gate G1 only when tests pass.
```

### WEEK 2

**M1-W2-A (→ Gate G10; requires G3 baselines. Until then test with a dummy matcher.)**
```
[Paste PROMPT 0 with X=1]
Task: src/mm/rolling_sim.py, a multi-round queue simulation. Each round: new players arrive
(Poisson), a Matcher from the registry solves the current queue, matched players leave, unmatched
players have wait += dt. Return per-round logs (queue length, mean/max wait, mean imbalance,
matches formed). Add scenario presets to config/ (light, medium, heavy, bursty arrivals).
Tests with a dummy matcher that matches nothing (queue must grow) and with the FIFO baseline
(once mm.baselines exists on develop).
```

**M1-W2-B**
```
[Paste PROMPT 0 with X=1]
Task: hardening. 1) Extend CI with ruff/flake8 and a coverage report. 2) Add a seeding audit:
a test that runs make_instance twice per scenario and checks bit-identical output. 3) Write
docs/data.md: how skill, arrivals, roles and ping are generated, plus a clear SYNTHETIC
disclaimer, plus a note that the papers' real traces were NOT used and why (check dataset
licensing before relying on any public data).
```

### WEEK 3

**M1-W3 (→ Gate G15, runs last; requires G14)**
```
[Paste PROMPT 0 with X=1]
Task: final integration QA. 1) From a clean clone, create a fresh venv, install requirements
and run the full test suite. 2) Run `python -m mm.run_experiments --config config/default.yaml
--quick` and confirm it completes. 3) Verify every figure in docs/report.md is reproducible from
results/ and name the exact command in a docs/REPRODUCE.md. 4) Freeze requirements, fix lint
issues in YOUR files only, and list any issues in other members' files as GitHub issues text
for me to forward. 5) Prepare the release notes and the git tag command for v1.0.
```

---

## 6. Member 2: Objectives, Baselines, Metrics, Conflict Study

### WEEK 1

**M2-W1-A (→ Gate G2; needs only G0)**
```
[Paste PROMPT 0 with X=2]
Task: src/mm/objectives.py implementing f1_imbalance, f2_wait, f3_mismatch (stub returning 0
when role/ping missing) and evaluate(inst, sol) exactly as defined in README section 3.
Vectorize with numpy (the genetic algorithms will call this tens of thousands of times).
Write docs/objectives.md with the math, one worked example, and the "why the objectives
conflict" argument (strictly fair matches need similar-skill players who may not be in the
queue, so players wait longer). Tests: hand-built fixtures with known f1/f2 values (no
simulator needed), the all-unmatched solution, and a perfectly balanced match.
```

**M2-W1-B (→ Gate G3; needs G1 and G2)**
```
[Paste PROMPT 0 with X=2]
Task: src/mm/baselines.py with three registered Matchers (use @register from registry.py):
- "sort_skill": sort by skill, group sequentially into matches of 2k, split into balanced teams
  (snake draft), leave the remainder in the queue.
- "fifo": take the longest-waiting 2k*m players first, then balance teams.
- "greedy_nearest": seed with the longest-waiting unmatched player, greedily add the nearest-skill
  players.
All baselines must respect the hard constraints from validate.py (force-match players near
max_wait, respect the relaxed skill gap) and return feasible solutions; use
validate.is_feasible in tests against simulator.make_instance for light/medium/heavy.
Report in the walkthrough the f1/f2 of each baseline on the medium scenario.
```

### WEEK 2

**M2-W2-A (→ Gate G9; needs G3)**
```
[Paste PROMPT 0 with X=2]
Task: 1) Fill f3_mismatch properly (role coverage per team + ping spread) behind cfg.use_f3.
2) src/mm/metrics.py: mean/median/p95/max wait of matched players, queue length after the round,
match rate, mean/max imbalance, runtime wrapper, feasibility rate. 3) src/mm/benchmarks.py:
run all registered baselines on a list of (scenario, seed) pairs and write a tidy pandas
DataFrame to results/baselines.csv. Tests for each metric on fixtures.
```

### WEEK 3

**M2-W3 (→ Gate G13; needs G12 results CSVs from M4)**
```
[Paste PROMPT 0 with X=2]
Task: 1) src/mm/conflict_study.py: for the three scenarios, sweep max_wait, gap_alpha, k and
queue size, and show (table + data for plots) how baselines behave: sorting cannot trade
fairness against waiting, which is the "why optimization is required" argument. 2) src/mm/stats.py:
Wilcoxon signed-rank + effect size between algorithms across seeds, reading results/*.csv
produced by M4 (do not edit those files). 3) Add the findings (5-8 bullet points with numbers) to
docs/objectives.md under "Empirical conflict evidence".
```

---

## 7. Member 3: Optimization Algorithms

### WEEK 1

**M3-W1 (→ Gate G4; needs G0 and G1. Start from the spec, push after G1.)**
```
[Paste PROMPT 0 with X=3]
Task: src/mm/encoding.py and src/mm/repair.py.
1. Encoding: a genome that represents a grouping/permutation, e.g. a permutation of player
   indices where consecutive blocks of 2k form matches (first k = team 0, next k = team 1)
   and the last queue_len mod 2k (plus optional "skip" markers) stay unmatched. Also define
   how many matches m to form (variable-length or fixed m with a skip mask). Document
   the choice in docs/algorithms.md.
2. decode(inst, genome) -> Solution.
3. repair(inst, solution) -> Solution: force-include players near max_wait, swap out
   pairs violating the relaxed skill gap, drop incomplete matches. After repair,
   validate.is_feasible must hold in >=99% of random genomes (test this).
4. random_feasible(inst, rng) generator, and mutation/crossover operators
   (swap, insert, order crossover) that preserve valid permutations.
Tests: decode round-trip, repair feasibility rate, operators preserve permutation validity.
```

### WEEK 2

**M3-W2-A (→ Gate G5; needs G2 and G4)**
```
[Paste PROMPT 0 with X=3]
Task: NSGA-II in src/mm/nsga2.py, registered as "nsga2" with @register. Use pymoo (preferred,
via a custom Problem/permutation operators from encoding.py) or a clean hand-written NSGA-II.
Objectives come from objectives.evaluate (f1, f2, optional f3). Handle constraints via
repair. Return the non-dominated set as list[Solution] and expose population size,
generations, seed through config. Test: on a small instance, the front must dominate or equal the
three baselines in at least one objective; the run must be deterministic for a given seed.
```

**M3-W2-B (→ Gate G6; needs G2 and G4; parallel with W2-A)**
```
[Paste PROMPT 0 with X=3]
Task: src/mm/weighted.py with two registered matchers, "sa_weighted" (Simulated Annealing)
and "ga_weighted" (single-objective GA). Both minimize w*f1_norm + (1-w)*f2_norm, with
normalisation constants taken from the baselines on the same instance. Provide
sweep_weights(inst, weights=np.linspace(0,1,11)) that runs the algorithm once per weight and
returns the collected solutions (this is the weighted-sum approach's version of a "front").
Tests: w=1 favours minimum imbalance, w=0 favours minimum wait; deterministic per seed.
```

### WEEK 3

**M3-W3 (→ Gate G11; needs G3, G5, G6)**
```
[Paste PROMPT 0 with X=3]
Task: src/mm/hybrid.py, registered as "hybrid_greedy_ls". 1) Seed the initial population/solution
with the output of the greedy_nearest and fifo baselines (import them via registry.get). 2) Add
swap-based local search (swap two players across teams/matches if it improves a scalarised
objective and stays feasible) as a memetic step inside NSGA-II or as a post-pass on the front.
3) Add an experiment flag so M4 can run hybrid vs plain NSGA-II. Write in docs/algorithms.md an
honest "Only a hybrid if experiments show a limitation" note: state the limitation of plain
NSGA-II you observed (e.g. slow convergence at 400 players) with numbers. 4) Tune
population/generations/mutation rates with a small grid; record the final values in
config/ ONLY by writing them in a new file config/algorithms_tuned.yaml (M1's default.yaml
stays untouched).
```

---

## 8. Member 4: Pareto Analysis, Experiments, Relaxation Rule, Report

### WEEK 1

**M4-W1 (→ Gate G7; needs only G0)**
```
[Paste PROMPT 0 with X=4]
Task: src/mm/pareto.py and src/mm/plots.py, both with NO dependency on algorithms.
pareto.py: non_dominated(F) for an (n, m) array of minimization objectives, hypervolume(F, ref)
for 2 objectives (+ a generic fallback for 3), spacing/spread metric, and
representative_points(F) returning three labelled operating points: "very_fair" (min f1),
"fast" (min f2) and "balanced" (knee point, e.g. max distance from the line joining the extremes).
plots.py: pareto_plot(fronts: dict[name, F]) , tradeoff_curve, boxplot of wait vs imbalance
across seeds. Test everything on synthetic arrays.
```

### WEEK 2

**M4-W2 (→ Gate G8; needs G1 and G3; soft-needs G5/G6, auto-detected via registry)**
```
[Paste PROMPT 0 with X=4]
Task: src/mm/run_experiments.py (CLI: --config, --quick, --scenarios, --seeds,
--algorithms). For every (scenario, seed, algorithm) pair: build the instance with
simulator.make_instance, call registry.get(name).solve(inst), evaluate every returned solution
with objectives.evaluate, validate each with validate.is_feasible (record the feasibility
rate), measure runtime, then write tidy CSVs into results/raw/ and the front per run into
results/fronts/. The runner must skip algorithms that are not on the branch yet (registry
lists only importable ones) and say so clearly. Add --quick (2 seeds, light scenario). Test the
runner using the baselines only. Produce a first baseline-only figure set in results/figures/.
```

### WEEK 3

**M4-W3-A (→ Gate G12; needs G8, G9, G11)**
```
[Paste PROMPT 0 with X=4]
Task: 1) Run the full experiment: all scenarios, >= 10 seeds, algorithms: sort_skill, fifo,
greedy_nearest, sa_weighted (weight sweep), ga_weighted, nsga2, hybrid_greedy_ls. 2) Compute
hypervolume and spread per algorithm, and the 3 representative operating points (very fair /
balanced / fast) per scenario. 3) src/mm/relaxation.py: derive the suggested rule "how much to
relax the skill gap as waiting increases", i.e. fit gap_alpha (or a piecewise rule) from the
Pareto fronts: for each wait bucket, which gap yields the balanced point? Output a table and a
one-line rule such as `gap(w) = gap0 + alpha*w`, validated by re-running the rolling_sim with
the suggested rule vs the fixed gap. 4) Final figures into results/figures/.
```

**M4-W3-B (→ Gate G14; needs G12 and G13)**
```
[Paste PROMPT 0 with X=4]
Task: write docs/report.md and docs/slides_outline.md using the actual files in results/ and
M2/M3's docs (read only, do not edit them). Sections: Problem statement, Decision variables,
Objectives and constraints, Data (clearly SYNTHETIC), Why optimization is required, Why
objectives conflict, Baselines, Algorithms (NSGA-II vs weighted SA/GA, hybrid), Results (Pareto
fronts, hypervolume, operating points), Suggested relaxation rule, Limitations, Reproducibility
(commands). Every number in the report must come from a file in results/; list that file next to
the number. Do not invent results; write "not run" where something is missing.
```

---

## 9. Weekly summary

| Week | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| **1** (Oct 27–Nov 2) | G0 scaffold, G1 simulator + validator | G2 objectives, G3 baselines | G4 encoding + repair | G7 Pareto + plots |
| **2** (Nov 3–9) | G10 rolling sim, hardening | G9 f3 + metrics + benchmarks | G5 NSGA-II, G6 SA/GA | G8 experiment runner |
| **3** (Nov 10–16) | G15 final QA + tag | G13 conflict study + stats | G11 hybrid + tuning | G12 final runs + relaxation rule, G14 report |

## 10. Definition of done (each prompt)
- [ ] `/gate-check` prints PASS (owner check, tests, rebase onto `develop`)
- [ ] Only owned files changed
- [ ] Tests written and passing
- [ ] Walkthrough reviewed by the member; PR opened to `develop`
- [ ] Gate announced in the team chat so dependents can pull
