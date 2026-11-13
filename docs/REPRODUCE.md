# Reproduction Guide: Multi-Objective Matchmaking Optimization

This document provides exact, deterministic commands to reproduce every experimental result, benchmark table, and figure presented in [`docs/report.md`](file:///c:/Users/HP/Documents/ot-project/docs/report.md).

> **DISCLAIMER — SYNTHETIC DATA ONLY:**  
> All player ratings, arrival sequences, queue wait times, roles, and network latencies are generated via controlled pseudo-random number generator (PRNG) seeds. No proprietary player telemetry or real-world gaming traces were used.

---

## 1. Environment Setup (Clean Clone)

### Prerequisites
- Python 3.11 or later
- Git

### Installation Steps

```bash
# 1. Clone repository and checkout release branch
git clone https://github.com/krishmulwani23/game-matchmaking.git matchmaking-opt
cd matchmaking-opt
git checkout develop

# 2. Create isolated virtual environment
python -m venv venv

# On Linux/macOS:
source venv/bin/activate
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip and install all locked dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Set PYTHONPATH to include src (or install editable)
# On Linux/macOS:
export PYTHONPATH=src
# On Windows PowerShell:
$env:PYTHONPATH="src"
```

### Verification (Full Test Suite)
Run all 209 unit tests across all 4 team members' modules:
```bash
pytest -q
```
Expected output: `209 passed in ~25-30s`.

---

## 2. Table-by-Table & Figure-by-Figure Reproduction Map

| Report Artifact | Source File in `results/` | Generating Command | Runtime Estimate |
|:---|:---|:---|:---:|
| **Section 5 & 6:** Heuristic Sensitivity Sweeps | `results/conflict/conflict_study_table.csv`<br>`results/conflict/sweep_gap_alpha.csv`<br>`results/conflict/sweep_k.csv`<br>`results/conflict/sweep_max_wait.csv`<br>`results/conflict/sweep_queue_size.csv`<br>`results/conflict/conflict_tradeoffs.png` | `python -m mm.conflict_study` | ~30 s |
| **Section 7:** Baseline Algorithms Benchmark | `results/baselines/baselines.csv` | `python -m mm.benchmarks --scenarios light medium heavy --seeds 42 43 44 45 46` | ~5 s |
| **Section 7 & 8:** Full Algorithms Evaluation & Front Extraction | `results/metrics_summary.csv`<br>`results/raw/experiments_summary.csv`<br>`results/raw/solutions_detail.csv`<br>`results/fronts/*.csv` | `python -m mm.run_experiments --config config/default.yaml` | ~3-4 min |
| **Section 8.3:** Wilcoxon Hypothesis Testing | `results/conflict/stats_comparisons.csv` | `python -m mm.stats` | ~2 s |
| **Section 9:** Pareto Frontiers & Operating Points | `results/operating_points.csv`<br>`results/figures/pareto_comparison_*.png`<br>`results/figures/boxplot_all_*.png`<br>`results/figures/tradeoff_operating_points_*.png`<br>`results/figures/baselines_*.png` | Generated automatically by `python -m mm.run_experiments --config config/default.yaml` | (Included above) |
| **Section 10.2:** Relaxation Policies Comparison | `results/relaxation_table.csv`<br>`results/figures/relaxation_rule_curve.png` | Generated automatically by `python -m mm.run_experiments` | (Included above) |
| **Section 10.3:** Dynamic Rolling Queue Validation | `results/relaxation_validation.csv`<br>`results/figures/relaxation_rolling_validation.png` | Generated automatically by `python -m mm.run_experiments` | (Included above) |

---

## 3. Step-by-Step Reproduction Instructions

### Step 3.1: Quick Smoke Test (< 30 seconds)
To verify the end-to-end simulation, metaheuristic search, front extraction, and figure generation pipelines quickly without running full multi-seed sweeps:

```bash
python -m mm.run_experiments --config config/default.yaml --quick
```

This runs:
- `light` scenario on seeds 42 and 43.
- Evaluates all 7 registered matchers (`sort_skill`, `fifo`, `greedy_nearest`, `nsga2`, `sa_weighted`, `ga_weighted`, `hybrid_greedy_ls`).
- Extracts non-dominated Pareto fronts into `results/fronts/`.
- Computes operating points into `results/operating_points.csv`.
- Generates all baseline and multi-algorithm Pareto tradeoff figures in `results/figures/`.
- Fits the dynamic skill gap relaxation rule and validates it in rolling simulation.

### Step 3.2: Full Production Benchmark Sweep (~3-4 minutes)
To reproduce the complete quantitative numbers published in `docs/report.md`:

```bash
# 1. Run full experimental grid across all scenarios, 10 seeds, and all 7 algorithms
python -m mm.run_experiments --config config/default.yaml

# 2. Run baseline benchmark evaluation
python -m mm.benchmarks --scenarios light medium heavy --seeds 42 43 44 45 46

# 3. Run empirical conflict parameter sweeps (generates conflict_tradeoffs.png)
python -m mm.conflict_study

# 4. Compute Wilcoxon signed-rank significance statistics
python -m mm.stats
```

### Step 3.3: Standalone Rolling Queue Simulation
To test multi-round arrival simulations (`mm.rolling_sim`) with specific matchers, scenarios, or relaxation parameters:

```bash
# Run 30-round simulation with FIFO on the light scenario
python -c "from mm.rolling_sim import run_rolling_sim, summarize_rolling_logs; df = run_rolling_sim('fifo', scenario='light', n_rounds=30); print(summarize_rolling_logs(df))"

# Compare fixed gap (alpha=0.0) vs adaptive relaxation (alpha=0.95)
python -c "from mm.rolling_sim import run_rolling_sim; df_fixed = run_rolling_sim('fifo', gap_alpha=0.0); df_adapt = run_rolling_sim('fifo', gap_alpha=0.95); print('Fixed mean wait:', df_fixed['mean_wait'].mean(), 'Adaptive mean wait:', df_adapt['mean_wait'].mean())"
```

---

## 4. Verification & Ownership Checks

Verify that only authorized files are modified and adhere to the project ownership table:

```bash
python scripts/check_owner.py 1
```
Expected output: `PASS: All touched file(s) belong to Member 1.`
