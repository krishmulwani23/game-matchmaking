# Final Technical Report: Multi-Objective Optimization of Queue-Based Multiplayer Matchmaking

**Authors:** Team Matchmaking Optimization  
- Member 1 (Krish): Foundation, Simulator, Rolling Simulation, Integration  
- Member 2 (Sarvesh): Mathematical Objectives, Baselines, Benchmarks, Conflict Study  
- Member 3 (Vedant): Encoding, Constraint Repair, NSGA-II, Weighted Metaheuristics, Hybrid Search  
- Member 4 (RAM): Pareto Frontiers, Experimentation, Skill Gap Relaxation Rule, Final Technical Report  

**Date:** November 2026  
**Repository:** `matchmaking-opt` | **Integration Branch:** `develop`  
**Artifact Classification:** Technical Project Report (Checkpoint G14)

---

## Executive Summary

This report presents a comprehensive multi-objective optimization framework for queue-based multiplayer matchmaking. In competitive online multiplayer games, matchmaking engines must continuously decide which queued players to group into matches and which to leave waiting in the queue. This decision creates a fundamental conflict between **match fairness** (minimizing team skill imbalance $f_1$) and **queue latency** (minimizing player waiting time $f_2$).

Through systematic empirical experimentation across three canonical traffic scenarios (`light`, `medium`, `heavy`), 10 random seeds, seven algorithms, and 513 sweep configurations, we demonstrate that traditional heuristics (FIFO, greedy nearest, and skill-sorting) cannot dynamically negotiate the fairness-latency trade-off. We deploy and benchmark multi-objective evolutionary algorithms (**NSGA-II**), scalarized metaheuristics (**SA/GA with weight sweeps**), and a novel **Hybrid Greedy + Local Search** algorithm (`hybrid_greedy_ls`). 

Finally, we derive an empirical dynamic skill gap relaxation rule:
$$\mathbf{gap(w) = 100.0 + 0.95 \cdot w}$$
which resolves queue starvation while limiting match imbalance degradation.

---

## 1. Problem Statement

Modern multiplayer games (e.g., 2v2 to 5v5 team competitive titles) evaluate pools of queued players at discrete periodic rounds of duration $\Delta t$. At round $t$, a pool of $N$ queued players $\mathcal{P} = \{p_0, p_1, \dots, p_{N-1}\}$ awaits assignment. 

The matchmaker faces a dual challenge:
1. Pairing players into balanced, highly competitive matches where neither team has an unfair advantage.
2. Servicing players promptly to prevent queue bloat, excessive wait times, and player abandonment.

Forming matches prematurely groups players with wide skill disparities, leading to one-sided blowout matches and poor player satisfaction. Conversely, waiting indefinitely for perfectly matched peers creates catastrophic queue accumulation and player starvation. This project formulates, solves, and validates this challenge as a constrained multi-objective combinatorial optimization problem delivering:
- An explicit **Pareto-optimal frontier** trading off skill imbalance vs. wait time.
- Three representative operational operating points: **Very Fair**, **Balanced (Knee)**, and **Fast**.
- A validated, empirical **skill gap relaxation rule** $\text{gap}(w) = \text{gap}_0 + \alpha \cdot w$.

---

## 2. Decision Variables & Chromosome Representation

### 2.1 Decision Variables
For a pool of $N$ players and team size $k$ (match size $2k$), a candidate solution $\mathbf{x} = (\mathbf{m}, \mathbf{t})$ is parameterized by two assignment vectors:
1. **Match Assignment Vector $\mathbf{m} \in \{-1, 0, 1, \dots, M-1\}^N$:**
   - $m_i = -1$: player $p_i$ remains unmatched in the queue for the current round.
   - $m_i = m \ge 0$: player $p_i$ is assigned to formed match $m$.
2. **Team Assignment Vector $\mathbf{t} \in \{0, 1\}^N$:**
   - $t_i \in \{0, 1\}$: assigns player $p_i$ to Team 0 or Team 1 within match $m_i$ (relevant only when $m_i \ge 0$).

### 2.2 Chromosome Representation
To circumvent the severe dimensionality and infeasibility bottlenecks of raw binary matrices, we adopt an indirect permutation-with-match-count representation:
$$\text{Genome} = (\pi, M)$$
where $\pi \in S_N$ is a permutation of player indices $\{0, \dots, N-1\}$ and $M \in [M_{\min}, M_{\max}]$ is the integer number of formed matches:
- Match $m$ takes slice $\pi[2k \cdot m : 2k \cdot (m+1)]$, split into Team 0 ($\pi[2km : 2km+k]$) and Team 1 ($\pi[2km+k : 2k(m+1)]$).
- Players in the tail $\pi[2k \cdot M : N]$ remain in the queue ($m_i = -1$).
- Bounded match limits: $M_{\max} = \lfloor N / (2k) \rfloor$, and $M_{\min} = \min(\lceil S_{\text{starving}} / (2k) \rceil, M_{\max})$.

---

## 3. Mathematical Objectives & Hard Constraints

All objectives are formulated to be **minimized**:

### 3.1 Objective 1: Team Skill Imbalance ($f_1$)
For each formed match $m \in \{0, \dots, M-1\}$, team mean skills are computed as:
$$\mu_{m, 0} = \frac{1}{k} \sum_{i \in \mathcal{T}_{m, 0}} s_i, \qquad \mu_{m, 1} = \frac{1}{k} \sum_{i \in \mathcal{T}_{m, 1}} s_i$$
The skill imbalance of match $m$ is the absolute difference $\Delta_m = |\mu_{m, 0} - \mu_{m, 1}|$. The overall objective is the mean across all $M$ matches:
$$f_1(\mathbf{x}) = \begin{cases} \frac{1}{M} \sum_{m=0}^{M-1} \Delta_m & \text{if } M > 0 \\ 0.0 & \text{if } M = 0 \end{cases}$$

### 3.2 Objective 2: Total Unmatched Queue Wait ($f_2$)
Unmatched players remain in the queue and incur an additional round latency $\Delta t$. The waiting time objective penalizes the total cumulative wait of all held players:
$$f_2(\mathbf{x}) = \sum_{i : m_i = -1} (w_i + \Delta t)$$
If all players are matched ($M = N / (2k)$), $f_2(\mathbf{x}) = 0.0$.

### 3.3 Objective 3: Role & Network Mismatch ($f_3$, Extensible)
Measures role coverage deficit and ping spread across matches:
$$f_3(\mathbf{x}) = \frac{1}{M} \sum_{m=0}^{M-1} \left( \text{RoleDeficit}_m + \text{PingSpread}_m \right)$$
Evaluated when role and ping attributes are active; defaults to $0.0$ when unmodelled.

### 3.4 Hard Constraints
A feasible solution $\mathbf{x}$ must strictly satisfy four hard constraints:
1. **Match Capacity & Symmetry:** Exactly $2k$ players per match ($k$ on Team 0, $k$ on Team 1).
2. **Starvation Prevention Deadline:** If $w_i + \Delta t > W_{\max}$, player $i$ **must** be matched ($m_i \ge 0$).
3. **Dynamic Relaxed Skill Gap:** For every pair in match $m$:
   $$|s_i - s_j| \le \text{gap}_0 + \alpha \cdot \max(w_i, w_j)$$
4. **Network Latency Limit:** $|\rho_i - \rho_j| \le \rho_{\max}$ for all pairs in match $m$.

---

## 4. Dataset Specification (SYNTHETIC)

> **IMPORTANT DISCLAIMER — SYNTHETIC DATA ONLY:**  
> All data used in this benchmark is **100% SYNTHETIC**, generated via stochastic statistical distributions implemented in `src/mm/simulator.py`. No real player traces, proprietary telemetry, or commercial gaming logs were used.

### Generative Distributions
- **Skill Rating ($s_i$):** $s_i \sim \mathcal{N}(\mu = 1500, \sigma = 300)$, clipped below at $100.0$ (Elo-like distribution).
- **Elapsed Wait Time ($w_i$):** $w_i \sim \text{Exponential}(\beta = 30.0\text{ s})$, modeling a steady-state queue arrival distribution.
- **Arrival Timestamp ($a_i$):** Generated via a Poisson arrival process with inter-arrival $\Delta t_{\text{arr}} \sim \text{Exponential}(5.0\text{ s})$.
- **Roles ($r_i$):** Categorical: Tank ($20\%$), DPS ($50\%$), Support ($30\%$).
- **Ping ($\rho_i$):** Lognormal distribution: $\rho_i \sim \text{Lognormal}(\mu = 3.5, \sigma = 0.5)\text{ ms}$ (median $\approx 33\text{ ms}$, long tail to $150+\text{ ms}$).

### Canonical Load Scenarios
- **`light`:** $N = 24$ queued players, $k = 2$ ($2k = 4$ per match). Tests behavior under player scarcity.
- **`medium`:** $N = 100$ queued players, $k = 2$ ($2k = 4$ per match). Standard competitive workload.
- **`heavy`:** $N = 400$ queued players, $k = 2$ ($2k = 4$ per match). Peak traffic stress testing.

---

## 5. Why Optimization is Required

Partitioning $N$ heterogeneous players into $M$ disjoint matches of size $2k$, subject to pairwise non-linear skill constraints and deadline guarantees, is strongly NP-hard (generalizing bin-packing and multi-way graph partitioning).

Standard greedy and sorting heuristics fail dramatically because they make myopic, irreversible decisions:
- **Heuristic Inflexibility:** As demonstrated in our conflict study (`results/conflict/conflict_study_table.csv`), `sort_skill` remains locked at a rigid $f_1 \approx 5.53\text{ Elo}$ and $f_2 \approx 313.4 - 337.3\text{ s}$ across sweeps of $W_{\max} \in [60, 240]\text{s}$. It cannot trade fairness to release starving players.
- **FIFO Catastrophic Collapse:** When skill gaps are held strictly tight ($\alpha = 0.0$), pure FIFO match rate collapses to **21.2%**, allowing total wait time $f_2$ to explode to **4,362.6 s** (`results/conflict/conflict_study_table.csv`).
- **Combinatorial Explosion:** For $N=100$ and $k=2$, the number of possible groupings exceeds $10^{60}$. Only metaheuristic and evolutionary search can discover Pareto-optimal trade-offs across this immense discrete landscape.

---

## 6. Why the Objectives Conflict

The tension between $f_1$ (fairness) and $f_2$ (latency) is rooted in queue supply-and-demand:
1. **Skill Rarity:** Players arrive across a continuous bell curve. Extreme skill tiers ($<1000$ or $>2000$ Elo) are rare in any discrete queue window.
2. **Holding Cost:** To achieve $f_1 \to 0$, the system must hold players until near-identical skill peers arrive. Every second of holding adds $\Delta f_2 = \Delta t \times N_{\text{held}}$ to Objective 2.
3. **Flushing Cost:** To achieve $f_2 \to 0$, the system must aggressively pair all available players immediately, grouping distant skill ratings and causing $f_1$ to surge.

### Quantitative Conflict Evidence
From our systematic parameter sweeps (`results/conflict/sweep_gap_alpha.csv` and `results/conflict/conflict_study_table.csv`):
- Relaxing $\alpha$ from $0.0$ to $2.0$ improves FIFO match rate from **21.2% to 45.9%** and cuts wait from $4,362.6\text{s} \to 2,290.6\text{s}$, but causes skill imbalance $f_1$ to spike by **+173%** (from $8.00\text{ Elo} \to 21.85\text{ Elo}$).
- Increasing team size from $k=1$ to $k=4$ causes FIFO match rate to plummet from **86.9% to 11.1%** and wait time to spike twenty-fold from **264.2 s to 5,280.6 s** (`results/conflict/sweep_k.csv`).

---

## 7. Baseline Algorithms Benchmark

Three single-solution heuristics serve as production baselines:
1. **`sort_skill`:** Sorts queue by skill $s_i$, slices into blocks of $2k$, snake drafts into teams.
2. **`fifo`:** Sorts queue by elapsed wait $w_i$ descending, prioritizing oldest players.
3. **`greedy_nearest`:** Iteratively selects the oldest waiting player and pairs them with their $2k-1$ nearest skill neighbors.

### Quantitative Performance Comparison
Across the evaluated benchmark instances (`results/metrics_summary.csv`):

| Scenario | Algorithm | Mean Hypervolume (HV) | Mean Spread ($\Delta$) | Mean Spacing ($S$) | Mean Runtime | Feasibility | Source File |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Light** | `sort_skill` | 126,108.2 | 0.0000 | 0.00 | 0.0002 s | 100% | `results/metrics_summary.csv` |
| **Light** | `fifo` | 65,528.6 | 0.0000 | 0.00 | 0.0008 s | 100% | `results/metrics_summary.csv` |
| **Light** | `greedy_nearest` | 102,834.9 | 0.0000 | 0.00 | 0.0005 s | 100% | `results/metrics_summary.csv` |
| **Medium** | `sort_skill` | 1,132,572.7 | 0.0000 | 0.00 | 0.0023 s | 100% | `results/metrics_summary.csv` |
| **Medium** | `fifo` | 719,961.1 | 0.0000 | 0.00 | 0.0068 s | 100% | `results/metrics_summary.csv` |
| **Medium** | `greedy_nearest` | 1,076,053.2 | 0.0000 | 0.00 | 0.0037 s | 100% | `results/metrics_summary.csv` |

*Finding:* While `sort_skill` achieves high hypervolume when liquidity is abundant, it produces only a single static solution and cannot adapt to operational latency requirements. `fifo` achieves the worst hypervolume across all scenarios due to ignoring skill discrepancies.

---

## 8. Optimization Algorithms: Design & Comparison

Four multi-objective and metaheuristic algorithms were implemented:
1. **`nsga2`:** Multi-Objective Non-dominated Sorting Genetic Algorithm II with binary tournament selection, crowding distance niching, Order Crossover (OX1), permutation mutation, and multi-stage repair.
2. **`sa_weighted`:** Simulated Annealing with geometric cooling ($\alpha=0.92$) minimizing the scalarized objective $J = w \cdot f_{1, \text{norm}} + (1-w) \cdot f_{2, \text{norm}}$ across systematic weight sweeps $w \in [0, 1]$.
3. **`ga_weighted`:** Single-objective Genetic Algorithm with elitism and tournament selection.
4. **`hybrid_greedy_ls`:** Novel hybrid method that seeds candidate solutions using `greedy_nearest` and `fifo`, then applies swap-based local search (within-match team swaps, inter-match swaps, queue swaps).

### Metaheuristic & Multi-Objective Performance (`results/metrics_summary.csv`)

| Scenario | Algorithm | Mean Hypervolume (HV) | Mean Spread ($\Delta$) | Mean Spacing ($S$) | Mean Runtime | Feasibility | Source File |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Light** | `sa_weighted` (sweep) | **130,873.1** | 0.9557 | 54.22 | 5.9770 s | 100% | `results/metrics_summary.csv` |
| **Light** | `nsga2` | 128,702.3 | 0.5653 | 56.11 | 1.0348 s | 100% | `results/metrics_summary.csv` |
| **Light** | `hybrid_greedy_ls` | 122,907.7 | 0.2025 | 41.45 | 0.3003 s | 100% | `results/metrics_summary.csv` |
| **Light** | `ga_weighted` | 119,350.3 | 0.0000 | 0.00 | 0.6442 s | 100% | `results/metrics_summary.csv` |
| **Medium** | `hybrid_greedy_ls` | **1,134,557.2** | 0.7424 | 172.51 | **2.8205 s** | 100% | `results/metrics_summary.csv` |
| **Medium** | `nsga2` | 1,042,956.7 | 0.7718 | 132.39 | 7.9896 s | 100% | `results/metrics_summary.csv` |
| **Medium** | `sa_weighted` | 926,728.0 | 0.3896 | 517.43 | 7.3570 s | 100% | `results/metrics_summary.csv` |
| **Medium** | `ga_weighted` | 508,058.5 | 0.0000 | 0.00 | 2.9090 s | 100% | `results/metrics_summary.csv` |

### Key Algorithmic Insights:
- **Light Scenarios ($N=24$):** `sa_weighted` with systematic weight sweeps achieved the highest hypervolume (130,873.1), discovering diverse extreme solutions.
- **Medium Scenarios ($N=100$):** `hybrid_greedy_ls` was the superior performer, reaching the global maximum hypervolume (1,134,557.2) nearly **3x faster** than NSGA-II (2.82 s vs. 7.99 s).
- **Statistical Significance:** Wilcoxon signed-rank tests (`results/conflict/stats_comparisons.csv`) confirm that `hybrid_greedy_ls` significantly outperforms FIFO (mean difference $+235,987.6$ HV, rank-biserial $r = 1.0$) and beats pure NSGA-II on medium workloads.

---

## 9. Experimental Results & Operating Points

### 9.1 Multi-Objective Operating Points
By evaluating the combined non-dominated Pareto front, we identified three distinct operational setpoints per scenario (`results/operating_points.csv`):

| Scenario | Operating Point | Imbalance ($f_1$, Elo) | Total Wait ($f_2$, s) | Operational Profile | Source File |
|:---|:---|:---:|:---:|:---|:---|
| **Light** | **Very Fair** | **0.00 Elo** | 713.50 s | Ranked competitive; perfect balance at cost of holding queue. | `results/operating_points.csv` |
| **Light** | **Balanced (Knee)** | **1.75 Elo** | 331.10 s | Optimal general matchmaking; minimal compromise in both axes. | `results/operating_points.csv` |
| **Light** | **Fast** | **6.89 Elo** | **145.60 s** | Casual quickplay; rapid queue clearance with slight imbalance. | `results/operating_points.csv` |
| **Medium** | **Very Fair** | **0.20 Elo** | 3,721.30 s | High-tier ranked; near-zero skill discrepancy. | `results/operating_points.csv` |
| **Medium** | **Balanced (Knee)** | **3.21 Elo** | 603.70 s | Recommended standard production setpoint. | `results/operating_points.csv` |
| **Medium** | **Fast** | **4.89 Elo** | **195.30 s** | Peak rush hours; near-zero remaining queue latency. | `results/operating_points.csv` |

---

## 10. Suggested Skill Gap Relaxation Rule

### 10.1 Mathematical Rule Formulation
To reconcile strict skill requirements with finite queue patience, we formulated the dynamic skill gap constraint:
$$\text{gap}(w) = \text{gap}_0 + \alpha \cdot w$$
Fitting $\alpha$ from our empirical Pareto frontiers yielded the recommended rule:
$$\mathbf{gap(w) = 100.0 + 0.95 \cdot w}$$
*(with piecewise alternative: $\text{gap}(w) = 100.0 + 1.50 \cdot \max(0, w - 20.0)$ providing a 20s grace period).*

### 10.2 Policy Comparison (`results/relaxation_table.csv`)

| Policy | $\alpha$ | Formula | Gap at $w=0\text{s}$ | Gap at $w=60\text{s}$ | Gap at $w=120\text{s}$ | Fairness Rating | Queue Starvation Risk | Source File |
|:---|:---:|:---|:---:|:---:|:---:|:---|:---|:---|
| **Fixed Strict Gap** | 0.00 | $\text{gap}(w) = 100.0$ | 100.0 | 100.0 | 100.0 | Superior (Zero Degradation) | **Critical** (Starvation & Queue Bloat) | `results/relaxation_table.csv` |
| **Conservative** | 0.50 | $\text{gap}(w) = 100.0 + 0.50w$ | 100.0 | 130.0 | 160.0 | High (Strict Elo matching) | Moderate (Delays in sparse queue) | `results/relaxation_table.csv` |
| **Recommended Adaptive** | **0.95** | $\mathbf{gap(w) = 100.0 + 0.95w}$ | **100.0** | **157.0** | **214.0** | **Balanced (Controlled Trade-off)** | **Low (Guaranteed Starvation Prevention)** | `results/relaxation_table.csv` |
| **Piecewise Grace** | 1.50 | $\text{gap}(w) = 100.0 + 1.50(w-20)^+$ | 100.0 | 160.0 | 250.0 | Very High early, decays later | Low (Fast recovery post grace period) | `results/relaxation_table.csv` |
| **Aggressive** | 2.50 | $\text{gap}(w) = 100.0 + 2.50w$ | 100.0 | 250.0 | 400.0 | Poor (High skill disparity) | Negligible (Immediate matching) | `results/relaxation_table.csv` |

### 10.3 Dynamic Validation via Rolling Queue Simulation (`results/relaxation_validation.csv`)
Simulating 20 rounds of multi-round rolling queue arrivals (`mm.rolling_sim`) proved the superiority of the adaptive rule over a fixed gap:
- **Light Scenario (Seed 42):** 
  - Fixed Gap ($\alpha=0$): Mean wait = **42.97 s**, Peak queue backlog = **43 players**, Feasible = False (`results/relaxation_validation.csv`).
  - Suggested Rule ($\alpha=0.95$): Mean wait = **35.19 s** ($-18.1\%$), Peak queue backlog = **33 players** ($-23.3\%$) (`results/relaxation_validation.csv`).
- **Medium Scenario (Seed 42):**
  - Fixed Gap ($\alpha=0$): Mean wait = **38.12 s**, Peak queue backlog = **58 players**, Feasible = False (`results/relaxation_validation.csv`).
  - Suggested Rule ($\alpha=0.95$): Mean wait = **29.09 s** ($-23.7\%$), Peak queue backlog = **44 players** ($-24.1\%$), **100% Feasible** (`results/relaxation_validation.csv`).

---

## 11. Limitations & Future Work

While the proposed framework demonstrates robust performance, several real-world engineering constraints warrant future extension:
1. **Unimodal Skill Assumption:** Real competitive multiplayer titles often exhibit multimodal skill distributions (e.g., smurf accounts, party/stack queues, and casual vs. competitive player tiers).
2. **Player Reneging / Abandonment:** The current simulation assumes infinite player patience up to $W_{\max}$. In production, queue abandonment increases non-linearly with wait time.
3. **Pre-made Parties & Social Matchmaking:** The model assumes independent solo-queue arrivals. Future extensions will model premade duos and squads of size $p \le k$.
4. **Distributed Matchmaking Buckets:** Production titles shard queues geographically across latency zones rather than executing in a centralized dispatcher.

---

## 12. Reproducibility & Verification

The codebase is structured to ensure complete, deterministic reproducibility across platforms:
- **Environment:** Python 3.11+ using `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `pymoo`, `pyyaml`, and `pytest`.
- **Deterministic PRNG Seeds:** Explicit seeds ($42, 43, \dots, 51$) guarantee identical instance generation across runs.
- **Verification Commands:**
  ```bash
  # Run full test suite (200 unit tests)
  pytest -q

  # Verify strict file ownership
  python scripts/check_owner.py 4

  # Execute quick smoke test
  python -m mm.run_experiments --quick

  # Re-run empirical conflict parameter sweeps
  python -m mm.conflict_study
  ```

---

*Report concluded. All figures and tables cited are verified on `develop` in `results/`.*

