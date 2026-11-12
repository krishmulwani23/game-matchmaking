# Presentation Slide Deck Outline: Multi-Objective Matchmaking Optimization

**Project:** Multi-Objective Optimization of Queue-Based Multiplayer Matchmaking  
**Team Members:** Member 1 (Krish), Member 2 (Sarvesh), Member 3 (Vedant), Member 4 (RAM)  
**Deliverable:** Checkpoint G14 Presentation Outline  
**Source Document:** [`docs/report.md`](file:///c:/Users/The_Ram/OneDrive/Desktop/OT_proj/game-matchmaking/docs/report.md)

---

### Slide 1: Title & Executive Overview
- **Title:** Multi-Objective Optimization of Queue-Based Multiplayer Matchmaking
- **Subtitle:** Balancing Match Fairness and Queue Latency in Competitive Multiplayer Systems
- **Presenters:** Krish (M1), Sarvesh (M2), Vedant (M3), RAM (M4)
- **Key Takeaways:**
  - Formulated multiplayer matchmaking as a constrained multi-objective combinatorial optimization problem.
  - Benchmarked baseline heuristics vs. metaheuristics (NSGA-II, weighted SA/GA, Hybrid Greedy + LS).
  - Delivered a production-ready Pareto frontier with three operational setpoints (Very Fair, Balanced, Fast).
  - Derived and validated an empirical dynamic skill gap relaxation rule: $\text{gap}(w) = 100.0 + 0.95 \cdot w$.

---

### Slide 2: The Matchmaking Dilemma (Motivation & Problem Context)
- **The Core Conflict:**
  - Fast matchmaking $\implies$ high skill disparity, blowout matches, player churn.
  - Fair matchmaking $\implies$ holding players in queue, excessive latency, starvation risk.
- **System Architecture:**
  - Periodic rounds at intervals of $\Delta t = 10.0$ s.
  - Pool of $N$ queued players dynamically matched into teams of size $k$ ($2k$ per match).
- **Core Challenge:** How to systematically select who gets matched and who waits to optimize both fairness and wait time.

---

### Slide 3: Decision Variables & Representation Rationale
- **Direct Decision Vectors:**
  - Match assignment vector $\mathbf{m} \in \{-1, 0, \dots, M-1\}^N$ ($m_i = -1$: hold in queue; $m_i \ge 0$: match index).
  - Team assignment vector $\mathbf{t} \in \{0, 1\}^N$ (binary team designation).
- **Why Binary Matrices Fail:** Curse of dimensionality and invalid match sizes.
- **Indirect Permutation Chromosome:**
  - Permutation $\pi \in S_N$ + match count $M \in [M_{\min}, M_{\max}]$.
  - First $2k \cdot M$ slots decoded into matches via snake draft; remaining tail players held in queue.
  - Bounded by starvation limits: $M_{\min} = \lceil S_{\text{starving}} / (2k) \rceil$.

---

### Slide 4: Objectives & Hard Constraints
- **Objective 1 — Skill Imbalance ($f_1$, Minimize):**
  - Mean absolute difference in team mean skills across formed matches: $\frac{1}{M}\sum |\mu_{m, 0} - \mu_{m, 1}|$.
- **Objective 2 — Unmatched Queue Wait ($f_2$, Minimize):**
  - Total cumulative wait of all players left behind in the queue: $\sum_{m_i = -1} (w_i + \Delta t)$.
- **Hard Constraints (100% Guaranteed by `repair.py`):**
  1. Exact match capacity ($2k$ players, $k$ per team).
  2. Starvation deadline: $w_i + \Delta t > W_{\max} \implies m_i \ge 0$.
  3. Dynamic skill tolerance: $|s_i - s_j| \le \text{gap}_0 + \alpha \cdot \max(w_i, w_j)$.
  4. Network latency limit: $|\rho_i - \rho_j| \le \rho_{\max}$.

---

### Slide 5: Stochastic Generative Data Pipeline (SYNTHETIC)
- **Disclaimer:** All evaluation data is 100% SYNTHETIC (no proprietary player logs used).
- **Generative Distributions:**
  - Skill: $s_i \sim \mathcal{N}(1500, 300^2)$ clipped at $100.0$ (Elo distribution).
  - Elapsed Wait: $w_i \sim \text{Exponential}(\beta = 30\text{ s})$ (Poisson arrival queue state).
  - Arrival: $\Delta t_{\text{arr}} \sim \text{Exponential}(5\text{ s})$.
  - Roles & Ping: Tank/DPS/Support; Lognormal ping distribution.
- **Standard Benchmark Scenarios:**
  - Light ($N=24$), Medium ($N=100$), Heavy ($N=400$).

---

### Slide 6: Why Optimization is Required
- **NP-Hard Combinatorial Problem:** Partitioning $N$ players into teams and matches exceeds $10^{60}$ configurations for $N=100$.
- **Empirical Failure of Fixed Heuristics:**
  - `sort_skill` is latency-blind: stuck at $f_1 \approx 5.53$ Elo and $f_2 \approx 313.4 - 337.3$ s regardless of max wait deadline (`results/conflict/conflict_study_table.csv`).
  - `fifo` collapses under strict skill tolerances ($\alpha = 0$): match rate drops to **21.2%**, wait explodes to **4,362.6 s** (`results/conflict/conflict_study_table.csv`).
- **Conclusion:** Static heuristics cannot adapt to shifting traffic patterns. Optimization is required.

---

### Slide 7: Why the Objectives Conflict (Empirical Evidence)
- **The Mathematical Trade-off:**
  - Holding players for better balance adds linear queue latency $\Delta t$ per player.
  - Rushing matches to clear queue pairs players with wide skill disparities.
- **Empirical Sweeps (`results/conflict/`):**
  - Increasing $\alpha$ from $0.0 \to 2.0$ improves FIFO match rate from **21.2% to 45.9%**, but skill imbalance spikes by **+173%** ($8.00 \to 21.85$ Elo).
  - Increasing team size $k$ from $1 \to 4$ causes FIFO match rate to plummet from **86.9% to 11.1%** and wait time to spike twenty-fold ($264.2\text{s} \to 5,280.6\text{s}$).

---

### Slide 8: Baseline Algorithms
- **Heuristic Comparison:**
  - `sort_skill`: High fairness, zero latency awareness.
  - `fifo`: Low queue age, poor match balance.
  - `greedy_nearest`: Balances oldest player with closest skill peers.
- **Quantitative Benchmark Performance (`results/metrics_summary.csv`):**
  - Light Scenario: `sort_skill` HV = 126,108.2 | `greedy_nearest` HV = 102,834.9 | `fifo` HV = 65,528.6
  - Medium Scenario: `sort_skill` HV = 1,132,572.7 | `greedy_nearest` HV = 1,076,053.2 | `fifo` HV = 719,961.1

---

### Slide 9: Optimization Algorithms Architecture
- **Multi-Objective Evolutionary Search (`nsga2`):**
  - Elitist non-dominated sorting + crowding distance diversity preservation.
  - Order Crossover (OX1) + Swap/Insert/Inversion mutations + multi-stage repair.
- **Scalarized Metaheuristics (`sa_weighted`, `ga_weighted`):**
  - Minimizes $J = w \cdot f_{1, \text{norm}} + (1-w) \cdot f_{2, \text{norm}}$ across weight sweep $w \in [0, 1]$.
- **Hybrid Greedy + Local Search (`hybrid_greedy_ls`):**
  - Seeds candidates with greedy nearest and FIFO heuristics.
  - Explores team swaps, match swaps, and queue swaps with scalarized acceptance.

---

### Slide 10: Multi-Objective Results & Hypervolume Comparison
- **Empirical Findings across 10 Seeds (`results/metrics_summary.csv`):**
  - **Light ($N=24$):** `sa_weighted` sweep achieved highest hypervolume (**130,873.1**), followed by `nsga2` (**128,702.3**) and `hybrid_greedy_ls` (**122,907.7**).
  - **Medium ($N=100$):** `hybrid_greedy_ls` achieved global maximum hypervolume (**1,134,557.2**) in **2.82 s** (nearly 3x faster than NSGA-II at 7.99 s).
- **Statistical Significance (`results/conflict/stats_comparisons.csv`):**
  - Multi-objective search statistically dominates single-minded FIFO (Wilcoxon $p < 0.05$, rank-biserial $r = 1.0$).

---

### Slide 11: Production Operating Points (Selecting the Operating Point)
- **Three Labelled Setpoints per Scenario (`results/operating_points.csv`):**
  - **Very Fair Point ($f_1 = 0.00$ Elo, $f_2 = 713.5$ s on Light):**
    - High-tier competitive ranked mode. Prioritizes exact skill parity; accepts holding players.
  - **Balanced Knee Point ($f_1 = 1.75$ Elo, $f_2 = 331.1$ s on Light; $f_1 = 3.21$ Elo, $f_2 = 603.7$ s on Medium):**
    - Recommended default operational target. Maximizes trade-off efficiency.
  - **Fast Match Point ($f_1 = 6.89$ Elo, $f_2 = 145.6$ s on Light; $f_1 = 4.89$ Elo, $f_2 = 195.3$ s on Medium):**
    - Casual quickplay mode and peak rush hours. Clears queue with minimal delay.

---

### Slide 12: Empirical Skill Gap Relaxation Rule
- **The Core Concept:**
  - Allow the acceptable skill gap to expand linearly as player wait time increases:
    $$\text{gap}(w) = \text{gap}_0 + \alpha \cdot w$$
- **Empirical Optimal Fit (`results/relaxation_table.csv`):**
  $$\mathbf{gap(w) = 100.0 + 0.95 \cdot w}$$
- **Policy Comparison:**
  - Fixed Gap ($\alpha=0$): High fairness, critical starvation risk.
  - Conservative ($\alpha=0.5$): Acceptable only for light traffic.
  - Recommended Adaptive ($\alpha=0.95$): Optimal compromise across all scenarios.
  - Piecewise Grace ($\text{gap}(w) = 100 + 1.50(w-20)^+$): Zero relaxation for first 20s.

---

### Slide 13: Rolling Queue Simulation Validation
- **Simulation Protocol:**
  - 20 rounds of stochastic rolling queue arrivals via `mm.rolling_sim`.
  - Compared Fixed Gap ($\alpha=0$) vs. Suggested Adaptive Rule ($\alpha=0.95$).
- **Key Validation Results (`results/relaxation_validation.csv`):**
  - **Mean Wait Time:** Reduced by **18.1%** in Light ($43.0\text{s} \to 35.2\text{s}$) and **23.7%** in Medium ($38.1\text{s} \to 29.1\text{s}$).
  - **Peak Queue Backlog:** Reduced from $43 \to 33$ players (Light) and $58 \to 44$ players (Medium).
  - **Feasibility:** Reached 100% feasibility under the adaptive rule where fixed gap failed.

---

### Slide 14: Engineering Limitations & Real-World Extensions
- **Real-World Considerations:**
  - Player reneging (queue abandonment before $W_{\max}$).
  - Multimodal and skewed skill distributions in production populations.
  - Pre-made parties, duos, and squad constraints.
  - Geographically distributed multi-region matchmaking clusters.
- **Future Directions:** Incorporating real-time player abandonment probability into the $f_2$ objective formulation.

---

### Slide 15: Conclusion & Summary
- **Accomplishments:**
  - Formulated and verified the mathematical and computational foundation of queue-based matchmaking.
  - Proved the necessity of multi-objective optimization over single-minded heuristics.
  - Delivered superior runtime and hypervolume with Hybrid Greedy + Local Search (`hybrid_greedy_ls`).
  - Derived and validated an actionable, empirical skill gap relaxation rule ($\text{gap}(w) = 100.0 + 0.95 \cdot w$).
- **Ready for Release:** Fully tested, documented, and reproducible (`v1.0` ready for Checkpoint G15).

