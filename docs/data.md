# Synthetic Data Specification

> **IMPORTANT DISCLAIMER — SYNTHETIC DATA ONLY:**
> All data used and produced by this benchmark is entirely **SYNTHETIC**.
> No real-world player telemetry, game logs, or proprietary match histories from commercial titles or academic trace dumps were used in this project.
> All player attributes, skill ratings, queue waiting times, network latencies, and arrival timestamps are generated from stochastic statistical models.

---

## 1. Overview & Motivation

In queue-based multiplayer matchmaking optimization, algorithms must be evaluated across diverse traffic loads and player distributions. Synthetic data provides:
- Fully reproducible evaluation using controlled pseudo-random number generator (PRNG) seeds.
- Transparent mathematical definitions without proprietary licensing constraints or privacy concerns.
- Controllable scenario scales to stress-test matchmaking algorithms under varying queue depths.

---

## 2. Stochastic Generative Models

All players are generated in [`src/mm/simulator.py`](file:///c:/Users/HP/Documents/ot-project/src/mm/simulator.py) using `numpy.random.default_rng(seed)`:

### 2.1 Skill Rating (`skill`)
- **Distribution:** $\text{Skill} \sim \mathcal{N}(\mu = 1500, \sigma = 300)$, clipped below at $100.0$.
- **Rationale:** Mirrors Elo and TrueSkill-style competitive gaming rating distributions, where the median casual player is centered near 1500 and higher-tier players extend beyond 2100.

### 2.2 Waiting Time in Queue (`wait`)
- **Distribution:** $\text{Wait} \sim \text{Exponential}(\lambda = 1/30.0)$, scale $\beta = 30.0$ seconds.
- **Rationale:** Models memoryless queue arrival states where most players have recently queued, with an exponential tail of players waiting longer.

### 2.3 Arrival Timestamps (`arrival`)
- **Distribution:** Inter-arrival intervals $\Delta t \sim \text{Exponential}(\beta = 5.0)$ seconds; arrival timestamps are cumulative sums: $t_{\text{arr}, i} = \sum_{j=1}^i \Delta t_j$.
- **Rationale:** Generates a homogeneous Poisson arrival process, standard in queuing theory and matchmaking simulation.

### 2.4 Roles (`role`)
- **Distribution:** Categorical distribution over $\{0, 1, 2\}$ representing:
  - `0`: Tank ($20\%$ probability)
  - `1`: DPS ($50\%$ probability)
  - `2`: Support ($30\%$ probability)
- **Rationale:** Matches common role compositions in hero/class-based team multiplayer titles (e.g. 1 Tank, 2 DPS, 1 Support per team).

### 2.5 Network Latency / Ping (`ping`)
- **Distribution:** $\text{Ping} \sim \text{Lognormal}(\mu = 3.5, \sigma = 0.5)$ ms.
- **Rationale:** Network latency cannot be negative and exhibits a positive skew with typical medians near $e^{3.5} \approx 33\text{ ms}$ and long tails reaching $100\text{--}150+\text{ ms}$.

---

## 3. Queue Scenarios

Instances are configured into three canonical load scenarios:

| Scenario | Default Queue Size ($N$) | Team Size ($k$) | Description |
|:---|:---:|:---:|:---|
| **`light`** | 24 players | 2 (4 per match) | Low-traffic queue, testing behavior under player scarcity. |
| **`medium`** | 100 players | 2 (4 per match) | Standard queue load, typical operating baseline. |
| **`heavy`** | 400 players | 2 (4 per match) | Peak traffic queue, testing scalability and Pareto trade-off diversity. |

---

## 4. Reproducibility

Each call to `make_instance(n_players, k, seed, scenario)` is deterministic for a given `seed`. 
Different seeds yield statistically independent instance realizations under the same scenario parameterizations.
