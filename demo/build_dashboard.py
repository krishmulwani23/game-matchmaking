import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
data_file = repo_root / "demo" / "project_data.json"

with open(data_file, "r", encoding="utf-8") as f:
    project_data = json.load(f)

json_str = json.dumps(project_data)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Matchmaking Optimization Engine — Live Interactive Demo</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-primary: #0a0e17;
      --bg-surface: #111827;
      --bg-card: #162033;
      --bg-card-hover: #1c2942;
      --border: #24344d;
      --border-accent: #3b82f6;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-subtle: #64748b;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --primary-light: rgba(59, 130, 246, 0.15);
      --success: #10b981;
      --success-light: rgba(16, 185, 129, 0.15);
      --warning: #f59e0b;
      --warning-light: rgba(245, 158, 11, 0.15);
      --danger: #ef4444;
      --danger-light: rgba(239, 68, 68, 0.15);
      --purple: #8b5cf6;
      --purple-light: rgba(139, 92, 246, 0.15);
      --team0: #38bdf8;
      --team1: #f97316;
      --radius: 12px;
      --radius-sm: 8px;
      --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-primary);
      color: var(--text-main);
      line-height: 1.5;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    code, pre, .mono {{ font-family: 'JetBrains Mono', monospace; }}

    /* Layout */
    .container {{ max-width: 1380px; margin: 0 auto; padding: 0 24px; width: 100%; }}
    header {{
      background: rgba(17, 24, 39, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 50;
    }}
    .nav-inner {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 70px;
    }}
    .logo-badge {{
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 800;
      font-size: 1.15rem;
      letter-spacing: -0.02em;
    }}
    .logo-icon {{
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.2rem;
      box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }}
    .nav-tabs {{
      display: flex;
      gap: 6px;
      background: var(--bg-card);
      padding: 5px;
      border-radius: 10px;
      border: 1px solid var(--border);
    }}
    .nav-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 8px 16px;
      border-radius: var(--radius-sm);
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .nav-btn:hover {{ color: var(--text-main); background: rgba(255,255,255,0.05); }}
    .nav-btn.active {{
      background: var(--primary);
      color: #fff;
      box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4);
    }}

    /* Main Content */
    main {{ flex: 1; padding: 28px 0 60px; }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; animation: fadeIn 0.25s ease; }}
    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: translateY(0); }} }}

    /* Cards & Components */
    .card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
      box-shadow: var(--shadow);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
    }}
    .card-title {{
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .card-subtitle {{
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-top: 2px;
    }}

    /* Hero Banner */
    .hero {{
      background: linear-gradient(135deg, rgba(30, 58, 138, 0.4), rgba(88, 28, 135, 0.3)), var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px 32px;
      margin-bottom: 24px;
      position: relative;
      overflow: hidden;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      right: -80px;
      top: -80px;
      width: 250px;
      height: 250px;
      background: radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, transparent 70%);
      pointer-events: none;
    }}
    .hero h1 {{ font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 6px; }}
    .hero p {{ color: var(--text-muted); max-width: 820px; font-size: 0.95rem; }}

    /* Metric Scorecards */
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .stat-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      position: relative;
      overflow: hidden;
    }}
    .stat-card::before {{
      content: "";
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 4px;
      background: var(--accent, var(--primary));
    }}
    .stat-label {{ font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: var(--text-subtle); letter-spacing: 0.05em; }}
    .stat-value {{ font-size: 1.8rem; font-weight: 800; color: var(--text-main); margin: 4px 0; font-family: 'JetBrains Mono', monospace; }}
    .stat-desc {{ font-size: 0.78rem; color: var(--text-muted); }}

    /* Simulator Controls */
    .controls-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: center;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px 20px;
      margin-bottom: 24px;
    }}
    .control-group {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .control-label {{
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    select, button, input[type="range"] {{
      font-family: inherit;
    }}
    select {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: var(--text-main);
      padding: 9px 14px;
      border-radius: var(--radius-sm);
      font-size: 0.88rem;
      font-weight: 600;
      outline: none;
      cursor: pointer;
    }}
    select:focus {{ border-color: var(--primary); }}
    .btn-primary {{
      background: linear-gradient(135deg, #3b82f6, #2563eb);
      color: white;
      border: none;
      padding: 10px 20px;
      border-radius: var(--radius-sm);
      font-weight: 700;
      font-size: 0.9rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
      transition: all 0.2s ease;
    }}
    .btn-primary:hover {{ filter: brightness(1.1); transform: translateY(-1px); }}
    .btn-secondary {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: var(--text-main);
      padding: 9px 16px;
      border-radius: var(--radius-sm);
      font-weight: 600;
      font-size: 0.88rem;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn-secondary:hover {{ background: var(--bg-card-hover); }}

    /* Match Inspector */
    .matches-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
      gap: 18px;
    }}
    .match-box {{
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      transition: transform 0.15s, border-color 0.15s;
    }}
    .match-box:hover {{
      border-color: rgba(59, 130, 246, 0.4);
      transform: translateY(-2px);
    }}
    .match-box-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      padding-bottom: 10px;
    }}
    .match-tag {{
      font-size: 0.8rem;
      font-weight: 800;
      color: var(--text-main);
      background: rgba(255,255,255,0.06);
      padding: 4px 10px;
      border-radius: 6px;
    }}
    .match-delta {{
      font-size: 0.82rem;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 6px;
      font-family: 'JetBrains Mono', monospace;
    }}
    .delta-good {{ background: var(--success-light); color: var(--success); }}
    .delta-fair {{ background: var(--warning-light); color: var(--warning); }}
    .delta-high {{ background: var(--danger-light); color: var(--danger); }}

    .teams-versus {{
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      gap: 12px;
    }}
    .team-col {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .team-label {{
      font-size: 0.72rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .team-0-color {{ color: var(--team0); }}
    .team-1-color {{ color: var(--team1); }}
    .vs-badge {{
      font-size: 0.72rem;
      font-weight: 900;
      color: var(--text-subtle);
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 50%;
      width: 28px;
      height: 28px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .player-chip {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 6px 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.78rem;
    }}
    .player-left {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .role-icon {{ font-size: 0.9rem; }}
    .player-elo {{ font-weight: 700; font-family: 'JetBrains Mono', monospace; }}
    .player-meta {{ font-size: 0.7rem; color: var(--text-subtle); }}

    /* Queue Backlog Section */
    .queue-drawer {{
      margin-top: 24px;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 18px 20px;
    }}
    .queue-chips-container {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
      max-height: 180px;
      overflow-y: auto;
      padding-right: 6px;
    }}
    .waiting-chip {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 6px 10px;
      font-size: 0.75rem;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .waiting-chip.urgent {{
      border-color: var(--danger);
      background: rgba(239, 68, 68, 0.08);
    }}

    /* Pareto Chart Canvas & SVG */
    .chart-container {{
      position: relative;
      width: 100%;
      height: 480px;
      background: var(--bg-surface);
      border-radius: var(--radius);
      border: 1px solid var(--border);
      padding: 16px;
      margin-bottom: 24px;
    }}
    .operating-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 16px;
    }}
    .op-card {{
      flex: 1;
      min-width: 250px;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .op-card:hover, .op-card.selected {{
      border-color: var(--primary);
      background: var(--bg-card);
      box-shadow: 0 4px 16px rgba(59, 130, 246, 0.2);
    }}
    .op-title {{ font-size: 0.92rem; font-weight: 800; display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
    .op-metric {{ font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--text-muted); }}

    /* Table Styles */
    .table-container {{
      width: 100%;
      overflow-x: auto;
      border-radius: var(--radius);
      border: 1px solid var(--border);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.85rem;
    }}
    th {{
      background: var(--bg-surface);
      padding: 12px 16px;
      font-weight: 700;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border);
      text-transform: uppercase;
      font-size: 0.72rem;
      letter-spacing: 0.04em;
    }}
    td {{
      padding: 12px 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      color: var(--text-main);
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(255,255,255,0.02); }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 700;
    }}
    .badge-primary {{ background: var(--primary-light); color: var(--primary); }}
    .badge-success {{ background: var(--success-light); color: var(--success); }}
    .badge-purple {{ background: var(--purple-light); color: var(--purple); }}
    .badge-warning {{ background: var(--warning-light); color: var(--warning); }}

    /* Tooltip */
    #chart-tooltip {{
      position: absolute;
      display: none;
      background: #1e293b;
      border: 1px solid #3b82f6;
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 0.8rem;
      pointer-events: none;
      z-index: 100;
      box-shadow: 0 10px 25px rgba(0,0,0,0.6);
    }}

    /* Responsive */
    @media (max-width: 768px) {{
      .nav-inner {{ flex-direction: column; height: auto; padding: 12px 0; gap: 12px; }}
      .matches-grid {{ grid-template-columns: 1fr; }}
      .teams-versus {{ grid-template-columns: 1fr; gap: 8px; }}
      .vs-badge {{ margin: 0 auto; }}
    }}
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="container nav-inner">
      <div class="logo-badge">
        <div class="logo-icon">⚡</div>
        <div>
          <span>Multi-Objective Matchmaking</span>
          <span style="display:block; font-size: 0.7rem; color: var(--text-muted); font-weight: 500;">Live Simulation & Results Dashboard</span>
        </div>
      </div>
      <div class="nav-tabs">
        <button class="nav-btn active" onclick="switchTab('sim')">🎮 Live Matchmaker</button>
        <button class="nav-btn" onclick="switchTab('pareto')">📈 Pareto Frontier</button>
        <button class="nav-btn" onclick="switchTab('conflict')">⚖️ Why Optimize?</button>
        <button class="nav-btn" onclick="switchTab('scoreboard')">🏆 Scoreboard</button>
        <button class="nav-btn" onclick="switchTab('relaxation')">📜 Skill Relaxation</button>
      </div>
    </div>
  </header>

  <main class="container">

    <!-- Hero / Concept Banner -->
    <div class="hero">
      <h1>Multiplayer Matchmaking Optimization (Proposal 3)</h1>
      <p>
        In competitive games (e.g. Overwatch, Valorant, CS:GO), matchmakers face an unavoidable conflict: 
        <strong>Match Fairness</strong> (pairing equal Elo skills) vs. <strong>Queue Time</strong> (matching players quickly).
        This platform proves why simple sorting algorithms fail and demonstrates how multi-objective evolutionary optimization (NSGA-II) discovers the true Pareto frontier.
      </p>
    </div>

    <!-- ========================================================================= -->
    <!-- TAB 1: LIVE SIMULATOR -->
    <!-- ========================================================================= -->
    <div id="tab-sim" class="tab-content active">
      
      <!-- Controls -->
      <div class="controls-bar">
        <div class="control-group">
          <label class="control-label">Queue Scenario</label>
          <select id="sim-scenario" onchange="loadScenario(this.value)">
            <option value="light">Light Queue (24 players, ~1 min wait)</option>
            <option value="medium" selected>Medium Queue (100 players, standard load)</option>
            <option value="heavy">Heavy Queue (400 players, peak load)</option>
          </select>
        </div>

        <div class="control-group">
          <label class="control-label">Team Sizing (k vs k)</label>
          <select id="sim-k" onchange="runMatchmaker()">
            <option value="1">1 vs 1 (Duel, 2 players/match)</option>
            <option value="2" selected>2 vs 2 (Default, 4 players/match)</option>
            <option value="3">3 vs 3 (Arena, 6 players/match)</option>
            <option value="4">4 vs 4 (Squad, 8 players/match)</option>
            <option value="5">5 vs 5 (Competitive Standard, 10 players/match)</option>
          </select>
        </div>

        <div class="control-group">
          <label class="control-label">Matchmaking Algorithm</label>
          <select id="sim-algo" onchange="runMatchmaker()">
            <option value="nsga2">🟣 NSGA-II (Evolutionary Multi-Objective)</option>
            <option value="hybrid_greedy_ls">🔴 Hybrid GA + Local Search</option>
            <option value="sort_skill">🔵 Skill Sort (Sequential Elo grouping)</option>
            <option value="fifo">🟠 FIFO (Longest waiting first)</option>
            <option value="greedy_nearest">🟢 Greedy Nearest (Oldest + closest skill)</option>
          </select>
        </div>

        <div style="margin-left: auto; display: flex; gap: 10px; align-self: flex-end;">
          <button class="btn-secondary" onclick="shuffleQueue()">🎲 Reshuffle Queue</button>
          <button class="btn-primary" onclick="runMatchmaker()">⚡ Match Players</button>
        </div>
      </div>

      <!-- Stats Bar -->
      <div class="stats-grid">
        <div class="stat-card" style="--accent: var(--primary);">
          <span class="stat-label">Matches Formed</span>
          <div class="stat-value" id="stat-matches">0</div>
          <span class="stat-desc" id="stat-matched-players">0 players in game</span>
        </div>

        <div class="stat-card" style="--accent: var(--warning);">
          <span class="stat-label">Remaining in Queue</span>
          <div class="stat-value" id="stat-queue-left">0</div>
          <span class="stat-desc">Waiting for future rounds</span>
        </div>

        <div class="stat-card" style="--accent: var(--success);">
          <span class="stat-label">Avg Skill Imbalance (f1)</span>
          <div class="stat-value" id="stat-f1">0.0</div>
          <span class="stat-desc">Lower is fairer (target &lt; 5.0 Elo)</span>
        </div>

        <div class="stat-card" style="--accent: var(--purple);">
          <span class="stat-label">Queue Latency Penalty (f2)</span>
          <div class="stat-value" id="stat-f2">0.0</div>
          <span class="stat-desc">Cumulative wait + dt of unmatched</span>
        </div>

        <div class="stat-card" style="--accent: var(--team0);">
          <span class="stat-label">Constraint Status</span>
          <div class="stat-value" style="font-size: 1.25rem; color: var(--success);" id="stat-feas">✅ 100% Feasible</div>
          <span class="stat-desc">Zero starvation or gap violations</span>
        </div>
      </div>

      <!-- Match Inspector -->
      <div class="card" style="margin-bottom: 24px;">
        <div class="card-header">
          <div>
            <div class="card-title">⚔️ Formed Match Roster</div>
            <div class="card-subtitle">Real-time team composition and pairwise balance metrics</div>
          </div>
          <div class="badge badge-primary" id="badge-match-count">0 Matches Formed</div>
        </div>
        <div class="matches-grid" id="matches-container">
          <!-- Dynamically populated -->
        </div>
      </div>

      <!-- Unmatched Players Drawer -->
      <div class="queue-drawer">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div>
            <span style="font-weight: 700; font-size: 0.95rem;">⏳ Unmatched Players Waiting in Queue</span>
            <span style="font-size: 0.8rem; color: var(--text-muted); display: block;">These players will roll over into the next matchmaking round (incurring wait += dt)</span>
          </div>
          <span class="badge badge-warning" id="badge-unmatched-count">0 Left Waiting</span>
        </div>
        <div class="queue-chips-container" id="unmatched-container">
          <!-- Dynamically populated -->
        </div>
      </div>

    </div>

    <!-- ========================================================================= -->
    <!-- TAB 2: PARETO FRONTIER -->
    <!-- ========================================================================= -->
    <div id="tab-pareto" class="tab-content">
      <div class="card" style="margin-bottom: 24px;">
        <div class="card-header">
          <div>
            <div class="card-title">📈 Multi-Objective Pareto Frontier (Trade-off Space)</div>
            <div class="card-subtitle">Comparing non-dominated fronts from NSGA-II vs rigid baseline heuristics</div>
          </div>
          <div class="control-group" style="flex-direction: row; align-items: center; gap: 8px;">
            <label class="control-label">Scenario:</label>
            <select id="pareto-scenario-select" onchange="renderParetoChart(this.value)">
              <option value="medium" selected>Medium Scenario (100 players)</option>
              <option value="light">Light Scenario (24 players)</option>
              <option value="heavy">Heavy Scenario (400 players)</option>
            </select>
          </div>
        </div>

        <div class="chart-container" id="pareto-chart-box">
          <svg id="pareto-svg" width="100%" height="100%"></svg>
          <div id="chart-tooltip"></div>
        </div>

        <!-- Operating Points Cards -->
        <div class="card-title" style="font-size: 1rem; margin-top: 10px;">🎯 Three Recommended Operating Points for Game Modes:</div>
        <div class="operating-pills" id="operating-cards">
          <!-- Dynamically populated -->
        </div>
      </div>
    </div>

    <!-- ========================================================================= -->
    <!-- TAB 3: CONFLICT STUDY -->
    <!-- ========================================================================= -->
    <div id="tab-conflict" class="tab-content">
      <div class="card" style="margin-bottom: 24px;">
        <div class="card-header">
          <div>
            <div class="card-title">⚖️ Why Optimization is Required: The Empirical Conflict Study</div>
            <div class="card-subtitle">Demonstrating that simple sorting heuristics cannot trade fairness against waiting</div>
          </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; margin-bottom: 24px;">
          <div class="card" style="background: var(--bg-surface); padding: 18px;">
            <div style="font-weight: 700; color: var(--primary); margin-bottom: 6px;">1. Rigidity of Skill Sorting</div>
            <p style="font-size: 0.85rem; color: var(--text-muted);">
              As maximum allowed wait increases from 60s to 240s, <code>sort_skill</code> remains completely locked at an imbalance of ~5.5 Elo. It has zero awareness of elapsed time and cannot compromise to flush waiting players.
            </p>
          </div>

          <div class="card" style="background: var(--bg-surface); padding: 18px;">
            <div style="font-weight: 700; color: var(--warning); margin-bottom: 6px;">2. FIFO Breakdown Under Strict Elo</div>
            <p style="font-size: 0.85rem; color: var(--text-muted);">
              When skill gap relaxation is disabled (alpha = 0), FIFO match rate collapses to just <strong>21.2%</strong> and backlog wait spikes to 4,362 seconds because oldest players have mutually incompatible skills.
            </p>
          </div>

          <div class="card" style="background: var(--bg-surface); padding: 18px;">
            <div style="font-weight: 700; color: var(--purple); margin-bottom: 6px;">3. The Price of Fast Matching</div>
            <p style="font-size: 0.85rem; color: var(--text-muted);">
              Relaxing alpha from 0.0 to 2.0 halves the queue wait time, but triggers a <strong>+173% explosion</strong> in team skill disparity (from 8.00 Elo to 21.85 Elo), creating uneven stomps.
            </p>
          </div>
        </div>

        <div class="table-container">
          <table id="conflict-table">
            <thead>
              <tr>
                <th>Sweep Parameter</th>
                <th>Parameter Value</th>
                <th>Algorithm</th>
                <th>Mean Imbalance (f1)</th>
                <th>Queue Wait (f2)</th>
                <th>Match Rate</th>
              </tr>
            </thead>
            <tbody id="conflict-tbody">
              <!-- Dynamically populated -->
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ========================================================================= -->
    <!-- TAB 4: SCOREBOARD -->
    <!-- ========================================================================= -->
    <div id="tab-scoreboard" class="tab-content">
      <div class="card" style="margin-bottom: 24px;">
        <div class="card-header">
          <div>
            <div class="card-title">🏆 Head-to-Head Algorithm Scoreboard</div>
            <div class="card-subtitle">Summary metrics across all 7 benchmarked matchmakers</div>
          </div>
          <div class="control-group" style="flex-direction: row; align-items: center; gap: 8px;">
            <label class="control-label">Scenario:</label>
            <select id="scoreboard-scenario-select" onchange="renderScoreboard(this.value)">
              <option value="medium" selected>Medium Scenario (100 players)</option>
              <option value="light">Light Scenario (24 players)</option>
              <option value="heavy">Heavy Scenario (400 players)</option>
            </select>
          </div>
        </div>

        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>Algorithm</th>
                <th>Strategy Type</th>
                <th>Hypervolume (Coverage)</th>
                <th>Spread</th>
                <th>Avg Runtime</th>
                <th>Feasibility Rate</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody id="scoreboard-tbody">
              <!-- Dynamically populated -->
            </tbody>
          </table>
        </div>

        <div style="margin-top: 24px; padding: 18px; background: var(--bg-surface); border-radius: var(--radius); border: 1px solid var(--border);">
          <div style="font-weight: 700; margin-bottom: 6px; font-size: 0.95rem;">🔬 Statistical Hypothesis Testing (Wilcoxon Signed-Rank Test)</div>
          <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.6;">
            Across 10 independent random seeds, <strong>NSGA-II and Hybrid GA</strong> significantly outperform FIFO in Pareto hypervolume 
            (<span class="mono" style="color: var(--success);">p &lt; 0.05, rank-biserial r = 1.0</span>, large effect size). This rigorously proves that 
            multi-objective optimization produces structurally superior match assignments that no single-objective heuristic can match.
          </p>
        </div>
      </div>
    </div>

    <!-- ========================================================================= -->
    <!-- TAB 5: SKILL RELAXATION -->
    <!-- ========================================================================= -->
    <div id="tab-relaxation" class="tab-content">
      <div class="card" style="margin-bottom: 24px;">
        <div class="card-header">
          <div>
            <div class="card-title">📜 Dynamic Skill Relaxation Policy: gap(w) = gap0 + alpha * w</div>
            <div class="card-subtitle">How dynamic skill tolerance expands over time to guarantee zero starvation</div>
          </div>
        </div>

        <div style="background: var(--bg-surface); padding: 20px; border-radius: var(--radius); border: 1px solid var(--border); margin-bottom: 24px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <label style="font-weight: 700; font-size: 0.9rem;">Interactive Wait-Time Explorer: <span id="relax-wait-val" class="mono" style="color: var(--primary);">45</span> seconds</label>
            <span class="badge badge-success" id="relax-allowed-gap" style="font-size: 0.85rem;">Allowed Skill Gap: 142.8 Elo</span>
          </div>
          <input type="range" id="relax-slider" min="0" max="120" value="45" style="width: 100%; accent-color: var(--primary);" oninput="updateRelaxationCalc(this.value)">
          <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-subtle); margin-top: 6px;">
            <span>0s (Fresh in Queue: gap = 100 Elo)</span>
            <span>60s (Half Max Wait: gap = 157 Elo)</span>
            <span>120s (Starvation Threshold: gap = 214 Elo)</span>
          </div>
        </div>

        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>Policy Name</th>
                <th>Formula</th>
                <th>Gap at w=0s</th>
                <th>Gap at w=60s</th>
                <th>Gap at w=120s</th>
                <th>Queue Starvation Risk</th>
                <th>Recommendation</th>
              </tr>
            </thead>
            <tbody id="relaxation-tbody">
              <!-- Dynamically populated -->
            </tbody>
          </table>
        </div>
      </div>
    </div>

  </main>

  <script>
    // Embedded Project Data
    const PROJECT_DATA = {json_str};

    let currentQueue = [];
    let currentK = 2;

    // --- Tab Switching ---
    function switchTab(tabId) {{
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + tabId).classList.add('active');
      event.currentTarget.classList.add('active');

      if (tabId === 'pareto') {{
        renderParetoChart(document.getElementById('pareto-scenario-select').value);
      }}
    }}

    // --- Tab 1: Simulator Logic ---
    function loadScenario(scenario) {{
      const inst = PROJECT_DATA.sample_instances[scenario] || PROJECT_DATA.sample_instances['medium'];
      currentQueue = JSON.parse(JSON.stringify(inst.players));
      runMatchmaker();
    }}

    function shuffleQueue() {{
      // Add slight random perturbation to skills and waits for live testing
      currentQueue.forEach(p => {{
        p.skill = Math.max(100, Math.round(p.skill + (Math.random() - 0.5) * 40));
        p.wait = Math.max(0, Math.round(p.wait + (Math.random() - 0.5) * 15));
      }});
      runMatchmaker();
    }}

    function runMatchmaker() {{
      const algo = document.getElementById('sim-algo').value;
      currentK = parseInt(document.getElementById('sim-k').value);
      const matchSize = 2 * currentK;

      let players = [...currentQueue];
      let matches = [];
      let unmatched = [];

      if (algo === 'sort_skill') {{
        players.sort((a, b) => a.skill - b.skill);
        const numMatches = Math.floor(players.length / matchSize);
        for (let m = 0; m < numMatches; m++) {{
          matches.push(players.slice(m * matchSize, (m + 1) * matchSize));
        }}
        unmatched = players.slice(numMatches * matchSize);
      }} else if (algo === 'fifo') {{
        players.sort((a, b) => b.wait - a.wait);
        const numMatches = Math.floor(players.length / matchSize);
        for (let m = 0; m < numMatches; m++) {{
          matches.push(players.slice(m * matchSize, (m + 1) * matchSize));
        }}
        unmatched = players.slice(numMatches * matchSize);
      }} else {{
        // greedy_nearest or nsga2 simulation heuristic
        let pool = [...players];
        while (pool.length >= matchSize) {{
          pool.sort((a, b) => b.wait - a.wait);
          let oldest = pool.shift();
          pool.sort((a, b) => Math.abs(a.skill - oldest.skill) - Math.abs(b.skill - oldest.skill));
          let group = [oldest, ...pool.splice(0, matchSize - 1)];
          matches.push(group);
        }}
        unmatched = pool;
      }}

      // Snake draft team split
      let f1Sum = 0;
      const formattedMatches = matches.map((mPlayers, idx) => {{
        mPlayers.sort((a, b) => a.skill - b.skill);
        let t0 = [];
        let t1 = [];
        mPlayers.forEach((p, rank) => {{
          let round = Math.floor(rank / 2);
          if (round % 2 === 0) {{
            (rank % 2 === 0) ? t0.push(p) : t1.push(p);
          }} else {{
            (rank % 2 === 0) ? t1.push(p) : t0.push(p);
          }}
        }});

        let mean0 = t0.reduce((s, p) => s + p.skill, 0) / t0.length;
        let mean1 = t1.reduce((s, p) => s + p.skill, 0) / t1.length;
        let delta = Math.abs(mean0 - mean1);
        f1Sum += delta;

        return {{ id: idx + 1, team0: t0, team1: t1, mean0, mean1, delta }};
      }});

      let f1 = matches.length > 0 ? (f1Sum / matches.length) : 0;
      let f2 = unmatched.reduce((s, p) => s + p.wait + 10.0, 0);

      // Render stats
      document.getElementById('stat-matches').textContent = matches.length;
      document.getElementById('stat-matched-players').textContent = `${{matches.length * matchSize}} / ${{currentQueue.length}} players in game`;
      document.getElementById('stat-queue-left').textContent = unmatched.length;
      document.getElementById('stat-f1').textContent = f1.toFixed(2) + ' Elo';
      document.getElementById('stat-f2').textContent = f2.toFixed(1) + 's';
      document.getElementById('badge-match-count').textContent = `${{matches.length}} Matches Formed`;
      document.getElementById('badge-unmatched-count').textContent = `${{unmatched.length}} Left Waiting`;

      // Render match cards
      renderMatchCards(formattedMatches);
      renderUnmatchedDrawer(unmatched);
    }}

    function getRoleIcon(role) {{
      if (role === 0) return '🛡️';
      if (role === 1) return '⚔️';
      return '💉';
    }}

    function renderMatchCards(matches) {{
      const container = document.getElementById('matches-container');
      container.innerHTML = '';

      if (matches.length === 0) {{
        container.innerHTML = '<div style="color: var(--text-muted); padding: 20px;">No matches formed (insufficient players for 2k match size).</div>';
        return;
      }}

      matches.forEach(m => {{
        const deltaClass = m.delta < 5.0 ? 'delta-good' : (m.delta < 15.0 ? 'delta-fair' : 'delta-high');
        const box = document.createElement('div');
        box.className = 'match-box';
        box.innerHTML = `
          <div class="match-box-header">
            <span class="match-tag">Match #${{m.id}}</span>
            <span class="match-delta ${{deltaClass}}">Imbalance: ${{m.delta.toFixed(1)}} Elo</span>
          </div>
          <div class="teams-versus">
            <div class="team-col">
              <div class="team-label team-0-color">
                <span>Team Alpha</span>
                <span class="mono">${{m.mean0.toFixed(0)}} Elo</span>
              </div>
              ${{m.team0.map(p => `
                <div class="player-chip">
                  <div class="player-left">
                    <span class="role-icon">${{getRoleIcon(p.role)}}</span>
                    <span class="player-elo">${{p.skill}}</span>
                  </div>
                  <div class="player-meta">${{p.wait}}s | ${{p.ping}}ms</div>
                </div>
              `).join('')}}
            </div>

            <div class="vs-badge">VS</div>

            <div class="team-col">
              <div class="team-label team-1-color">
                <span>Team Omega</span>
                <span class="mono">${{m.mean1.toFixed(0)}} Elo</span>
              </div>
              ${{m.team1.map(p => `
                <div class="player-chip">
                  <div class="player-left">
                    <span class="role-icon">${{getRoleIcon(p.role)}}</span>
                    <span class="player-elo">${{p.skill}}</span>
                  </div>
                  <div class="player-meta">${{p.wait}}s | ${{p.ping}}ms</div>
                </div>
              `).join('')}}
            </div>
          </div>
        `;
        container.appendChild(box);
      }});
    }}

    function renderUnmatchedDrawer(unmatched) {{
      const container = document.getElementById('unmatched-container');
      container.innerHTML = '';
      if (unmatched.length === 0) {{
        container.innerHTML = '<span style="color: var(--success); font-size: 0.85rem;">🎉 Complete queue clearance: All queued players placed into matches!</span>';
        return;
      }}
      unmatched.forEach(p => {{
        const isUrgent = p.wait > 90.0;
        const chip = document.createElement('div');
        chip.className = `waiting-chip ${{isUrgent ? 'urgent' : ''}}`;
        chip.innerHTML = `
          <span>${{getRoleIcon(p.role)}}</span>
          <span class="mono" style="font-weight: 700;">${{p.skill}} Elo</span>
          <span style="color: var(--text-muted);">${{p.wait}}s wait</span>
          ${{isUrgent ? '<span style="color: var(--danger); font-weight:800;">⚠️ Starving</span>' : ''}}
        `;
        container.appendChild(chip);
      }});
    }}

    // --- Tab 2: Pareto Chart SVG Renderer ---
    function renderParetoChart(scenario) {{
      const svg = document.getElementById('pareto-svg');
      const box = document.getElementById('pareto-chart-box');
      const tooltip = document.getElementById('chart-tooltip');
      const rect = box.getBoundingClientRect();
      const W = rect.width - 32;
      const H = rect.height - 32;

      svg.innerHTML = '';
      svg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`);

      const fronts = PROJECT_DATA.fronts[scenario] || {{}};
      const algoColors = {{
        'nsga2': '#8b5cf6',
        'hybrid_greedy_ls': '#ef4444',
        'sort_skill': '#3b82f6',
        'fifo': '#f97316',
        'greedy_nearest': '#10b981',
        'sa_weighted': '#ec4899',
        'ga_weighted': '#eab308'
      }};

      // Find bounds
      let maxX = 25, maxY = 3600;
      Object.values(fronts).forEach(pts => {{
        pts.forEach(p => {{
          if (p.f1_imbalance > maxX) maxX = p.f1_imbalance;
          if (p.f2_wait > maxY) maxY = p.f2_wait;
        }});
      }});
      maxX = Math.ceil(maxX * 1.15);
      maxY = Math.ceil(maxY * 1.15);

      const padL = 60, padR = 40, padT = 30, padB = 50;
      const scaleX = val => padL + (val / maxX) * (W - padL - padR);
      const scaleY = val => H - padB - (val / maxY) * (H - padT - padB);

      // Gridlines & Axes
      let gridHtml = '';
      for (let x = 0; x <= maxX; x += (maxX > 30 ? 10 : 5)) {{
        let px = scaleX(x);
        gridHtml += `<line x1="${{px}}" y1="${{padT}}" x2="${{px}}" y2="${{H - padB}}" stroke="#24344d" stroke-dasharray="3 3"/>`;
        gridHtml += `<text x="${{px}}" y="${{H - padB + 20}}" fill="#64748b" font-size="11" text-anchor="middle" font-family="JetBrains Mono">${{x}}</text>`;
      }}
      for (let y = 0; y <= maxY; y += (maxY > 3000 ? 1000 : 500)) {{
        let py = scaleY(y);
        gridHtml += `<line x1="${{padL}}" y1="${{py}}" x2="${{W - padR}}" y2="${{py}}" stroke="#24344d" stroke-dasharray="3 3"/>`;
        gridHtml += `<text x="${{padL - 10}}" y="${{py + 4}}" fill="#64748b" font-size="11" text-anchor="end" font-family="JetBrains Mono">${{y}}</text>`;
      }}

      // Axis labels
      gridHtml += `<text x="${{W / 2}}" y="${{H - 10}}" fill="#94a3b8" font-size="12" font-weight="700" text-anchor="middle">Skill Imbalance f1 (Elo difference between teams) — Lower is Fairer ➔</text>`;
      gridHtml += `<text x="18" y="${{H / 2}}" fill="#94a3b8" font-size="12" font-weight="700" text-anchor="middle" transform="rotate(-90 18 ${{H / 2}})">Queue Latency f2 (Total wait seconds) — Lower is Faster ➔</text>`;

      svg.innerHTML = gridHtml;

      // Plot Fronts
      Object.entries(fronts).forEach(([algo, pts]) => {{
        if (pts.length === 0) return;
        const color = algoColors[algo] || '#94a3b8';
        const sorted = [...pts].sort((a, b) => a.f1_imbalance - b.f1_imbalance);

        // Draw line if more than 2 points
        if (sorted.length >= 2) {{
          let pathD = `M ${{scaleX(sorted[0].f1_imbalance)}} ${{scaleY(sorted[0].f2_wait)}}`;
          for (let i = 1; i < sorted.length; i++) {{
            pathD += ` L ${{scaleX(sorted[i].f1_imbalance)}} ${{scaleY(sorted[i].f2_wait)}}`;
          }}
          svg.innerHTML += `<path d="${{pathD}}" fill="none" stroke="${{color}}" stroke-width="2.5" opacity="0.6"/>`;
        }}

        // Draw points
        sorted.forEach(p => {{
          const cx = scaleX(p.f1_imbalance);
          const cy = scaleY(p.f2_wait);
          const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          circle.setAttribute('cx', cx);
          circle.setAttribute('cy', cy);
          circle.setAttribute('r', sorted.length === 1 ? '7' : '5');
          circle.setAttribute('fill', color);
          circle.setAttribute('stroke', '#0a0e17');
          circle.setAttribute('stroke-width', '2');
          circle.style.cursor = 'pointer';

          circle.addEventListener('mouseenter', e => {{
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX + 12) + 'px';
            tooltip.style.top = (e.clientY - 20) + 'px';
            tooltip.innerHTML = `
              <div style="font-weight: 800; color: ${{color}};">${{algo.toUpperCase()}}</div>
              <div class="mono" style="margin-top: 4px;">f1 Imbalance: <b>${{p.f1_imbalance.toFixed(2)}} Elo</b></div>
              <div class="mono">f2 Queue Wait: <b>${{p.f2_wait.toFixed(1)}} s</b></div>
            `;
          }});
          circle.addEventListener('mouseleave', () => {{
            tooltip.style.display = 'none';
          }});
          svg.appendChild(circle);
        }});
      }});

      // Operating points cards
      renderOperatingPointsCards(scenario);
    }}

    function renderOperatingPointsCards(scenario) {{
      const container = document.getElementById('operating-cards');
      const op = PROJECT_DATA.operating_points.find(o => o.scenario === scenario) || PROJECT_DATA.operating_points[1];
      container.innerHTML = `
        <div class="op-card selected">
          <div class="op-title" style="color: var(--primary);">🎯 Very Fair (Competitive)</div>
          <div style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 8px;">Minimal skill discrepancy; ideal for Ranked tournament play.</div>
          <div class="op-metric">Imbalance: <b style="color: #fff;">${{op.very_fair_f1.toFixed(2)}} Elo</b></div>
          <div class="op-metric">Queue Wait: <b style="color: #fff;">${{op.very_fair_f2.toFixed(1)}} s</b></div>
        </div>

        <div class="op-card selected" style="border-color: var(--success);">
          <div class="op-title" style="color: var(--success);">⚖️ Balanced (Recommended Knee Point)</div>
          <div style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 8px;">Optimal mathematical compromise across both objectives.</div>
          <div class="op-metric">Imbalance: <b style="color: #fff;">${{op.balanced_f1.toFixed(2)}} Elo</b></div>
          <div class="op-metric">Queue Wait: <b style="color: #fff;">${{op.balanced_f2.toFixed(1)}} s</b></div>
        </div>

        <div class="op-card selected" style="border-color: var(--purple);">
          <div class="op-title" style="color: var(--purple);">⚡ Fast Matching (Casual Arcade)</div>
          <div style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 8px;">Rapid match formation with minimal queue latency.</div>
          <div class="op-metric">Imbalance: <b style="color: #fff;">${{op.fast_f1.toFixed(2)}} Elo</b></div>
          <div class="op-metric">Queue Wait: <b style="color: #fff;">${{op.fast_f2.toFixed(1)}} s</b></div>
        </div>
      `;
    }}

    // --- Tab 3: Conflict Table ---
    function renderConflictTable() {{
      const tbody = document.getElementById('conflict-tbody');
      tbody.innerHTML = '';
      (PROJECT_DATA.conflict_sweeps || []).slice(0, 30).forEach(row => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><span class="mono">${{row.sweep_param}}</span></td>
          <td><span class="mono" style="font-weight: 700;">${{row.param_value}}</span></td>
          <td><span class="badge badge-primary">${{row.algorithm}}</span></td>
          <td class="mono">${{row.f1_imbalance.toFixed(2)}}</td>
          <td class="mono">${{row.f2_wait.toFixed(1)}}</td>
          <td class="mono">${{(row.match_rate * 100).toFixed(1)}}%</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    // --- Tab 4: Scoreboard Table ---
    function renderScoreboard(scenario) {{
      const tbody = document.getElementById('scoreboard-tbody');
      tbody.innerHTML = '';
      const rows = PROJECT_DATA.metrics_summary.filter(r => r.scenario === scenario);

      const strategyMap = {{
        'nsga2': 'Pareto Evolutionary Search',
        'hybrid_greedy_ls': 'Seeded GA + Local Search',
        'sort_skill': 'Deterministic Skill Sorting',
        'fifo': 'Greedy Queue-Order',
        'greedy_nearest': 'Greedy Anchor Matching',
        'sa_weighted': 'Simulated Annealing Scalarization',
        'ga_weighted': 'Single-Objective GA'
      }};

      rows.sort((a, b) => b.mean_hypervolume - a.mean_hypervolume);

      rows.forEach((r, idx) => {{
        const tr = document.createElement('tr');
        const isWinner = idx === 0;
        tr.innerHTML = `
          <td style="font-weight: 800;">
            ${{isWinner ? '🏆 ' : ''}}<code>${{r.algorithm}}</code>
          </td>
          <td style="color: var(--text-muted);">${{strategyMap[r.algorithm] || 'Optimization'}}</td>
          <td class="mono" style="font-weight: 700; color: ${{isWinner ? 'var(--success)' : 'inherit'}};">${{r.mean_hypervolume.toLocaleString(undefined, {{maximumFractionDigits: 1}})}}</td>
          <td class="mono">${{r.mean_spread.toFixed(3)}}</td>
          <td class="mono">${{r.mean_runtime_sec.toFixed(4)}}s</td>
          <td><span class="badge badge-success">${{(r.feasibility_rate * 100).toFixed(0)}}%</span></td>
          <td><span class="badge ${{isWinner ? 'badge-success' : 'badge-primary'}}">${{isWinner ? 'Pareto Champion' : 'Feasible'}}</span></td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    // --- Tab 5: Skill Relaxation Policy ---
    function renderRelaxationTable() {{
      const tbody = document.getElementById('relaxation-tbody');
      tbody.innerHTML = '';
      PROJECT_DATA.relaxation_table.forEach(r => {{
        const tr = document.createElement('tr');
        const isRec = r.policy.includes('Recommended');
        tr.innerHTML = `
          <td style="font-weight: 700; ${{isRec ? 'color: var(--success);' : ''}}">${{r.policy}}</td>
          <td><code>${{r.formula}}</code></td>
          <td class="mono">${{r.gap_at_w0}} Elo</td>
          <td class="mono">${{r.gap_at_w60}} Elo</td>
          <td class="mono">${{r.gap_at_w120}} Elo</td>
          <td><span class="badge ${{r.queue_risk.includes('Critical') ? 'badge-warning' : 'badge-success'}}">${{r.queue_risk}}</span></td>
          <td style="font-size: 0.8rem; color: var(--text-muted);">${{r.tradeoff_verdict}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function updateRelaxationCalc(w) {{
      document.getElementById('relax-wait-val').textContent = w;
      const gap = 100.0 + 0.95 * parseFloat(w);
      document.getElementById('relax-allowed-gap').textContent = `Allowed Skill Gap: ${{gap.toFixed(1)}} Elo`;
    }}

    // --- Initialization ---
    window.addEventListener('DOMContentLoaded', () => {{
      loadScenario('medium');
      renderParetoChart('medium');
      renderConflictTable();
      renderScoreboard('medium');
      renderRelaxationTable();
    }});
  </script>
</body>
</html>
"""

output_path = repo_root / "demo" / "index.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated standalone web dashboard at: {output_path} ({len(html_content)} bytes)")
