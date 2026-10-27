---
name: gate-check
description: Run gate acceptance checks for a member (ownership check, member pytest suite, git sync check with origin/develop)
---

# Gate Check Workflow

This workflow executes the pre-PR gate validation checklist described in `README.md` (Section 5 & 12).

## Steps for the Agent

1. **Identify Member Number:**
   Ask the user for their Member Number (`1` for Krish, `2` for Sarvesh, `3` for Vedant, `4` for RAM) or infer it from the active branch name (e.g. `m1/...`).

2. **Run Ownership Check:**
   Execute:
   ```bash
   python scripts/check_owner.py <MEMBER_NUMBER>
   ```
   *(Note: For Member 1 completing Checkpoint G0 only, add `--gate G0`)*.
   If the command fails (non-zero exit code), immediately report the offending files and print **FAIL**.

3. **Run Member Tests:**
   Run `pytest -q` on the test files owned by this member:
   - **M1 (Krish):** `pytest -q tests/test_types.py tests/test_registry.py` (and any existing `tests/test_simulator.py`, `tests/test_validate.py`, `tests/test_rolling*.py`)
   - **M2 (Sarvesh):** `pytest -q tests/test_objectives*.py tests/test_baselines*.py tests/test_metrics*.py tests/test_stats*.py`
   - **M3 (Vedant):** `pytest -q tests/test_encoding*.py tests/test_repair*.py tests/test_nsga2*.py tests/test_weighted*.py tests/test_hybrid*.py`
   - **M4 (RAM):** `pytest -q tests/test_pareto*.py tests/test_plots*.py tests/test_relaxation*.py`

   If any test fails, report the traceback and print **FAIL**.

4. **Git Sync Check:**
   Execute:
   ```bash
   git fetch origin
   ```
   Inspect whether the current branch is behind `origin/develop`:
   ```bash
   git rev-list --left-right --count HEAD...origin/develop
   ```
   - If behind `origin/develop` (second count > 0):
     Report: **Branch is behind origin/develop.**
     Instruct the user to rebase manually:
     ```bash
     git rebase origin/develop
     ```
     **DO NOT run git rebase for the user.**
     Print **FAIL (Branch needs rebase)**.

5. **Verdict:**
   - If all checks pass and branch is up to date:
     Print **PASS: Gate checks passed. You are ready to push and open a Pull Request.**
   - Otherwise, print **FAIL** with a bulleted list of issues to resolve.
