#!/usr/bin/env python3
# OWNER: M1 - do not edit unless you are M1
"""Ownership check script.

Verifies that changed files compared to origin/develop strictly belong to the
specified team member according to config/owners.yaml.
"""
from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
import subprocess
import sys
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


def load_owners_config(config_path: Path) -> tuple[dict[int, list[str]], list[str]]:
    """Load member path ownership and shared append-only paths from owners.yaml."""
    if not config_path.exists():
        print(f"Error: owners config not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    content = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(content) or {}
    else:
        # Fallback parser for owners.yaml when PyYAML is not installed
        data = _parse_owners_fallback(content)

    members_data = data.get("members", {})
    member_paths: dict[int, list[str]] = {}
    for m_id, m_info in members_data.items():
        try:
            m_num = int(m_id)
        except ValueError:
            continue
        paths = m_info.get("paths", []) if isinstance(m_info, dict) else []
        member_paths[m_num] = [p.replace("\\", "/").strip() for p in paths]

    shared = data.get("shared_append_only", [])
    shared_paths = [p.replace("\\", "/").strip() for p in shared]
    return member_paths, shared_paths


def _parse_owners_fallback(text: str) -> dict[str, Any]:
    """Simple parser for owners.yaml structure without external dependencies."""
    result: dict[str, Any] = {"members": {}, "shared_append_only": []}
    current_section = None
    current_member = None
    in_paths = False

    for raw_line in text.splitlines():
        line = raw_line.split("#")[0].rstrip()
        if not line:
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        if indent == 0 and stripped.startswith("members:"):
            current_section = "members"
            current_member = None
            in_paths = False
        elif indent == 0 and stripped.startswith("shared_append_only:"):
            current_section = "shared_append_only"
            current_member = None
            in_paths = False
        elif current_section == "shared_append_only":
            if stripped.startswith("-"):
                val = stripped.lstrip("-").strip().strip("\"'")
                result["shared_append_only"].append(val)
        elif current_section == "members":
            if indent == 2 and ":" in stripped and not stripped.startswith("-"):
                k = stripped.split(":", 1)[0].strip()
                try:
                    current_member = int(k)
                    result["members"][current_member] = {"paths": []}
                    in_paths = False
                except ValueError:
                    current_member = None
            elif indent == 4 and stripped.startswith("paths:"):
                in_paths = True
            elif indent >= 6 and in_paths and stripped.startswith("-") and current_member is not None:
                val = stripped.lstrip("-").strip().strip("\"'")
                result["members"][current_member]["paths"].append(val)

    return result


def path_matches_pattern(file_path: str, pattern: str) -> bool:
    """Check if file_path matches an ownership pattern."""
    norm_path = file_path.replace("\\", "/").lstrip("./")
    norm_pat = pattern.replace("\\", "/").lstrip("./")

    if norm_pat.endswith("/"):
        return norm_path.startswith(norm_pat)
    if any(ch in norm_pat for ch in "*?[]"):
        return fnmatch.fnmatch(norm_path, norm_pat)
    return norm_path == norm_pat


def get_changed_files(base_ref: str = "origin/develop") -> list[str]:
    """Retrieve list of modified, added, or untracked files compared to base_ref."""
    # Check if base_ref is available
    check_ref = subprocess.run(
        ["git", "rev-parse", "--verify", base_ref],
        capture_output=True,
        text=True,
    )

    ref_to_diff = base_ref
    if check_ref.returncode != 0:
        # Fallback candidates if origin/develop is not yet fetched
        for candidate in ["develop", "origin/main", "main", "HEAD"]:
            c_res = subprocess.run(
                ["git", "rev-parse", "--verify", candidate],
                capture_output=True,
                text=True,
            )
            if c_res.returncode == 0:
                ref_to_diff = candidate
                break

    changed: set[str] = set()

    # Diff against base revision
    diff_res = subprocess.run(
        ["git", "diff", "--name-only", ref_to_diff],
        capture_output=True,
        text=True,
    )
    if diff_res.returncode == 0:
        for line in diff_res.stdout.splitlines():
            if line.strip():
                changed.add(line.strip().replace("\\", "/"))

    # Also include untracked and staged files
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
    )
    if untracked.returncode == 0:
        for line in untracked.stdout.splitlines():
            if line.strip():
                changed.add(line.strip().replace("\\", "/"))

    return sorted(changed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify file ownership against config/owners.yaml.")
    parser.add_argument("member", type=int, help="Member number (1, 2, 3, or 4)")
    parser.add_argument("--gate", type=str, default=None, help="Gate identifier, e.g. G0")
    parser.add_argument("--base", type=str, default="origin/develop", help="Base git ref (default: origin/develop)")

    args = parser.parse_args()

    # G0 exception: Member 1 is authorized to touch all paths during scaffold creation
    if args.gate and args.gate.upper() == "G0":
        if args.member == 1:
            print("PASS: Member 1 authorized for all paths under gate G0 (scaffold commit).")
            sys.exit(0)
        else:
            print("Warning: --gate G0 only applies to Member 1 (scaffold creator). Applying strict ownership check.")

    repo_root = Path(__file__).resolve().parents[1]
    owners_yaml = repo_root / "config" / "owners.yaml"
    member_paths, shared_paths = load_owners_config(owners_yaml)

    if args.member not in member_paths:
        print(f"FAIL: Unknown member number {args.member}. Valid members: {sorted(member_paths.keys())}", file=sys.stderr)
        sys.exit(1)

    allowed_patterns = member_paths[args.member] + shared_paths
    changed_files = get_changed_files(args.base)

    if not changed_files:
        print(f"PASS: No changed files detected compared to {args.base}.")
        sys.exit(0)

    offending_files = []
    for f in changed_files:
        if not any(path_matches_pattern(f, pat) for pat in allowed_patterns):
            offending_files.append(f)

    if offending_files:
        print(f"FAIL: Member {args.member} touched {len(offending_files)} file(s) they do not own:")
        for off in offending_files:
            print(f"  - {off}")
        sys.exit(1)

    print(f"PASS: All {len(changed_files)} touched file(s) belong to Member {args.member}.")
    sys.exit(0)


if __name__ == "__main__":
    main()
