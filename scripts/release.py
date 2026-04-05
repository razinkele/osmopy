#!/usr/bin/env python3
"""Bump version and generate/update CHANGELOG.md from conventional commits.

Usage:
    python scripts/release.py patch   # 0.2.0 -> 0.2.1
    python scripts/release.py minor   # 0.2.0 -> 0.3.0
    python scripts/release.py major   # 0.2.0 -> 1.0.0
    python scripts/release.py --changelog-only  # regenerate changelog without bumping

Reads version from osmose/__version__.py (single source of truth).
Generates CHANGELOG.md from git log using conventional commit prefixes.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "osmose" / "__version__.py"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"

# Conventional commit categories
CATEGORIES = {
    "feat": "Features",
    "fix": "Bug Fixes",
    "perf": "Performance",
    "refactor": "Refactoring",
    "test": "Tests",
    "docs": "Documentation",
    "style": "Styling",
    "chore": "Chores",
    "ci": "CI/CD",
}


def read_version() -> str:
    """Read current version from osmose/__version__.py."""
    text = VERSION_FILE.read_text()
    m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not m:
        print(f"ERROR: Could not parse version from {VERSION_FILE}", file=sys.stderr)
        sys.exit(1)
    return m.group(1)


def write_version(version: str) -> None:
    """Write version to osmose/__version__.py."""
    VERSION_FILE.write_text(
        f'"""Single source of truth for the OSMOSE Python package version."""\n\n'
        f'__version__ = "{version}"\n'
    )


def bump_version(current: str, part: str) -> str:
    """Bump version by part (major, minor, patch)."""
    parts = current.split(".")
    if len(parts) != 3:
        print(f"ERROR: Version {current!r} is not semver (X.Y.Z)", file=sys.stderr)
        sys.exit(1)
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    if part == "major":
        return f"{major + 1}.0.0"
    elif part == "minor":
        return f"{major}.{minor + 1}.0"
    elif part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        print(f"ERROR: Unknown bump part {part!r}", file=sys.stderr)
        sys.exit(1)


def git_tags() -> list[str]:
    """Return list of version tags sorted by version."""
    result = subprocess.run(["git", "tag", "-l", "v*"], capture_output=True, text=True, cwd=ROOT)
    tags = [t.strip() for t in result.stdout.splitlines() if t.strip()]
    return sorted(tags, key=lambda t: [int(x) for x in t.lstrip("v").split(".")])


def git_log_between(from_ref: str | None, to_ref: str = "HEAD") -> list[dict]:
    """Get commits between two refs, parsed into conventional commit format."""
    if from_ref:
        range_spec = f"{from_ref}..{to_ref}"
    else:
        range_spec = to_ref

    result = subprocess.run(
        ["git", "log", range_spec, "--pretty=format:%H|%s"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    commits = []
    pattern = re.compile(r"^(\w+)(?:\(([^)]+)\))?\s*:\s*(.+)$")
    for line in result.stdout.splitlines():
        if "|" not in line:
            continue
        sha, subject = line.split("|", 1)
        m = pattern.match(subject)
        if m:
            commits.append(
                {
                    "sha": sha[:7],
                    "type": m.group(1),
                    "scope": m.group(2),
                    "description": m.group(3).strip(),
                }
            )
        else:
            commits.append(
                {
                    "sha": sha[:7],
                    "type": "other",
                    "scope": None,
                    "description": subject.strip(),
                }
            )
    return commits


def format_section(commits: list[dict]) -> str:
    """Format commits into markdown sections by category."""
    by_category: dict[str, list[dict]] = {}
    for c in commits:
        cat = CATEGORIES.get(c["type"], "Other")
        by_category.setdefault(cat, []).append(c)

    # Order: Features first, then Bug Fixes, then rest alphabetically
    priority = ["Features", "Bug Fixes", "Performance"]
    ordered = []
    for cat in priority:
        if cat in by_category:
            ordered.append((cat, by_category.pop(cat)))
    for cat in sorted(by_category.keys()):
        ordered.append((cat, by_category[cat]))

    lines = []
    for cat, cat_commits in ordered:
        lines.append(f"### {cat}\n")
        for c in cat_commits:
            scope = f"**{c['scope']}:** " if c["scope"] else ""
            lines.append(f"- {scope}{c['description']} ({c['sha']})")
        lines.append("")
    return "\n".join(lines)


def generate_changelog() -> str:
    """Generate full CHANGELOG.md content from git history."""
    tags = git_tags()
    current_version = read_version()

    sections = []

    # Header
    sections.append("# Changelog\n")
    sections.append("All notable changes to this project will be documented in this file.\n")
    sections.append(
        "Format based on [Keep a Changelog](https://keepachangelog.com/), "
        "generated from [Conventional Commits](https://www.conventionalcommits.org/).\n"
    )

    # Unreleased / current version (commits since last tag)
    last_tag = tags[-1] if tags else None
    unreleased = git_log_between(last_tag, "HEAD")
    if unreleased:
        title = f"## [{current_version}] - {date.today().isoformat()}"
        sections.append(f"{title}\n")
        sections.append(format_section(unreleased))

    # Previous tagged releases
    for i in range(len(tags) - 1, 0, -1):
        tag = tags[i]
        prev_tag = tags[i - 1]
        version = tag.lstrip("v")
        commits = git_log_between(prev_tag, tag)
        if commits:
            # Try to get tag date
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ai", tag],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            tag_date = result.stdout.strip()[:10] if result.stdout.strip() else "unknown"
            sections.append(f"## [{version}] - {tag_date}\n")
            sections.append(format_section(commits))

    # First tag (all commits before it)
    if tags:
        first_tag = tags[0]
        version = first_tag.lstrip("v")
        commits = git_log_between(None, first_tag)
        if commits:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ai", first_tag],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            tag_date = result.stdout.strip()[:10] if result.stdout.strip() else "unknown"
            sections.append(f"## [{version}] - {tag_date}\n")
            sections.append(format_section(commits))

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description="Bump version and generate changelog")
    parser.add_argument(
        "part",
        nargs="?",
        choices=["major", "minor", "patch"],
        help="Version part to bump",
    )
    parser.add_argument(
        "--changelog-only",
        action="store_true",
        help="Only regenerate changelog, don't bump version",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without making changes",
    )
    args = parser.parse_args()

    if not args.changelog_only and not args.part:
        parser.error("Specify a version part (major/minor/patch) or --changelog-only")

    current = read_version()

    if args.changelog_only:
        new_version = current
        print(f"Regenerating changelog for v{current}")
    else:
        new_version = bump_version(current, args.part)
        print(f"Bumping version: {current} -> {new_version}")

    if args.dry_run:
        print("\n[DRY RUN] Would update:")
        print(f"  {VERSION_FILE} -> {new_version}")
        print(f"  {CHANGELOG_FILE} -> regenerated")
        print(f"  git tag v{new_version}")
        return

    # Write new version
    if not args.changelog_only:
        write_version(new_version)
        print(f"  Updated {VERSION_FILE.relative_to(ROOT)}")

    # Generate changelog
    changelog = generate_changelog()
    CHANGELOG_FILE.write_text(changelog)
    print(f"  Generated {CHANGELOG_FILE.relative_to(ROOT)}")

    print(f"\nVersion: {new_version}")
    print("Next steps:")
    print("  git add osmose/__version__.py CHANGELOG.md")
    print(f'  git commit -m "release: v{new_version}"')
    print(f"  git tag v{new_version}")
    print("  git push origin master --tags")


if __name__ == "__main__":
    main()
