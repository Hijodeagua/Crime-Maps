"""
Regression tests for the snapshot data relay.

The deployed Streamlit Cloud app cannot reach the city APIs directly; it
serves real data from snapshots that the nightly refresh workflow commits
under data/cache/. That relay silently no-opped for months because
data/cache/ was gitignored — `git add data/cache` staged nothing, so no
snapshot was ever committed and the app fell back to bundled fixtures.

These tests pin the invariant: the directory cache.py reads/writes must be
committable.
"""

import subprocess
from pathlib import Path

import pytest

from crimemaps import cache

REPO_ROOT = Path(__file__).parents[1]


def _check_ignore(path: str) -> int:
    """Return git check-ignore's exit code: 0=ignored, 1=not ignored."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode


@pytest.fixture(autouse=True)
def require_git_repo():
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not running inside a git checkout")
    probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip("git unavailable")


def test_snapshot_parquet_path_is_not_gitignored():
    # Use the actual base directory cache.py writes to, so this test follows
    # the code if the layout ever moves.
    snapshot = (
        cache._BASE / "charlotte" / "cmpd_incidents"
        / "20260101T000000__2026-01-01_2026-01-02" / "data.parquet"
    )
    rel = snapshot.relative_to(REPO_ROOT).as_posix()
    assert _check_ignore(rel) == 1, (
        f"{rel} is gitignored — the nightly refresh workflow can never commit "
        "snapshots, and the deployed app will silently serve stale/fixture data"
    )


def test_manifest_path_is_not_gitignored():
    manifest = cache._BASE / "charlotte" / "cmpd_incidents" / "manifest.jsonl"
    rel = manifest.relative_to(REPO_ROOT).as_posix()
    assert _check_ignore(rel) == 1, f"{rel} is gitignored — relay broken"


def test_cache_base_is_inside_the_repo():
    # The relay only works if the app reads snapshots from the repo checkout.
    assert REPO_ROOT in cache._BASE.parents
