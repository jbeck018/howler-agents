"""Repo-scoped path management for per-repository howler-agents storage."""

from __future__ import annotations

from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

_HOWLER_DIR_NAME = ".howler-agents"
_DB_FILE_NAME = "evolution.db"
_GITIGNORE_ENTRIES = ("*.db", "*.db-wal", "*.db-shm")


def find_repo_root(start_path: Path | None = None) -> Path:
    """Walk up from start_path looking for a .git directory.

    Returns the directory containing .git, or cwd if no git repo is found.
    """
    current = (start_path or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            log.debug("repo_root_found", path=str(candidate))
            return candidate
    log.debug("repo_root_not_found_using_cwd", cwd=str(current))
    return current


def get_howler_dir(repo_root: Path | None = None) -> Path:
    """Return the .howler-agents directory, creating it if missing."""
    root = repo_root if repo_root is not None else find_repo_root()
    howler_dir = root / _HOWLER_DIR_NAME
    howler_dir.mkdir(exist_ok=True)
    log.debug("howler_dir_resolved", path=str(howler_dir))
    return howler_dir


def get_db_path(repo_root: Path | None = None) -> Path:
    """Return the path to the SQLite database file."""
    return get_howler_dir(repo_root) / _DB_FILE_NAME


def ensure_gitignore(howler_dir: Path) -> None:
    """Create or update .howler-agents/.gitignore to exclude DB files."""
    gitignore_path = howler_dir / ".gitignore"
    existing_lines: list[str] = []

    if gitignore_path.exists():
        existing_lines = gitignore_path.read_text(encoding="utf-8").splitlines()

    missing = [entry for entry in _GITIGNORE_ENTRIES if entry not in existing_lines]
    if not missing:
        return

    lines = existing_lines + missing
    gitignore_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("gitignore_updated", path=str(gitignore_path), added=missing)
