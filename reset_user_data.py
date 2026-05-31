#!/usr/bin/env python3
"""
Reset sim-search user data from Python.

Default behavior:
- delete all User rows (which cascades related matrices/feedback)
- delete all Django sessions
- remove and recreate the user_matrices directory

Optional:
- wipe the SQLite database file entirely
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def _setup_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "djangoProject.settings")
    os.environ.setdefault("SIMSEARCH_SKIP_APP_AUTOLOAD", "1")
    import django

    django.setup()


def _reset_database_rows() -> tuple[int, int]:
    from django.contrib.sessions.models import Session
    from base.models import User

    user_count = User.objects.count()
    session_count = Session.objects.count()

    # User cascade removes UserMetricMatrix and FeedbackItem rows.
    User.objects.all().delete()
    Session.objects.all().delete()
    return user_count, session_count


def _wipe_sqlite_file() -> Path:
    db_path = Path(os.environ.get("DJANGO_DB_PATH", BASE_DIR / "db.sqlite3")).resolve()
    if db_path.exists():
        db_path.unlink()
    return db_path


def _reset_user_matrices(path: Path, recreate: bool = True) -> tuple[bool, Path]:
    existed = path.exists()
    if existed:
        shutil.rmtree(path)
    if recreate:
        path.mkdir(parents=True, exist_ok=True)
    return existed, path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset user DB rows and user_matrices folder.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Run without interactive confirmation.",
    )
    parser.add_argument(
        "--wipe-db-file",
        action="store_true",
        help="Delete the SQLite database file instead of only deleting User rows.",
    )
    parser.add_argument(
        "--matrices-dir",
        default=str(BASE_DIR / "user_matrices"),
        help="Path to the user_matrices directory.",
    )
    parser.add_argument(
        "--no-recreate-matrices-dir",
        action="store_true",
        help="Do not recreate the user_matrices directory after deletion.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    matrices_dir = Path(args.matrices_dir).resolve()

    print("This will reset sim-search user data:")
    if args.wipe_db_file:
        db_path = Path(os.environ.get("DJANGO_DB_PATH", BASE_DIR / "db.sqlite3")).resolve()
        print(f"  - delete SQLite DB file: {db_path}")
    else:
        print("  - delete all User rows")
        print("  - delete all Django sessions")
    print(f"  - remove user_matrices dir: {matrices_dir}")
    if not args.no_recreate_matrices_dir:
        print("  - recreate empty user_matrices dir")

    if not args.yes:
        answer = input("Type 'yes' to continue: ").strip().lower()
        if answer != "yes":
            print("Cancelled.")
            return 1

    if args.wipe_db_file:
        db_path = _wipe_sqlite_file()
        print(f"Deleted SQLite database file: {db_path}")
    else:
        _setup_django()
        deleted_users, deleted_sessions = _reset_database_rows()
        print(f"Deleted users: {deleted_users}")
        print(f"Deleted sessions: {deleted_sessions}")

    existed, final_matrices_dir = _reset_user_matrices(
        matrices_dir,
        recreate=not args.no_recreate_matrices_dir,
    )
    if existed:
        print(f"Removed directory: {final_matrices_dir}")
    else:
        print(f"Directory did not exist: {final_matrices_dir}")
    if not args.no_recreate_matrices_dir:
        print(f"Recreated empty directory: {final_matrices_dir}")

    print("Reset finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
