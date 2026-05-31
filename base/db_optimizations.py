from __future__ import annotations

import os

from django.db.backends.signals import connection_created
from django.dispatch import receiver


_TRUE_VALUES = {'1', 'true', 'yes', 'on'}


def _parse_bool(value, default):
    if value is None:
        return default
    return str(value).strip().lower() in _TRUE_VALUES


@receiver(connection_created)
def configure_sqlite_runtime(sender, connection, **kwargs):
    if connection.vendor != 'sqlite':
        return

    try:
        busy_timeout_ms = max(1000, int(os.environ.get('DJANGO_SQLITE_BUSY_TIMEOUT_MS', '30000')))
    except (TypeError, ValueError):
        busy_timeout_ms = 30000

    wal_enabled = _parse_bool(os.environ.get('DJANGO_SQLITE_WAL', '1'), True)
    synchronous = str(os.environ.get('DJANGO_SQLITE_SYNCHRONOUS', 'NORMAL') or 'NORMAL').strip().upper()
    temp_store = str(os.environ.get('DJANGO_SQLITE_TEMP_STORE', 'MEMORY') or 'MEMORY').strip().upper()

    try:
        mmap_size = int(os.environ.get('DJANGO_SQLITE_MMAP_SIZE', str(256 * 1024 * 1024)))
    except (TypeError, ValueError):
        mmap_size = 256 * 1024 * 1024

    try:
        cache_size = int(os.environ.get('DJANGO_SQLITE_CACHE_SIZE', str(-64 * 1024)))
    except (TypeError, ValueError):
        cache_size = -64 * 1024

    try:
        with connection.cursor() as cursor:
            cursor.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
            if wal_enabled:
                cursor.execute("PRAGMA journal_mode = WAL")
            if synchronous:
                cursor.execute(f"PRAGMA synchronous = {synchronous}")
            if temp_store:
                cursor.execute(f"PRAGMA temp_store = {temp_store}")
            if mmap_size > 0:
                cursor.execute(f"PRAGMA mmap_size = {mmap_size}")
            if cache_size != 0:
                cursor.execute(f"PRAGMA cache_size = {cache_size}")
    except Exception as exc:
        print(f"[WARN] Could not configure SQLite runtime pragmas: {exc}", flush=True)
