"""
In-process startup status tracker for frontend loading screen.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
import threading


_LOCK = threading.Lock()
_MAX_LOG_LINES = 500

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNTIME_DIR = os.environ.get(
    "SIMSEARCH_RUNTIME_DIR",
    os.path.join(_BASE_DIR, "runtime"),
)
_STATUS_FILE = os.environ.get(
    "SIMSEARCH_STARTUP_STATUS_FILE",
    os.path.join(_RUNTIME_DIR, "startup_status.json"),
)
_LOG_FILE = os.environ.get(
    "SIMSEARCH_STARTUP_LOG_FILE",
    os.path.join(_RUNTIME_DIR, "startup_logs.log"),
)

_DEFAULT_STATE = {
    "phase": "idle",  # idle | starting | ready | error | disabled
    "started_at": None,
    "updated_at": None,
    "error": None,
}
_IN_MEMORY_STATE = dict(_DEFAULT_STATE)
_IN_MEMORY_LOGS = []


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _ensure_runtime_dir() -> None:
    os.makedirs(os.path.dirname(_STATUS_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)


def _read_state_file() -> dict:
    try:
        with open(_STATUS_FILE, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            result = dict(_DEFAULT_STATE)
            result.update(payload)
            return result
    except Exception:
        pass
    return dict(_DEFAULT_STATE)


def _write_state_file(state: dict) -> None:
    _ensure_runtime_dir()
    temp_path = f"{_STATUS_FILE}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, _STATUS_FILE)


def _append_log_line(line: str) -> None:
    _ensure_runtime_dir()
    with open(_LOG_FILE, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def _remember_state(state: dict) -> dict:
    merged = dict(_DEFAULT_STATE)
    merged.update(state or {})
    _IN_MEMORY_STATE.clear()
    _IN_MEMORY_STATE.update(merged)
    return dict(_IN_MEMORY_STATE)


def _remember_log_line(line: str) -> None:
    _IN_MEMORY_LOGS.append(line)
    if len(_IN_MEMORY_LOGS) > _MAX_LOG_LINES:
        del _IN_MEMORY_LOGS[:-_MAX_LOG_LINES]


def _read_recent_logs(max_lines: int = _MAX_LOG_LINES) -> list[str]:
    try:
        with open(_LOG_FILE, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        if len(lines) > max_lines:
            return lines[-max_lines:]
        return lines
    except Exception:
        return []


def log(message: str) -> None:
    text = str(message).rstrip("\n")
    if not text:
        return
    with _LOCK:
        line = f"{_now_iso()} {text}"
        _remember_log_line(line)
        try:
            _append_log_line(line)
        except Exception:
            pass

        state = _remember_state(_read_state_file())
        state["updated_at"] = _now_iso()
        _remember_state(state)
        try:
            _write_state_file(state)
        except Exception:
            pass


def start(message: str | None = None) -> None:
    with _LOCK:
        state = _remember_state(_read_state_file())
        if state.get("started_at") is None:
            state["started_at"] = _now_iso()
        state["phase"] = "starting"
        state["error"] = None
        state["updated_at"] = _now_iso()
        _remember_state(state)
        try:
            _write_state_file(state)
        except Exception:
            pass
    if message:
        log(message)


def mark_ready(message: str | None = None) -> None:
    with _LOCK:
        state = _remember_state(_read_state_file())
        state["phase"] = "ready"
        state["error"] = None
        state["updated_at"] = _now_iso()
        _remember_state(state)
        try:
            _write_state_file(state)
        except Exception:
            pass
    if message:
        log(message)


def mark_error(error_message: str) -> None:
    with _LOCK:
        state = _remember_state(_read_state_file())
        state["phase"] = "error"
        state["error"] = str(error_message)
        state["updated_at"] = _now_iso()
        _remember_state(state)
        try:
            _write_state_file(state)
        except Exception:
            pass
    log(f"[ERROR] {error_message}")


def mark_disabled(message: str | None = None) -> None:
    with _LOCK:
        state = _remember_state(_read_state_file())
        state["phase"] = "disabled"
        state["error"] = None
        state["updated_at"] = _now_iso()
        _remember_state(state)
        try:
            _write_state_file(state)
        except Exception:
            pass
    if message:
        log(message)


def get_snapshot() -> dict:
    with _LOCK:
        state = _remember_state(_read_state_file())
        logs = _read_recent_logs(_MAX_LOG_LINES)
        if not logs:
            logs = list(_IN_MEMORY_LOGS)
        return {
            "phase": state.get("phase", "idle"),
            "started_at": state.get("started_at"),
            "updated_at": state.get("updated_at"),
            "error": state.get("error"),
            "logs": logs,
        }
