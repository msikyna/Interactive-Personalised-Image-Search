import os
import threading
import time
import urllib.request

try:
    import fcntl
except Exception:  # pragma: no cover - non-Unix fallback
    fcntl = None

from . import config


_START_LOCK = threading.Lock()
_STARTED_IN_PROCESS = False
_LEADER_LOCK_FD = None
_KEEPALIVE_THREAD = None


def _runtime_dir():
    return os.path.join(getattr(config, 'BASE_DIR', os.getcwd()), 'runtime')


def _lock_path():
    return os.path.join(_runtime_dir(), 'keepalive.lock')


def _acquire_leader_lock():
    global _LEADER_LOCK_FD

    if _LEADER_LOCK_FD is not None:
        return True

    if fcntl is None:
        return False

    os.makedirs(_runtime_dir(), exist_ok=True)
    fd = os.open(_lock_path(), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return False

    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode('utf-8'))
    except Exception:
        pass

    _LEADER_LOCK_FD = fd
    return True


def _keepalive_url():
    port = str(os.environ.get('PORT') or os.environ.get('SIMSEARCH_PORT') or '8942').strip()
    path = str(getattr(config, 'KEEPALIVE_URL_PATH', '/api/startup-status/') or '/api/startup-status/').strip()
    if not path.startswith('/'):
        path = '/' + path
    return f"http://127.0.0.1:{port}{path}"


def _default_host_header():
    raw_hosts = str(os.environ.get('DJANGO_ALLOWED_HOSTS', '') or '').strip()
    candidates = [item.strip() for item in raw_hosts.split(',') if item.strip()]
    for host in candidates:
        if host not in {'*', '127.0.0.1', 'localhost'}:
            return host
    for host in candidates:
        if host in {'127.0.0.1', 'localhost'}:
            return host
    return ''


def _perform_keepalive_request():
    url = _keepalive_url()
    headers = {
        'User-Agent': 'sim-search-keepalive/1.0',
    }

    host_header = os.environ.get('SIMSEARCH_KEEPALIVE_HOST_HEADER', '').strip() or _default_host_header()
    if host_header:
        headers['Host'] = host_header

    request = urllib.request.Request(url, headers=headers, method='GET')
    timeout = max(5, int(getattr(config, 'KEEPALIVE_TIMEOUT_SECONDS', 20)))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status_code = int(response.getcode())
        response.read(256)
    return url, status_code


def _keepalive_loop():
    interval = max(30, int(getattr(config, 'KEEPALIVE_INTERVAL_SECONDS', 240)))
    print(f"[keepalive] leader started in pid={os.getpid()} interval={interval}s", flush=True)

    while True:
        started = time.perf_counter()
        try:
            url, status_code = _perform_keepalive_request()
            print(f"[keepalive] GET {url} -> {status_code}", flush=True)
        except Exception as e:
            print(f"[keepalive] probe failed: {e}", flush=True)

        elapsed = time.perf_counter() - started
        sleep_for = max(1.0, float(interval) - elapsed)
        time.sleep(sleep_for)


def ensure_keepalive_started():
    global _STARTED_IN_PROCESS, _KEEPALIVE_THREAD

    if not getattr(config, 'KEEPALIVE_ENABLED', False):
        return False

    with _START_LOCK:
        if _STARTED_IN_PROCESS:
            return _KEEPALIVE_THREAD is not None
        _STARTED_IN_PROCESS = True

        if not _acquire_leader_lock():
            return False

        _KEEPALIVE_THREAD = threading.Thread(
            target=_keepalive_loop,
            name='simsearch-keepalive',
            daemon=True,
        )
        _KEEPALIVE_THREAD.start()
        return True
