"""
Configuration for image similarity search application.
Edit these settings to customize the auto-load behavior.
"""

import os
import signal
import subprocess
import sys
from functools import lru_cache

_TRUE_VALUES = {'1', 'true', 'yes', 'on'}

# Auto-load dataset on Django startup
AUTO_LOAD_DATASET = str(os.environ.get('SIMSEARCH_AUTO_LOAD_DATASET', '1')).strip().lower() in _TRUE_VALUES
# Run auto-load in a background thread so app startup is non-blocking
AUTO_LOAD_BACKGROUND = str(os.environ.get('SIMSEARCH_AUTO_LOAD_BACKGROUND', '1')).strip().lower() in _TRUE_VALUES

# Auto-load CLIP model on Django startup (requires dataset to be loaded first)
AUTO_LOAD_CLIP_MODEL = str(os.environ.get('SIMSEARCH_AUTO_LOAD_CLIP_MODEL', '1')).strip().lower() in _TRUE_VALUES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Runtime logging/performance settings
# Verbose logs can significantly slow down concurrent request throughput.
DEFAULT_VERBOSE_RUNTIME_LOGS = False
# Per-query/feedback disk logging (JSON + matrix snapshots) can add notable I/O latency.
DEFAULT_ENABLE_QUERY_FILE_LOGGING = True
DEFAULT_SAVE_MATRIX_SNAPSHOTS = True
DEFAULT_QUERY_LOG_RESULTS_COUNT = 100
# If enabled, query logging may run an extra search to fill QUERY_LOG_RESULTS_COUNT.
# Disable to favor lower request latency.
DEFAULT_QUERY_LOG_EXPAND_RESULTS = True
DEFAULT_DEBUG_RESPONSE_HEADERS = False
DEFAULT_KEEPALIVE_ENABLED = False
DEFAULT_KEEPALIVE_INTERVAL_SECONDS = 240
DEFAULT_KEEPALIVE_TIMEOUT_SECONDS = 20
DEFAULT_KEEPALIVE_URL_PATH = '/api/startup-status/'
DEFAULT_MATRIX_CACHE_SIZE = 32

# Progressive filter+refine: return a quick partial superset first, then full range in background.
DEFAULT_PROGRESSIVE_FILTER_REFINE = False
DEFAULT_PROGRESSIVE_QUICK_RANGE_RATIO = 1.0
# Automatically reset personalized matrix when switching to a different query.
DEFAULT_RESET_MATRIX_ON_QUERY_CHANGE = False

# Feature files location on server (can be overridden via env vars)
DEFAULT_SERVER_FEATURES_DIR = '/share/datasets/profimedia/features-CLIP/ViT-L14'
DEFAULT_LOCAL_FEATURES_DIR = os.path.join(BASE_DIR, 'data-images')
DEFAULT_MAX_DATASET_FILES = None

# FAISS index settings
DEFAULT_USE_FAISS_INDEX = True
DEFAULT_ENABLE_FAISS_L2_INDEX = True
DEFAULT_ENABLE_FAISS_IP_INDEX = True
DEFAULT_FAISS_L2_INDEX_OPTIONS = {
    'ip_32g_q1': (
        '/home/xsikyna/Indices/AutofaissL2/'
        'Profimedia_AutofaissL2_21000000_current_memory_available-128G_'
        'max_index_memory_usage-32G_max_index_query_time_ms-0.0001_'
        'min_nearest_neighbors_to_retrieve-10.faiss'
    ),
}
DEFAULT_FAISS_IP_INDEX_OPTIONS = {
    'ip_32g_q1': (
        '/home/xsikyna/Indices/AutofaissIP/'
        'Profimedia_AutofaissIP_21000000_current_memory_available-128G_'
        'max_index_memory_usage-32G_max_index_query_time_ms-1_'
        'min_nearest_neighbors_to_retrieve-10.faiss'
    ),
    'ip_8g_q0_1': (
        '/home/xsikyna/Indices/AutofaissIP/'
        'Profimedia_AutofaissIP_21000000_current_memory_available-128G_'
        'max_index_memory_usage-8G_max_index_query_time_ms-0.1_'
        'min_nearest_neighbors_to_retrieve-10.faiss'
    ),
}
DEFAULT_FAISS_L2_INDEX_KEY = 'l2_32g_q1'
DEFAULT_FAISS_IP_INDEX_KEY = 'ip_8g_q0_1'
DEFAULT_FAISS_CANDIDATE_MULTIPLIER = 20
# In FAISS mode, skip loading vectors from .txt and use index reconstruction instead.
DEFAULT_FAISS_SKIP_TXT_LOAD = True
# Build FAISS IVF direct-map during startup to avoid first-query reconstruct latency.
DEFAULT_FAISS_INIT_DIRECT_MAP_ON_LOAD = True
DEFAULT_IMAGE_NAMES_CACHE_PATH = os.path.join(BASE_DIR, 'cache', 'image_names.txt')
DEFAULT_IMAGE_NAMES_FILE_L2 = '/home/xsikyna/sim-search/cache/L2/21000000_ids.npy'
DEFAULT_IMAGE_NAMES_FILE_IP = '/home/xsikyna/sim-search/cache/IP/21000000_ids.npy'
# Progress logging interval (seconds) while scanning image directories.
DEFAULT_IMAGE_SCAN_PROGRESS_SECONDS = 10
# Folders skipped when loading image-name mapping (to align with filtered index build).
DEFAULT_SKIP_MAPPING_FOLDER_RANGES = ''
DEFAULT_IMAGE_X_SENDFILE = False
DEFAULT_IMAGE_X_ACCEL_REDIRECT_PREFIX = ''


def _parse_dataset_files(value):
    """Parse comma-separated dataset file names from env var."""
    if not value:
        return []
    files = [item.strip() for item in value.split(',') if item.strip()]
    return files


def _discover_dataset_files(directory):
    """Discover all .txt feature files in the dataset directory."""
    try:
        return sorted(
            [name for name in os.listdir(directory) if name.lower().endswith('.txt')]
        )
    except OSError:
        return []


def _parse_bool(value, default):
    """Parse boolean env var values."""
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _parse_max_dataset_files(value):
    """Parse max dataset files from env var."""
    if value is None or str(value).strip() == '':
        return DEFAULT_MAX_DATASET_FILES
    try:
        parsed = int(value)
        if parsed <= 0:
            return None
        return parsed
    except (TypeError, ValueError):
        return DEFAULT_MAX_DATASET_FILES


def _parse_positive_int(value, default):
    """Parse positive integer from env var."""
    if value is None or str(value).strip() == '':
        return default
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _parse_ratio(value, default):
    """Parse ratio in [0.0, 1.0] from env var."""
    if value is None or str(value).strip() == '':
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, parsed))


def _parse_folder_ranges(value):
    """
    Parse comma-separated folder ids/ranges into a set of folder names.
    Example: "331-339,517-520,600" -> {"331", ... "339", "517", ... "520", "600"}
    """
    result = set()
    if value is None:
        return result

    tokens = [token.strip() for token in str(value).split(',') if token.strip()]
    for token in tokens:
        if '-' in token:
            left, right = token.split('-', 1)
            left = left.strip()
            right = right.strip()
            if left.isdigit() and right.isdigit():
                start = int(left)
                end = int(right)
                if start <= end:
                    for number in range(start, end + 1):
                        result.add(str(number))
                    continue
        result.add(token)
    return result


def _resolve_index_path(direct_env_name, key_env_name, default_key, options):
    """
    Resolve FAISS index path with priority:
    1) direct env path (SIMSEARCH_FAISS_*_INDEX)
    2) selected key (SIMSEARCH_FAISS_*_INDEX_KEY)
    3) default key
    """
    direct_path = os.environ.get(direct_env_name, '').strip()
    if direct_path:
        return direct_path

    selected_key = os.environ.get(key_env_name, default_key).strip()
    if selected_key in options:
        return options[selected_key]

    if default_key in options:
        return options[default_key]

    # Last-resort: any configured option
    for _, path in options.items():
        return path
    return ''


DATASET_DIRECTORY = os.environ.get(
    'SIMSEARCH_FEATURES_DIR',
    DEFAULT_SERVER_FEATURES_DIR if os.path.isdir(DEFAULT_SERVER_FEATURES_DIR) else DEFAULT_LOCAL_FEATURES_DIR
)
_explicit_files = _parse_dataset_files(os.environ.get('SIMSEARCH_DATASET_FILES'))
MAX_DATASET_FILES = _parse_max_dataset_files(os.environ.get('SIMSEARCH_MAX_DATASET_FILES'))

if _explicit_files:
    DATASET_FILES = _explicit_files[:MAX_DATASET_FILES] if MAX_DATASET_FILES else _explicit_files
else:
    discovered_files = _discover_dataset_files(DATASET_DIRECTORY)
    DATASET_FILES = discovered_files[:MAX_DATASET_FILES] if MAX_DATASET_FILES else discovered_files

# Allowed image folders derived from selected txt files (e.g. "561.txt" -> "561")
ALLOWED_IMAGE_FOLDERS = [os.path.splitext(os.path.basename(name))[0] for name in DATASET_FILES]
ALLOWED_IMAGE_FOLDERS_SET = set(ALLOWED_IMAGE_FOLDERS)

# Search backend settings
USE_FAISS_INDEX = _parse_bool(os.environ.get('SIMSEARCH_USE_FAISS_INDEX'), DEFAULT_USE_FAISS_INDEX)
FAISS_ENABLE_L2_INDEX = _parse_bool(
    os.environ.get('SIMSEARCH_ENABLE_FAISS_L2_INDEX'),
    DEFAULT_ENABLE_FAISS_L2_INDEX
)
FAISS_ENABLE_IP_INDEX = _parse_bool(
    os.environ.get('SIMSEARCH_ENABLE_FAISS_IP_INDEX'),
    DEFAULT_ENABLE_FAISS_IP_INDEX
)
FAISS_L2_INDEX_KEY = os.environ.get(
    'SIMSEARCH_FAISS_L2_INDEX_KEY',
    DEFAULT_FAISS_L2_INDEX_KEY
).strip()
FAISS_IP_INDEX_KEY = os.environ.get(
    'SIMSEARCH_FAISS_IP_INDEX_KEY',
    DEFAULT_FAISS_IP_INDEX_KEY
).strip()
FAISS_L2_INDEX_PATH = _resolve_index_path(
    direct_env_name='SIMSEARCH_FAISS_L2_INDEX',
    key_env_name='SIMSEARCH_FAISS_L2_INDEX_KEY',
    default_key=DEFAULT_FAISS_L2_INDEX_KEY,
    options=DEFAULT_FAISS_L2_INDEX_OPTIONS,
)
FAISS_IP_INDEX_PATH = _resolve_index_path(
    direct_env_name='SIMSEARCH_FAISS_IP_INDEX',
    key_env_name='SIMSEARCH_FAISS_IP_INDEX_KEY',
    default_key=DEFAULT_FAISS_IP_INDEX_KEY,
    options=DEFAULT_FAISS_IP_INDEX_OPTIONS,
)
FAISS_CANDIDATE_MULTIPLIER = _parse_positive_int(
    os.environ.get('SIMSEARCH_FAISS_CANDIDATE_MULTIPLIER'),
    DEFAULT_FAISS_CANDIDATE_MULTIPLIER
)
FAISS_SKIP_TXT_LOAD = _parse_bool(
    os.environ.get('SIMSEARCH_FAISS_SKIP_TXT_LOAD'),
    DEFAULT_FAISS_SKIP_TXT_LOAD
)
FAISS_INIT_DIRECT_MAP_ON_LOAD = _parse_bool(
    os.environ.get('SIMSEARCH_FAISS_INIT_DIRECT_MAP_ON_LOAD'),
    DEFAULT_FAISS_INIT_DIRECT_MAP_ON_LOAD
)
IMAGE_NAMES_FILE = os.environ.get('SIMSEARCH_IMAGE_NAMES_FILE', '').strip()
IMAGE_NAMES_FILE_L2 = os.environ.get(
    'SIMSEARCH_IMAGE_NAMES_FILE_L2',
    DEFAULT_IMAGE_NAMES_FILE_L2
).strip()
IMAGE_NAMES_FILE_IP = os.environ.get(
    'SIMSEARCH_IMAGE_NAMES_FILE_IP',
    DEFAULT_IMAGE_NAMES_FILE_IP
).strip()
DISA_BASE_URL = os.environ.get('DISA_BASE_URL', '').strip()
ALTERNATIVE_IMAGES_BASE_URL = os.environ.get('ALTERNATIVE_IMAGES_BASE_URL', '').strip()
USE_DISA_PROFIMEDIA = _parse_bool(
    os.environ.get('USE_DISA_PROFIMEDIA') or None,
    True
)
IMAGE_X_SENDFILE = _parse_bool(
    os.environ.get('SIMSEARCH_IMAGE_X_SENDFILE'),
    DEFAULT_IMAGE_X_SENDFILE
)
IMAGE_X_ACCEL_REDIRECT_PREFIX = os.environ.get(
    'SIMSEARCH_IMAGE_X_ACCEL_REDIRECT_PREFIX',
    DEFAULT_IMAGE_X_ACCEL_REDIRECT_PREFIX
).strip()
IMAGE_NAMES_CACHE_PATH = os.environ.get(
    'SIMSEARCH_IMAGE_NAMES_CACHE',
    DEFAULT_IMAGE_NAMES_CACHE_PATH
).strip()
IMAGE_SCAN_PROGRESS_SECONDS = _parse_positive_int(
    os.environ.get('SIMSEARCH_IMAGE_SCAN_PROGRESS_SECONDS'),
    DEFAULT_IMAGE_SCAN_PROGRESS_SECONDS
)
MAPPING_SKIP_FOLDERS_SET = _parse_folder_ranges(
    os.environ.get('SIMSEARCH_SKIP_MAPPING_FOLDERS', DEFAULT_SKIP_MAPPING_FOLDER_RANGES)
)

# Runtime/performance toggles
VERBOSE_RUNTIME_LOGS = _parse_bool(
    os.environ.get('SIMSEARCH_VERBOSE_RUNTIME_LOGS'),
    DEFAULT_VERBOSE_RUNTIME_LOGS
)
ENABLE_QUERY_FILE_LOGGING = _parse_bool(
    os.environ.get('SIMSEARCH_ENABLE_QUERY_FILE_LOGGING'),
    DEFAULT_ENABLE_QUERY_FILE_LOGGING
)
SAVE_MATRIX_SNAPSHOTS = _parse_bool(
    os.environ.get('SIMSEARCH_SAVE_MATRIX_SNAPSHOTS'),
    DEFAULT_SAVE_MATRIX_SNAPSHOTS
)
PROGRESSIVE_FILTER_REFINE = _parse_bool(
    os.environ.get('SIMSEARCH_PROGRESSIVE_FILTER_REFINE'),
    DEFAULT_PROGRESSIVE_FILTER_REFINE
)
PROGRESSIVE_QUICK_RANGE_RATIO = _parse_ratio(
    os.environ.get('SIMSEARCH_PROGRESSIVE_QUICK_RANGE_RATIO'),
    DEFAULT_PROGRESSIVE_QUICK_RANGE_RATIO
)
RESET_MATRIX_ON_QUERY_CHANGE = _parse_bool(
    os.environ.get('SIMSEARCH_RESET_MATRIX_ON_QUERY_CHANGE'),
    DEFAULT_RESET_MATRIX_ON_QUERY_CHANGE
)
QUERY_LOG_RESULTS_COUNT = _parse_positive_int(
    os.environ.get('SIMSEARCH_QUERY_LOG_RESULTS_COUNT'),
    DEFAULT_QUERY_LOG_RESULTS_COUNT
)
QUERY_LOG_EXPAND_RESULTS = _parse_bool(
    os.environ.get('SIMSEARCH_QUERY_LOG_EXPAND_RESULTS'),
    DEFAULT_QUERY_LOG_EXPAND_RESULTS
)
DEBUG_RESPONSE_HEADERS = _parse_bool(
    os.environ.get('SIMSEARCH_DEBUG_RESPONSE_HEADERS'),
    DEFAULT_DEBUG_RESPONSE_HEADERS
)
KEEPALIVE_ENABLED = _parse_bool(
    os.environ.get('SIMSEARCH_KEEPALIVE_ENABLED'),
    DEFAULT_KEEPALIVE_ENABLED
)
KEEPALIVE_INTERVAL_SECONDS = _parse_positive_int(
    os.environ.get('SIMSEARCH_KEEPALIVE_INTERVAL_SECONDS'),
    DEFAULT_KEEPALIVE_INTERVAL_SECONDS
)
KEEPALIVE_TIMEOUT_SECONDS = _parse_positive_int(
    os.environ.get('SIMSEARCH_KEEPALIVE_TIMEOUT_SECONDS'),
    DEFAULT_KEEPALIVE_TIMEOUT_SECONDS
)
KEEPALIVE_URL_PATH = os.environ.get(
    'SIMSEARCH_KEEPALIVE_URL_PATH',
    DEFAULT_KEEPALIVE_URL_PATH
).strip() or DEFAULT_KEEPALIVE_URL_PATH
MATRIX_CACHE_SIZE = _parse_positive_int(
    os.environ.get('SIMSEARCH_MATRIX_CACHE_SIZE'),
    DEFAULT_MATRIX_CACHE_SIZE
)


def _format_native_import_return_code(return_code: int) -> str:
    if return_code < 0:
        signal_number = abs(return_code)
        try:
            signal_name = signal.Signals(signal_number).name
        except Exception:
            signal_name = f'signal {signal_number}'
        return f'{return_code} ({signal_name})'
    if return_code == 128 + signal.SIGILL:
        return f'{return_code} (SIGILL)'
    return str(return_code)


@lru_cache(maxsize=None)
def probe_native_import(module_name: str, label: str | None = None) -> None:
    """
    Import native-heavy modules in a child process first. If a wheel was built
    for unsupported CPU instructions, the child can die with SIGILL while the
    Gunicorn process still gets a readable error.
    """
    if not _parse_bool(os.environ.get('SIMSEARCH_NATIVE_IMPORT_PROBE'), True):
        return

    display_name = label or module_name
    code = "import importlib; importlib.import_module(%r)" % module_name
    try:
        result = subprocess.run(
            [sys.executable, '-c', code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out while probing {display_name} import.") from exc

    stderr = (result.stderr or '').strip()
    illegal_instruction = 'illegal instruction' in stderr.lower()
    if result.returncode == 0 and not illegal_instruction:
        return

    return_code = 128 + signal.SIGILL if illegal_instruction and result.returncode == 0 else result.returncode
    detail = f" stderr: {stderr}" if stderr else ''
    raise RuntimeError(
        f"{display_name} failed an isolated import probe with exit code "
        f"{_format_native_import_return_code(return_code)}.{detail} "
        "If this mentions SIGILL or exit code 132, the installed wheel likely uses CPU "
        "instructions that this host does not support."
    )
