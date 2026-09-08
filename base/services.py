"""
Business logic for image similarity search.
Keep views thin by moving complex logic here.
"""

import glob
import json
import os
import random
import sys
import time
from datetime import datetime
import numpy as np

from . import config
try:
    from . import startup_status  # type: ignore
except Exception:
    startup_status = None

faiss = None
_FAISS_IMPORT_ATTEMPTED = False
_FAISS_IMPORT_ERROR = None

# Add implementations folder to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'implementations'))

from implementations.dataset_loader import load_dataset
from implementations.utils import (
    vectorized_euclidean_distances,
    vectorized_mahalanobis_distances,
    vectorized_cosine_distances_normalized,
    euclidean_range_search,
    dot_product_range_search,
    vectorized_dot_product_similarity,
    dot_product_to_euclidean_distance,
    euclidean_to_dot_product_threshold,
)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tif', '.tiff'}
_FAISS_RUNTIME_CONFIGURED = False


def _get_faiss():
    global faiss, _FAISS_IMPORT_ATTEMPTED, _FAISS_IMPORT_ERROR

    if faiss is not None:
        return faiss
    if _FAISS_IMPORT_ATTEMPTED:
        if _FAISS_IMPORT_ERROR is not None:
            raise _FAISS_IMPORT_ERROR
        return faiss

    _FAISS_IMPORT_ATTEMPTED = True
    try:
        config.probe_native_import('faiss', 'FAISS')
        import faiss as faiss_module
        faiss = faiss_module
        return faiss
    except ImportError as exc:
        _FAISS_IMPORT_ERROR = RuntimeError(
            "SIMSEARCH_USE_FAISS_INDEX is enabled, but faiss is not installed. "
            "Install faiss-cpu/faiss-gpu first."
        )
        raise _FAISS_IMPORT_ERROR from exc
    except RuntimeError as exc:
        _FAISS_IMPORT_ERROR = exc
        raise


def _vprint(*args, **kwargs):
    if getattr(config, 'VERBOSE_RUNTIME_LOGS', False):
        print(*args, **kwargs)


def _startup_log(message):
    text = str(message)
    if startup_status is not None:
        try:
            startup_status.log(text)
        except Exception:
            pass
    print(text, flush=True)


class TxtFeatureStore:
    """Disk-backed access to vectors stored in alternating-line txt files."""

    def __init__(self, directory, files_list):
        self.directory = directory
        self.files_list = files_list
        self.image_names = []
        self.dimension = None
        self._file_paths = []
        self._vector_locations = []  # list[(file_idx, byte_offset)]
        self._open_handles = {}

    @staticmethod
    def _parse_vector_line(line, expected_dim=None):
        vector = np.fromstring(line.strip(), sep=',', dtype=np.float32)
        if expected_dim is not None and vector.size != expected_dim:
            raise ValueError(
                f"Vector dimension mismatch. Expected {expected_dim}, got {vector.size}."
            )
        return vector

    def load_metadata(self):
        """
        Parse txt files and store:
        - image_names in dataset order
        - vector byte offsets for lazy random access
        """
        self.image_names = []
        self._file_paths = []
        self._vector_locations = []
        self.dimension = None

        for filename in self.files_list:
            if not filename.endswith('.txt'):
                continue

            path = os.path.join(self.directory, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Feature file not found: {path}")

            file_idx = len(self._file_paths)
            self._file_paths.append(path)
            image_folder = os.path.splitext(filename)[0]

            with open(path, 'r', encoding='utf-8', errors='strict') as handle:
                line_number = 1
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break

                    stripped = line.strip()
                    if line_number % 2 != 0:
                        image_name = stripped.lstrip('#')
                        self.image_names.append(f"{image_folder}/{image_name}")
                    else:
                        self._vector_locations.append((file_idx, offset))
                        if self.dimension is None:
                            self.dimension = self._parse_vector_line(stripped).size

                    line_number += 1

        if self.dimension is None:
            raise ValueError("No vectors found in feature files.")

        if len(self.image_names) != len(self._vector_locations):
            raise ValueError(
                f"Parsed image/vector count mismatch: {len(self.image_names)} names vs "
                f"{len(self._vector_locations)} vectors."
            )

        return self.image_names, self.dimension

    def __len__(self):
        return len(self._vector_locations)

    def _get_handle(self, file_idx):
        handle = self._open_handles.get(file_idx)
        if handle is None or handle.closed:
            path = self._file_paths[file_idx]
            handle = open(path, 'r', encoding='utf-8', errors='strict')
            self._open_handles[file_idx] = handle
        return handle

    def close(self):
        """Close all open file handles."""
        for handle in self._open_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._open_handles = {}

    def get_vector_by_index(self, index):
        """Load a single vector by global dataset index."""
        idx = int(index)
        if idx < 0 or idx >= len(self._vector_locations):
            raise IndexError(f"Index {idx} out of bounds for dataset size {len(self._vector_locations)}")

        file_idx, offset = self._vector_locations[idx]
        handle = self._get_handle(file_idx)
        handle.seek(offset)
        line = handle.readline()
        return self._parse_vector_line(line, self.dimension)

    def get_vectors_by_indices(self, indices):
        """Load multiple vectors by global indices into a dense float32 matrix."""
        idx_array = np.asarray(indices)
        if idx_array.ndim == 0:
            return self.get_vector_by_index(int(idx_array))

        if idx_array.dtype == bool:
            idx_array = np.flatnonzero(idx_array)
        idx_array = idx_array.astype(np.int64, copy=False)

        vectors = np.empty((len(idx_array), self.dimension), dtype=np.float32)
        grouped = {}

        for out_pos, idx in enumerate(idx_array):
            if idx < 0 or idx >= len(self._vector_locations):
                raise IndexError(f"Index {idx} out of bounds for dataset size {len(self._vector_locations)}")
            file_idx, offset = self._vector_locations[int(idx)]
            grouped.setdefault(file_idx, []).append((out_pos, offset))

        for file_idx, positions in grouped.items():
            handle = self._get_handle(file_idx)
            for out_pos, offset in positions:
                handle.seek(offset)
                line = handle.readline()
                vectors[out_pos] = self._parse_vector_line(line, self.dimension)

        return vectors


class LazyDatasetView:
    """Array-like wrapper backed by TxtFeatureStore for on-demand vector access."""

    def __init__(self, store):
        self._store = store
        self.shape = (len(store), store.dimension)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._store.get_vector_by_index(int(key))

        if isinstance(key, slice):
            indices = np.arange(*key.indices(self.shape[0]), dtype=np.int64)
            return self._store.get_vectors_by_indices(indices)

        idx_array = np.asarray(key)
        if idx_array.dtype == bool:
            idx_array = np.flatnonzero(idx_array)
        return self._store.get_vectors_by_indices(idx_array.astype(np.int64, copy=False))


class FaissDatasetView:
    """Array-like wrapper that reconstructs vectors directly from a FAISS index."""

    def __init__(self, service):
        self._service = service
        self.shape = (len(service.image_names or []), int(service.vector_dim))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            vectors = self._service._reconstruct_vectors_by_indices([int(key)])
            return vectors[0]

        if isinstance(key, slice):
            indices = np.arange(*key.indices(self.shape[0]), dtype=np.int64)
            return self._service._reconstruct_vectors_by_indices(indices)

        idx_array = np.asarray(key)
        if idx_array.dtype == bool:
            idx_array = np.flatnonzero(idx_array)
        return self._service._reconstruct_vectors_by_indices(idx_array.astype(np.int64, copy=False))


class ImageSimilarityService:
    """Handle image similarity search operations."""

    def __init__(self):
        self._configure_faiss_runtime()
        self.image_names = None
        self.image_names_l2 = None
        self.image_names_ip = None
        self.dataset = None
        self.dataset_loaded = False
        self.feature_store = None

        self.faiss_enabled = False
        self.faiss_l2_index = None
        self.faiss_ip_index = None

        self.vector_dim = 768
        self.faiss_candidate_multiplier = max(1, int(getattr(config, 'FAISS_CANDIDATE_MULTIPLIER', 20)))
        self._faiss_direct_map_initialized = set()

    @staticmethod
    def _configure_faiss_runtime():
        global _FAISS_RUNTIME_CONFIGURED
        if _FAISS_RUNTIME_CONFIGURED:
            return

        try:
            faiss_module = _get_faiss()
        except RuntimeError as exc:
            print(f"[WARN] FAISS unavailable: {exc}", flush=True)
            _FAISS_RUNTIME_CONFIGURED = True
            return

        raw_threads = os.environ.get('FAISS_NUM_THREADS', '').strip()
        if not raw_threads:
            _FAISS_RUNTIME_CONFIGURED = True
            return

        try:
            threads = max(1, int(raw_threads))
            faiss_module.omp_set_num_threads(threads)
            print(f"[INFO] FAISS threads configured: {threads}", flush=True)
        except Exception as exc:
            print(f"[WARN] Could not configure FAISS threads: {exc}", flush=True)
        finally:
            _FAISS_RUNTIME_CONFIGURED = True

    @staticmethod
    def _natural_folder_sort_key(name):
        if name.isdigit():
            return (0, int(name))
        return (1, name)

    @staticmethod
    def _resolve_image_base_path():
        """Resolve image base path without importing Django settings at module import time."""
        env_path = os.environ.get('SIMSEARCH_IMAGE_BASE_PATH', '').strip()
        if env_path:
            return env_path

        try:
            from django.conf import settings as django_settings
            configured = getattr(django_settings, 'IMAGE_BASE_PATH', '')
            if configured:
                return configured
        except Exception:
            pass

        return '/share/datasets/profimedia/data-images'

    def _load_image_names_from_file(self, file_path, apply_skip_folders=False):
        path = os.path.abspath(file_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image names file not found: {path}")

        _startup_log(f"[INFO] Loading image names from file: {path}")
        started = time.perf_counter()
        if path.lower().endswith('.npy'):
            values = np.load(path, allow_pickle=True)
            image_names = [str(item) for item in values.tolist()]
        else:
            image_names = []
            with open(path, 'r', encoding='utf-8', errors='strict') as handle:
                for raw in handle:
                    name = raw.strip()
                    if name:
                        image_names.append(name)

        skip_folders = set(getattr(config, 'MAPPING_SKIP_FOLDERS_SET', set()) or set())
        if apply_skip_folders and skip_folders:
            before_count = len(image_names)
            image_names = [
                name for name in image_names
                if name.split('/', 1)[0] not in skip_folders
            ]
            removed = before_count - len(image_names)
            if removed > 0:
                print(
                    f"[INFO] Skipped {removed} mapping entries from excluded folders "
                    f"({len(skip_folders)} folder ids configured).",
                    flush=True
                )

        elapsed = time.perf_counter() - started
        _startup_log(
            f"[INFO] Loaded image names mapping: {len(image_names)} entries in {elapsed:.1f}s"
        )
        return image_names

    @staticmethod
    def _atomic_write_text_lines(path, values):
        temp_path = f"{path}.tmp.{os.getpid()}"
        with open(temp_path, 'w', encoding='utf-8') as handle:
            for value in values:
                handle.write(value)
                handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)

    @staticmethod
    def _atomic_write_json(path, payload):
        temp_path = f"{path}.tmp.{os.getpid()}"
        with open(temp_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)

    def _save_image_names_cache(self, image_names, cache_path=None):
        cache_path = cache_path or getattr(config, 'IMAGE_NAMES_CACHE_PATH', '')
        if not cache_path:
            return

        try:
            started = time.perf_counter()
            cache_path = os.path.abspath(cache_path)
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            self._atomic_write_text_lines(cache_path, image_names)

            metadata = {
                'created_at_utc': datetime.utcnow().isoformat() + 'Z',
                'entries': int(len(image_names)),
                'image_base_path': os.path.abspath(self._resolve_image_base_path()),
                'mapping_file': cache_path,
            }
            self._atomic_write_json(f"{cache_path}.meta.json", metadata)

            elapsed = time.perf_counter() - started
            print(
                f"[INFO] Saved image names cache: {cache_path} "
                f"({len(image_names)} entries in {elapsed:.1f}s)",
                flush=True
            )
        except Exception as e:
            print(f"[WARN] Could not write image names cache {cache_path}: {e}")

    def _scan_image_names_from_directory(self):
        image_base_path = os.path.abspath(self._resolve_image_base_path())
        if not os.path.isdir(image_base_path):
            raise FileNotFoundError(f"Image base path not found: {image_base_path}")

        folder_names = list(getattr(config, 'ALLOWED_IMAGE_FOLDERS', []) or [])
        if folder_names:
            folder_names = sorted(set(folder_names), key=self._natural_folder_sort_key)
        else:
            folder_names = sorted(
                [name for name in os.listdir(image_base_path)
                 if os.path.isdir(os.path.join(image_base_path, name))],
                key=self._natural_folder_sort_key
            )

        total_folders = len(folder_names)
        report_interval_sec = max(1, int(getattr(config, 'IMAGE_SCAN_PROGRESS_SECONDS', 10)))
        _startup_log(
            f"[INFO] Scanning image folders under {image_base_path}: {total_folders} folder(s)."
        )

        image_names = []
        total_images = 0
        started = time.perf_counter()
        last_report = started

        for folder_idx, folder in enumerate(folder_names, start=1):
            folder_path = os.path.join(image_base_path, folder)
            if not os.path.isdir(folder_path):
                continue

            files = sorted(
                name for name in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, name))
                and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
            )
            total_images += len(files)
            image_names.extend(f"{folder}/{name}" for name in files)

            now = time.perf_counter()
            should_report = (now - last_report) >= report_interval_sec or folder_idx == total_folders
            if should_report:
                elapsed = now - started
                folders_pct = (100.0 * folder_idx / total_folders) if total_folders else 100.0
                speed = (total_images / elapsed) if elapsed > 0.0 else 0.0
                _startup_log(
                    "[INFO] Image mapping progress: "
                    f"folders {folder_idx}/{total_folders} ({folders_pct:.1f}%), "
                    f"images {total_images}, elapsed {elapsed:.1f}s, "
                    f"rate {speed:.0f} img/s"
                )
                last_report = now

        elapsed_total = time.perf_counter() - started
        _startup_log(
            f"[INFO] Image mapping complete: {total_images} images from {total_folders} "
            f"folder(s) in {elapsed_total:.1f}s"
        )
        return image_names

    @staticmethod
    def _pick_best_ids_mapping_file():
        """
        Auto-discover prebuilt FAISS-id mapping files like:
        cache/200000_ids.npy or cache/20000000_ids.npy
        """
        candidates = []
        search_dirs = []

        cache_path = (getattr(config, 'IMAGE_NAMES_CACHE_PATH', '') or '').strip()
        if cache_path:
            search_dirs.append(os.path.dirname(os.path.abspath(cache_path)))

        base_dir = getattr(config, 'BASE_DIR', None)
        if base_dir:
            search_dirs.append(os.path.join(base_dir, 'cache'))

        # Keep order, drop duplicates
        seen = set()
        dedup_dirs = []
        for path in search_dirs:
            if path and path not in seen:
                dedup_dirs.append(path)
                seen.add(path)

        for directory in dedup_dirs:
            if not os.path.isdir(directory):
                continue
            candidates.extend(sorted(glob.glob(os.path.join(directory, '*_ids.npy'))))

        if not candidates:
            return None

        # Prefer newest file if multiple are present.
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    @staticmethod
    def _mapping_matches_expected_count(image_names, expected_count):
        if expected_count is None:
            return True
        return int(len(image_names)) == int(expected_count)

    def _mapping_candidates_for_index(self, label, expected_count=None):
        label_upper = (label or '').upper()
        specific = (
            getattr(config, 'IMAGE_NAMES_FILE_L2', '')
            if label_upper == 'L2'
            else getattr(config, 'IMAGE_NAMES_FILE_IP', '')
        )
        generic = getattr(config, 'IMAGE_NAMES_FILE', '')
        cache_path = getattr(config, 'IMAGE_NAMES_CACHE_PATH', '')
        auto_path = self._pick_best_ids_mapping_file()

        candidates = []
        for path in (specific, generic, cache_path, auto_path):
            if not path:
                continue
            abs_path = os.path.abspath(path)
            if os.path.isfile(abs_path):
                candidates.append(abs_path)

        # dedupe while preserving order
        unique = []
        seen = set()
        for path in candidates:
            if path in seen:
                continue
            unique.append(path)
            seen.add(path)
        return unique

    def _load_mapping_for_index(self, label, index_obj):
        if index_obj is None:
            return None

        expected_count = self._index_ntotal(index_obj)
        for path in self._mapping_candidates_for_index(label, expected_count=expected_count):
            try:
                names = self._load_image_names_from_file(path, apply_skip_folders=False)
            except Exception as e:
                _startup_log(f"[WARN] Failed loading {label} mapping from {path}: {e}")
                continue

            if expected_count is not None and len(names) != expected_count:
                _startup_log(
                    f"[WARN] {label} mapping count mismatch in {path}: "
                    f"{len(names)} (expected {expected_count})."
                )
                continue

            _startup_log(f"[INFO] Using {label} mapping file: {path}")
            return names

        return None

    def _load_image_names_index_only(self, expected_count=None):
        explicit_file = getattr(config, 'IMAGE_NAMES_FILE', '')
        if explicit_file:
            explicit_names = self._load_image_names_from_file(explicit_file, apply_skip_folders=False)
            if not self._mapping_matches_expected_count(explicit_names, expected_count):
                print(
                    f"[WARN] Explicit mapping count mismatch: {len(explicit_names)} "
                    f"(expected {expected_count}). Keeping explicit mapping and "
                    "skipping directory rebuild.",
                    flush=True
                )
            return explicit_names

        cache_path = getattr(config, 'IMAGE_NAMES_CACHE_PATH', '')
        if cache_path and os.path.isfile(cache_path):
            try:
                cached_names = self._load_image_names_from_file(cache_path, apply_skip_folders=False)
                if self._mapping_matches_expected_count(cached_names, expected_count):
                    return cached_names
                print(
                    f"[WARN] Cached mapping count mismatch: {len(cached_names)} "
                    f"(expected {expected_count}).",
                    flush=True
                )
            except Exception as e:
                print(f"[WARN] Failed to load cached image mapping {cache_path}: {e}")
                print("[INFO] Rebuilding mapping from image directory.", flush=True)

        # Prefer prebuilt ids mapping from index creation when available.
        auto_ids_path = self._pick_best_ids_mapping_file()
        if auto_ids_path:
            try:
                auto_names = self._load_image_names_from_file(auto_ids_path, apply_skip_folders=False)
                if self._mapping_matches_expected_count(auto_names, expected_count):
                    print(f"[INFO] Using auto-discovered ids mapping: {auto_ids_path}", flush=True)
                    return auto_names
                print(
                    f"[WARN] Auto-discovered ids mapping count mismatch: {len(auto_names)} "
                    f"(expected {expected_count}) from {auto_ids_path}.",
                    flush=True
                )
            except Exception as e:
                print(f"[WARN] Failed to load auto-discovered ids mapping {auto_ids_path}: {e}", flush=True)

        _startup_log("[INFO] Building image name mapping from image directory (no txt parsing).")
        image_names = self._scan_image_names_from_directory()
        if not image_names:
            raise RuntimeError(
                "No images found when building image-name mapping. "
                "Set SIMSEARCH_IMAGE_NAMES_FILE or verify SIMSEARCH_IMAGE_BASE_PATH."
            )
        self._save_image_names_cache(image_names)
        return image_names

    def _align_faiss_indices_to_mapping(self, mapping_size):
        """
        Keep only FAISS indexes whose ntotal matches mapping size.
        This avoids id-shift bugs and prevents expensive mapping rebuild fallbacks.
        """
        kept_any = False
        for label, attr in (('L2', 'faiss_l2_index'), ('IP', 'faiss_ip_index')):
            index_obj = getattr(self, attr)
            if index_obj is None:
                continue

            ntotal = self._index_ntotal(index_obj)
            if ntotal is None or ntotal == int(mapping_size):
                kept_any = True
                continue

            print(
                f"[WARN] Disabling {label} FAISS index due to mapping mismatch: "
                f"index ntotal={ntotal}, mapping={mapping_size}.",
                flush=True
            )
            setattr(self, attr, None)

        self.faiss_enabled = self.faiss_l2_index is not None or self.faiss_ip_index is not None
        if not self.faiss_enabled:
            raise RuntimeError(
                "No FAISS index matches the provided image mapping size. "
                "Provide a mapping file aligned to one configured index."
            )

        if kept_any:
            self._update_vector_dim_from_index()

    def _get_primary_faiss_index(self):
        return self.faiss_l2_index or self.faiss_ip_index

    def _mapping_for_metric(self, distance_metric='euclidean', distance_mode=None):
        metric = (distance_metric or '').lower()
        mode = (distance_mode or '').lower()

        if mode == 'dot_product' or metric in ('cosine', 'dot_product'):
            return self.image_names_ip or self.image_names_l2 or self.image_names or []

        return self.image_names_l2 or self.image_names_ip or self.image_names or []

    def _mapping_size_for_metric(self, distance_metric='euclidean', distance_mode=None):
        return len(self._mapping_for_metric(distance_metric=distance_metric, distance_mode=distance_mode))

    def _get_image_name_for_metric(self, index, distance_metric='euclidean', distance_mode=None):
        names = self._mapping_for_metric(distance_metric=distance_metric, distance_mode=distance_mode)
        idx = int(index)
        if idx < 0 or idx >= len(names):
            raise ValueError(f"Index {idx} out of range for selected mapping (size={len(names)}).")
        return names[idx]

    @staticmethod
    def _index_ntotal(index_obj):
        if index_obj is None:
            return None
        try:
            return int(index_obj.ntotal)
        except Exception:
            return None

    def _update_vector_dim_from_index(self):
        index_obj = self._get_primary_faiss_index()
        if index_obj is None:
            return
        try:
            self.vector_dim = int(index_obj.d)
        except Exception:
            pass

    def _get_faiss_ntotal(self):
        index_obj = self._get_primary_faiss_index()
        if index_obj is None:
            return None
        try:
            return int(index_obj.ntotal)
        except Exception:
            return None

    def _ensure_direct_map_for_reconstruct(self, index_obj):
        """
        Some IVF-based indexes (often wrapped in IndexPreTransform) require a direct
        map for reconstruct(id). This can be called during startup prewarm or lazily
        on first reconstruct.
        """
        marker = id(index_obj)
        if marker in self._faiss_direct_map_initialized:
            return True

        ivf_index = None
        faiss_module = _get_faiss()
        try:
            if hasattr(faiss_module, 'extract_index_ivf'):
                ivf_index = faiss_module.extract_index_ivf(index_obj)
        except Exception:
            ivf_index = None

        if ivf_index is None:
            return False

        try:
            _startup_log("[INFO] Initializing FAISS direct map for reconstruct()...")
            # Prefer hashtable direct map type when available.
            if hasattr(ivf_index, 'set_direct_map_type') and hasattr(faiss_module, 'DirectMap'):
                try:
                    ivf_index.set_direct_map_type(faiss_module.DirectMap.Hashtable)
                except Exception:
                    pass

            if hasattr(ivf_index, 'make_direct_map'):
                ivf_index.make_direct_map()
            else:
                return False

            self._faiss_direct_map_initialized.add(marker)
            _startup_log("[INFO] FAISS direct map initialized.")
            return True
        except Exception as e:
            print(f"[WARN] Could not initialize FAISS direct map: {e}", flush=True)
            return False

    def _index_for_metric(self, distance_metric='euclidean'):
        metric = (distance_metric or '').lower()
        if metric in ('cosine', 'dot_product'):
            return self.faiss_ip_index or self.faiss_l2_index
        return self.faiss_l2_index or self.faiss_ip_index

    def _reconstruct_vectors_by_indices(self, indices, distance_metric='euclidean'):
        index_obj = self._index_for_metric(distance_metric=distance_metric)
        if index_obj is None:
            raise RuntimeError("No FAISS index available for vector reconstruction.")

        idx_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx_array.size == 0:
            return np.empty((0, self.vector_dim), dtype=np.float32)

        mapping_size = self._mapping_size_for_metric(distance_metric=distance_metric)
        invalid = np.where((idx_array < 0) | (idx_array >= mapping_size))[0]
        if invalid.size > 0:
            bad_idx = int(idx_array[int(invalid[0])])
            raise IndexError(
                f"Index {bad_idx} out of bounds for dataset size {mapping_size}"
            )

        idx_array = np.ascontiguousarray(idx_array, dtype=np.int64)

        # Fast path: reconstruct in batches via FAISS C++ API (much faster than per-id Python loop).
        if hasattr(index_obj, 'reconstruct_batch'):
            batch_size = 8192
            vectors = np.empty((len(idx_array), self.vector_dim), dtype=np.float32)
            try:
                for start in range(0, len(idx_array), batch_size):
                    end = min(len(idx_array), start + batch_size)
                    chunk_ids = idx_array[start:end]
                    try:
                        chunk = index_obj.reconstruct_batch(chunk_ids)
                    except Exception as e:
                        msg = str(e).lower()
                        if "direct map not initialized" in msg and self._ensure_direct_map_for_reconstruct(index_obj):
                            chunk = index_obj.reconstruct_batch(chunk_ids)
                        else:
                            raise

                    chunk = np.asarray(chunk, dtype=np.float32)
                    if chunk.ndim != 2 or chunk.shape[1] != self.vector_dim or chunk.shape[0] != (end - start):
                        raise RuntimeError(
                            f"Unexpected reconstruct_batch output shape {chunk.shape}, "
                            f"expected ({end - start}, {self.vector_dim})"
                        )
                    vectors[start:end] = chunk
                return vectors
            except Exception as batch_error:
                _vprint(f"[WARN] Falling back to per-id FAISS reconstruct(): {batch_error}")

        # Compatibility fallback for index types/wrappers that do not support reconstruct_batch.
        vectors = np.empty((len(idx_array), self.vector_dim), dtype=np.float32)
        for out_pos, idx in enumerate(idx_array):
            idx_int = int(idx)
            try:
                vectors[out_pos] = index_obj.reconstruct(idx_int)
            except Exception as e:
                msg = str(e).lower()
                # Retry once after building direct map for IVF-based indexes.
                if "direct map not initialized" in msg and self._ensure_direct_map_for_reconstruct(index_obj):
                    try:
                        vectors[out_pos] = index_obj.reconstruct(idx_int)
                        continue
                    except Exception as retry_error:
                        e = retry_error

                raise RuntimeError(
                    f"Failed to reconstruct vector {idx_int} from FAISS index: {e}. "
                    "If your index type does not support reconstruct, disable "
                    "SIMSEARCH_FAISS_SKIP_TXT_LOAD to use txt-backed vectors."
                ) from e
        return vectors

    def _resolve_faiss_index_file(self, index_path):
        """Resolve a configured index path (file or directory) to a readable index file."""
        if not index_path:
            return None

        if os.path.isfile(index_path):
            return index_path

        if not os.path.isdir(index_path):
            return None

        # Prefer explicit FAISS files first.
        candidates = sorted(glob.glob(os.path.join(index_path, '*.faiss')))
        if candidates:
            return candidates[0]

        # Fallback for indexes without .faiss extension.
        fallback = []
        for name in sorted(os.listdir(index_path)):
            full_path = os.path.join(index_path, name)
            if not os.path.isfile(full_path):
                continue
            if name.lower().endswith(('.json', '.txt', '.csv', '.meta', '.md', '.log')):
                continue
            fallback.append(full_path)
        return fallback[0] if fallback else None

    def _load_single_faiss_index(self, label, configured_path):
        resolved = self._resolve_faiss_index_file(configured_path)
        if not resolved:
            print(f"[WARN] {label} FAISS index not found at {configured_path}")
            return None

        _startup_log(f"[INFO] Loading {label} FAISS index from: {resolved}")
        try:
            index = _get_faiss().read_index(resolved)
        except Exception as e:
            raise RuntimeError(f"Failed to load {label} FAISS index from {resolved}: {e}") from e

        try:
            _startup_log(f"[INFO] {label} index ntotal: {index.ntotal}")
        except Exception:
            pass
        return index

    def _load_faiss_indices(self):
        """Load configured FAISS indexes."""
        self.faiss_l2_index = None
        self.faiss_ip_index = None

        if getattr(config, 'FAISS_ENABLE_L2_INDEX', True):
            self.faiss_l2_index = self._load_single_faiss_index('L2', getattr(config, 'FAISS_L2_INDEX_PATH', None))
        else:
            print("[INFO] L2 FAISS index loading disabled by SIMSEARCH_ENABLE_FAISS_L2_INDEX.", flush=True)

        if getattr(config, 'FAISS_ENABLE_IP_INDEX', True):
            self.faiss_ip_index = self._load_single_faiss_index('IP', getattr(config, 'FAISS_IP_INDEX_PATH', None))
        else:
            print("[INFO] IP FAISS index loading disabled by SIMSEARCH_ENABLE_FAISS_IP_INDEX.", flush=True)

        self.faiss_enabled = self.faiss_l2_index is not None or self.faiss_ip_index is not None
        if not self.faiss_enabled:
            raise RuntimeError(
                "FAISS mode is enabled, but no index could be loaded. "
                "Check SIMSEARCH_ENABLE_FAISS_L2_INDEX / SIMSEARCH_ENABLE_FAISS_IP_INDEX "
                "and SIMSEARCH_FAISS_L2_INDEX / SIMSEARCH_FAISS_IP_INDEX paths."
            )

        self._update_vector_dim_from_index()
        expected = len(self.image_names or [])
        for label, index_obj in (('L2', self.faiss_l2_index), ('IP', self.faiss_ip_index)):
            if index_obj is None:
                continue
            try:
                ntotal = int(index_obj.ntotal)
            except Exception:
                ntotal = None
            if expected > 0 and ntotal is not None and ntotal != expected:
                print(
                    f"[WARN] {label} index size mismatch: index ntotal={ntotal}, "
                    f"parsed images={expected}. Results with out-of-range ids will be ignored."
                )

        if getattr(config, 'FAISS_INIT_DIRECT_MAP_ON_LOAD', True):
            for label, index_obj in (('L2', self.faiss_l2_index), ('IP', self.faiss_ip_index)):
                if index_obj is None:
                    continue
                try:
                    initialized = self._ensure_direct_map_for_reconstruct(index_obj)
                    if not initialized:
                        _vprint(
                            f"[INFO] {label} index does not expose IVF direct-map init "
                            "or does not need reconstruct direct-map."
                        )
                except Exception as e:
                    print(
                        f"[WARN] Failed startup direct-map init for {label} index: {e}",
                        flush=True
                    )

    def _ensure_query_vector(self, vector):
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.size != self.vector_dim:
            raise ValueError(
                f"Query vector has dimension {vec.size}, expected {self.vector_dim}."
            )
        return vec

    def _normalize_vector(self, vector):
        vec = self._ensure_query_vector(vector).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        return vec

    def _sanitize_neighbor_results(self, indices, values, mapping_size=None):
        idx = np.asarray(indices, dtype=np.int64)
        vals = np.asarray(values, dtype=np.float32)

        if mapping_size is None:
            mapping_size = len(self.image_names or [])
        valid = (idx >= 0) & (idx < int(mapping_size))
        if not np.any(valid):
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

        return idx[valid], vals[valid]

    def _faiss_search_l2(self, query_vector, k):
        if self.faiss_l2_index is None:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

        q = self._ensure_query_vector(query_vector).reshape(1, -1).astype(np.float32, copy=False)
        d2, idx = self.faiss_l2_index.search(q, int(k))
        mapping_size = len(self.image_names_l2 or self.image_names or [])
        return self._sanitize_neighbor_results(idx[0], d2[0], mapping_size=mapping_size)

    def _faiss_search_ip(self, query_vector, k):
        if self.faiss_ip_index is None:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

        q = self._normalize_vector(query_vector).reshape(1, -1).astype(np.float32, copy=False)
        scores, idx = self.faiss_ip_index.search(q, int(k))
        mapping_size = len(self.image_names_ip or self.image_names or [])
        return self._sanitize_neighbor_results(idx[0], scores[0], mapping_size=mapping_size)

    def _faiss_range_search_l2(self, query_vector, radius):
        if self.faiss_l2_index is None:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

        q = self._ensure_query_vector(query_vector).reshape(1, -1).astype(np.float32, copy=False)
        lims, d2, idx = self.faiss_l2_index.range_search(q, np.float32(radius * radius))
        start, end = int(lims[0]), int(lims[1])
        mapping_size = len(self.image_names_l2 or self.image_names or [])
        return self._sanitize_neighbor_results(idx[start:end], d2[start:end], mapping_size=mapping_size)

    def _faiss_range_search_ip(self, query_vector, similarity_threshold):
        if self.faiss_ip_index is None:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

        q = self._normalize_vector(query_vector).reshape(1, -1).astype(np.float32, copy=False)
        lims, scores, idx = self.faiss_ip_index.range_search(q, np.float32(similarity_threshold))
        start, end = int(lims[0]), int(lims[1])
        mapping_size = len(self.image_names_ip or self.image_names or [])
        return self._sanitize_neighbor_results(idx[start:end], scores[start:end], mapping_size=mapping_size)

    def _get_vectors_by_indices(self, indices, distance_metric='euclidean'):
        if self.feature_store is not None:
            return self.feature_store.get_vectors_by_indices(indices)
        if self.faiss_enabled:
            return self._reconstruct_vectors_by_indices(indices, distance_metric=distance_metric)
        return self.dataset[np.asarray(indices, dtype=np.int64)]

    def get_vectors_by_indices(self, indices, distance_metric='euclidean'):
        """Return dataset embeddings for a collection of public image indices."""
        if not self.dataset_loaded:
            raise ValueError("Dataset not loaded. Call load_dataset_from_files first.")

        index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        mapping_size = self._mapping_size_for_metric(distance_metric=distance_metric)
        if np.any(index_array < 0) or np.any(index_array >= mapping_size):
            raise IndexError(f"Image index out of bounds for dataset size {mapping_size}.")

        return self._get_vectors_by_indices(index_array, distance_metric=distance_metric)

    def load_dataset_from_files(self, directory, files_list):
        """
        Load dataset from embedding files.

        Args:
            directory: Path to the directory containing embedding files
            files_list: List of .txt files containing embeddings

        Returns:
            tuple: (image_names, dataset)
        """
        if self.feature_store is not None:
            self.feature_store.close()
            self.feature_store = None

        self.faiss_enabled = False
        self.faiss_l2_index = None
        self.faiss_ip_index = None
        self.image_names_l2 = None
        self.image_names_ip = None

        if getattr(config, 'USE_FAISS_INDEX', False):
            _get_faiss()

            self._load_faiss_indices()
            if getattr(config, 'FAISS_SKIP_TXT_LOAD', True):
                if self.faiss_l2_index is not None:
                    self.image_names_l2 = self._load_mapping_for_index('L2', self.faiss_l2_index)
                    if self.image_names_l2 is None:
                        print("[WARN] Disabling L2 FAISS index: no matching L2 mapping file found.", flush=True)
                        self.faiss_l2_index = None

                if self.faiss_ip_index is not None:
                    self.image_names_ip = self._load_mapping_for_index('IP', self.faiss_ip_index)
                    if self.image_names_ip is None:
                        print("[WARN] Disabling IP FAISS index: no matching IP mapping file found.", flush=True)
                        self.faiss_ip_index = None

                self.faiss_enabled = self.faiss_l2_index is not None or self.faiss_ip_index is not None
                if not self.faiss_enabled:
                    raise RuntimeError(
                        "No FAISS index could be aligned with provided mapping files. "
                        "Set SIMSEARCH_IMAGE_NAMES_FILE_L2 / SIMSEARCH_IMAGE_NAMES_FILE_IP correctly."
                    )

                self.image_names = self.image_names_l2 or self.image_names_ip
                if self.image_names_l2 is not None and self.image_names_ip is not None:
                    if len(self.image_names_l2) != len(self.image_names_ip):
                        print(
                            f"[WARN] L2/IP mapping lengths differ: "
                            f"L2={len(self.image_names_l2)}, IP={len(self.image_names_ip)}.",
                            flush=True
                        )

                self._update_vector_dim_from_index()
                self.dataset = FaissDatasetView(self)
            else:
                self.feature_store = TxtFeatureStore(directory, files_list)
                self.image_names, self.vector_dim = self.feature_store.load_metadata()
                self.dataset = LazyDatasetView(self.feature_store)
                ntotal = self._get_faiss_ntotal()
                if ntotal is not None and ntotal != len(self.image_names):
                    print(
                        f"[WARN] FAISS index size mismatch: index ntotal={ntotal}, "
                        f"parsed images={len(self.image_names)}. "
                        "Out-of-range ids will be ignored."
                    )

            self.dataset_loaded = True
            return self.image_names, self.dataset

        # Fallback: original in-memory loader
        self.image_names, dataset = load_dataset(directory, files_list)
        self.dataset = np.asarray(dataset, dtype=np.float32)
        self.vector_dim = self.dataset.shape[1] if self.dataset.size else 768
        self.dataset_loaded = True
        return self.image_names, self.dataset

    def nearest_indices_euclidean(self, anchor_vector, num_indices=50):
        """
        Find the nearest images using Euclidean distance.

        Parameters:
            anchor_vector: The anchor image embedding vector
            num_indices: Number of nearest images to return

        Returns:
            tuple: (nearest_indices, nearest_distances)
        """
        if not self.dataset_loaded:
            raise ValueError("Dataset not loaded. Call load_dataset_from_files first.")

        k = int(num_indices)

        if self.faiss_enabled:
            idx, d2 = self._faiss_search_l2(anchor_vector, k)
            if idx.size > 0:
                return idx, np.sqrt(np.maximum(d2, 0.0))

            # Fallback to IP-derived Euclidean distance for normalized vectors.
            idx, scores = self._faiss_search_ip(anchor_vector, k)
            if idx.size > 0:
                d_e = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.clip(scores, -1.0, 1.0)))
                return idx, d_e

            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

        distances = vectorized_euclidean_distances(anchor_vector, self.dataset)
        nearest_indices = np.argsort(distances)[:k]
        nearest_distances = distances[nearest_indices]
        return nearest_indices, nearest_distances

    def nearest_indices_cosine(self, anchor_vector, num_indices=50):
        """
        Find the nearest images using Cosine distance.

        Parameters:
            anchor_vector: The anchor image embedding vector (should be normalized)
            num_indices: Number of nearest images to return

        Returns:
            tuple: (nearest_indices, nearest_distances)
        """
        if not self.dataset_loaded:
            raise ValueError("Dataset not loaded. Call load_dataset_from_files first.")

        k = int(num_indices)

        if self.faiss_enabled:
            idx, scores = self._faiss_search_ip(anchor_vector, k)
            if idx.size > 0:
                # cosine distance = 1 - cosine similarity
                distances = 1.0 - np.clip(scores, -1.0, 1.0)
                return idx, distances.astype(np.float32, copy=False)

            # Fallback via L2 if IP index is not available and vectors are normalized.
            idx, d2 = self._faiss_search_l2(anchor_vector, k)
            if idx.size > 0:
                distances = 0.5 * np.maximum(d2, 0.0)
                return idx, distances.astype(np.float32, copy=False)

            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

        distances = vectorized_cosine_distances_normalized(anchor_vector, self.dataset)
        nearest_indices = np.argsort(distances)[:k]
        nearest_distances = distances[nearest_indices]
        return nearest_indices, nearest_distances

    def nearest_indices_mahalanobis(
        self,
        anchor_vector,
        metric_matrix,
        num_indices=50,
        base_metric='euclidean',
    ):
        """
        Find the nearest images using Mahalanobis distance.

        Parameters:
            anchor_vector: The anchor image embedding vector
            metric_matrix: The metric matrix for Mahalanobis distance
            num_indices: Number of nearest images to return
            base_metric: FAISS candidate metric ('euclidean' or 'cosine')

        Returns:
            tuple: (nearest_indices, nearest_distances)
        """
        if not self.dataset_loaded:
            raise ValueError("Dataset not loaded. Call load_dataset_from_files first.")

        k = int(num_indices)
        normalized_base_metric = str(base_metric or 'euclidean').strip().lower()
        if normalized_base_metric in {'cosine', 'inner_product', 'dot_product'}:
            normalized_base_metric = 'cosine'
        elif normalized_base_metric != 'euclidean':
            raise ValueError("base_metric must be 'euclidean' or 'cosine'.")

        if self.faiss_enabled:
            # Approximate exact Mahalanobis by refining a larger candidate set from FAISS.
            candidate_count = min(
                self._mapping_size_for_metric(distance_metric=normalized_base_metric),
                max(k, k * self.faiss_candidate_multiplier)
            )

            if normalized_base_metric == 'cosine':
                cand_indices, _ = self.nearest_indices_cosine(anchor_vector, candidate_count)
            else:
                cand_indices, _ = self.nearest_indices_euclidean(anchor_vector, candidate_count)
            if cand_indices.size == 0:
                return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

            # Reconstruct from the same index family used for candidate generation,
            # including the fallback used when the preferred FAISS index is absent.
            if normalized_base_metric == 'cosine':
                refine_metric = 'cosine' if self.faiss_ip_index is not None else 'euclidean'
            else:
                refine_metric = 'euclidean' if self.faiss_l2_index is not None else 'cosine'
            cand_vectors = self._get_vectors_by_indices(cand_indices, distance_metric=refine_metric)
            distances = vectorized_mahalanobis_distances(anchor_vector, cand_vectors, metric_matrix)
            order = np.argsort(distances)[:k]
            return cand_indices[order], distances[order]

        distances = vectorized_mahalanobis_distances(anchor_vector, self.dataset, metric_matrix)
        nearest_indices = np.argsort(distances)[:k]
        nearest_distances = distances[nearest_indices]
        return nearest_indices, nearest_distances

    @staticmethod
    def _resolve_progressive_radius(range_m, range_e, range_stage='full', quick_radius_ratio=0.35):
        stage = str(range_stage or 'full').strip().lower()
        if stage != 'quick':
            return float(range_e), 'full'
        if range_e <= range_m:
            return float(range_e), 'full'

        try:
            ratio = float(quick_radius_ratio)
        except (TypeError, ValueError):
            ratio = 0.35
        ratio = min(1.0, max(0.0, ratio))

        radius = float(range_m + (range_e - range_m) * ratio)
        if radius >= float(range_e):
            return float(range_e), 'full'
        return radius, 'quick'

    def filter_and_refine_search(self, query_vector, metric_matrix, scaling_factor,
                                  num_results=20, growth_factor=1.0, distance_mode='euclidean',
                                  range_stage='full', quick_radius_ratio=0.35, return_meta=False):
        """
        Implement filter and refine approach for personalized search.

        This approach:
        1. Performs kNN search for k nearest using base distance (Euclidean or Dot Product)
        2. Gets rangeM from the k-th nearest neighbor distance
        3. Calculates rangeE = scalingFactor * rangeM * growth_factor
        4. Performs range search with rangeE to get candidate superset
        5. Refines superset with Mahalanobis distance for final k personalized results

        Args:
            query_vector: Query embedding vector
            metric_matrix: User's metric matrix (M)
            scaling_factor: Scaling factor from eigenvalues of M
            num_results: Number of results to return (default 20)
            growth_factor: Optional multiplier for rangeE (default 1.0)
            distance_mode: 'euclidean' or 'dot_product'
            range_stage: 'full' (default) or 'quick' for progressive first pass
            quick_radius_ratio: interpolation ratio in [0,1] for quick stage
            return_meta: if True, also return execution metadata

        Returns:
            tuple: (indices, distances) or (indices, distances, meta)
        """
        if not self.dataset_loaded:
            raise ValueError("Dataset not loaded. Call load_dataset_from_files first.")

        query_vector = self._ensure_query_vector(query_vector)
        k = int(num_results)
        requested_stage = str(range_stage or 'full').strip().lower()
        if requested_stage not in {'full', 'quick'}:
            requested_stage = 'full'
        try:
            quick_ratio = float(quick_radius_ratio)
        except (TypeError, ValueError):
            quick_ratio = 0.35
        quick_ratio = min(1.0, max(0.0, quick_ratio))

        faiss_knn_ms = None
        faiss_range_ms = None
        reconstruct_ms = None
        refine_ms = None

        _vprint(f"\n[SEARCH] FILTER-AND-REFINE SEARCH:")
        _vprint(f"   Distance mode: {distance_mode}")
        _vprint(f"   Stage: {requested_stage}")
        _vprint(f"   Scaling factor: {scaling_factor:.6f}")
        _vprint(f"   Growth factor: {growth_factor}")
        _vprint(f"   Requesting {k} results")

        if self.faiss_enabled:
            if distance_mode == 'dot_product':
                t_knn = time.perf_counter()
                knn_indices, knn_scores = self._faiss_search_ip(query_vector, k)
                faiss_knn_ms = (time.perf_counter() - t_knn) * 1000
                if knn_indices.size == 0:
                    raise ValueError("Dot-product FAISS index search returned no neighbors.")

                s_k = float(knn_scores[min(k - 1, len(knn_scores) - 1)])
                rangeM = dot_product_to_euclidean_distance(s_k)
                rangeE = scaling_factor * rangeM * growth_factor
                search_radius, effective_stage = self._resolve_progressive_radius(
                    rangeM,
                    rangeE,
                    range_stage=requested_stage,
                    quick_radius_ratio=quick_ratio
                )
                similarity_threshold = euclidean_to_dot_product_threshold(search_radius)

                _vprint(f"   k-th similarity (s_k): {s_k:.6f}")
                _vprint(f"   rangeM (Euclidean equiv): {rangeM:.6f}")
                _vprint(f"   rangeE: {rangeE:.6f}")
                _vprint(f"   rangeSearch ({effective_stage}): {search_radius:.6f}")
                _vprint(f"   Similarity threshold: {similarity_threshold:.6f}")

                t_range = time.perf_counter()
                cand_indices, _ = self._faiss_range_search_ip(query_vector, similarity_threshold)
                faiss_range_ms = (time.perf_counter() - t_range) * 1000

            else:
                # Prefer native L2 index for Euclidean filter stage.
                if self.faiss_l2_index is not None:
                    t_knn = time.perf_counter()
                    knn_indices, knn_d2 = self._faiss_search_l2(query_vector, k)
                    faiss_knn_ms = (time.perf_counter() - t_knn) * 1000
                    if knn_indices.size == 0:
                        raise ValueError("Euclidean FAISS index search returned no neighbors.")

                    dE_k = float(np.sqrt(max(float(knn_d2[min(k - 1, len(knn_d2) - 1)]), 0.0)))
                    rangeM = dE_k
                    rangeE = scaling_factor * rangeM * growth_factor
                    search_radius, effective_stage = self._resolve_progressive_radius(
                        rangeM,
                        rangeE,
                        range_stage=requested_stage,
                        quick_radius_ratio=quick_ratio
                    )

                    _vprint(f"   rangeM (k-th distance): {rangeM:.6f}")
                    _vprint(f"   rangeE: {rangeE:.6f}")
                    _vprint(f"   rangeSearch ({effective_stage}): {search_radius:.6f}")

                    t_range = time.perf_counter()
                    cand_indices, _ = self._faiss_range_search_l2(query_vector, search_radius)
                    faiss_range_ms = (time.perf_counter() - t_range) * 1000
                else:
                    # Fallback via IP index + normalized-vector conversion.
                    t_knn = time.perf_counter()
                    knn_indices, knn_scores = self._faiss_search_ip(query_vector, k)
                    faiss_knn_ms = (time.perf_counter() - t_knn) * 1000
                    if knn_indices.size == 0:
                        raise ValueError("No FAISS index available for euclidean filter search.")

                    s_k = float(knn_scores[min(k - 1, len(knn_scores) - 1)])
                    rangeM = dot_product_to_euclidean_distance(s_k)
                    rangeE = scaling_factor * rangeM * growth_factor
                    search_radius, effective_stage = self._resolve_progressive_radius(
                        rangeM,
                        rangeE,
                        range_stage=requested_stage,
                        quick_radius_ratio=quick_ratio
                    )
                    similarity_threshold = euclidean_to_dot_product_threshold(search_radius)

                    _vprint(f"   rangeM (from IP k-th similarity): {rangeM:.6f}")
                    _vprint(f"   rangeE: {rangeE:.6f}")
                    _vprint(f"   rangeSearch ({effective_stage}): {search_radius:.6f}")
                    _vprint(f"   Similarity threshold: {similarity_threshold:.6f}")

                    t_range = time.perf_counter()
                    cand_indices, _ = self._faiss_range_search_ip(query_vector, similarity_threshold)
                    faiss_range_ms = (time.perf_counter() - t_range) * 1000

            _vprint(f"   Candidates found: {len(cand_indices)}")

            if len(cand_indices) == 0:
                _vprint("   [WARN] No candidates in range, falling back to kNN")
                cand_indices = knn_indices

            refine_metric = 'cosine' if distance_mode == 'dot_product' else 'euclidean'
            t_reconstruct = time.perf_counter()
            cand_vectors = self._get_vectors_by_indices(cand_indices, distance_metric=refine_metric)
            reconstruct_ms = (time.perf_counter() - t_reconstruct) * 1000

            t_refine = time.perf_counter()
            maha_distances = vectorized_mahalanobis_distances(
                query_vector, cand_vectors, metric_matrix
            )

            refined_mask = maha_distances <= rangeM
            refined_indices = cand_indices[refined_mask]
            refined_distances = maha_distances[refined_mask]

            _vprint(f"   Refined (d_M <= r_M): {len(refined_indices)} candidates")

            if len(refined_indices) == 0:
                _vprint("   [WARN] No candidates passed d_M <= r_M filter, using all candidates")
                refined_order = np.argsort(maha_distances)[:k]
                final_indices = cand_indices[refined_order]
                final_distances = maha_distances[refined_order]
            else:
                sorted_order = np.argsort(refined_distances)
                final_indices = refined_indices[sorted_order][:k]
                final_distances = refined_distances[sorted_order][:k]

            _vprint(f"   Final results: {len(final_indices)}")
            if len(final_distances) > 0:
                _vprint(f"   Min Mahalanobis distance: {final_distances[0]:.6f}")
                _vprint(f"   Max Mahalanobis distance: {final_distances[-1]:.6f}")
            refine_ms = (time.perf_counter() - t_refine) * 1000

            meta = {
                'requested_stage': requested_stage,
                'stage': effective_stage,
                'range_m': float(rangeM),
                'range_e': float(rangeE),
                'range_search': float(search_radius),
                'num_results_candidate_set': int(len(cand_indices)),
                'num_results_refined_set': int(len(final_indices)),
                'faiss_knn_ms': round(faiss_knn_ms, 3) if faiss_knn_ms is not None else None,
                'faiss_range_ms': round(faiss_range_ms, 3) if faiss_range_ms is not None else None,
                'reconstruct_ms': round(reconstruct_ms, 3) if reconstruct_ms is not None else None,
                'refine_ms': round(refine_ms, 3) if refine_ms is not None else None,
                'quick_radius_ratio': float(quick_ratio),
                'progressive_pending': bool(effective_stage == 'quick' and search_radius < rangeE),
            }
            if return_meta:
                return final_indices, final_distances, meta
            return final_indices, final_distances

        # Legacy in-memory implementation
        if distance_mode == 'dot_product':
            similarities = vectorized_dot_product_similarity(query_vector, self.dataset)
            knn_indices = np.argsort(-similarities)[:k]
            s_k = similarities[knn_indices[-1]]
            rangeM = dot_product_to_euclidean_distance(s_k)
            rangeE = scaling_factor * rangeM * growth_factor
            search_radius, effective_stage = self._resolve_progressive_radius(
                rangeM,
                rangeE,
                range_stage=requested_stage,
                quick_radius_ratio=quick_ratio
            )
            similarity_threshold = euclidean_to_dot_product_threshold(search_radius)

            _vprint(f"   k-th similarity (s_k): {s_k:.6f}")
            _vprint(f"   rangeM (Euclidean equiv): {rangeM:.6f}")
            _vprint(f"   rangeE: {rangeE:.6f}")
            _vprint(f"   rangeSearch ({effective_stage}): {search_radius:.6f}")
            _vprint(f"   Similarity threshold: {similarity_threshold:.6f}")

            cand_indices, _ = dot_product_range_search(
                query_vector, self.dataset, similarity_threshold
            )
        else:
            distances = vectorized_euclidean_distances(query_vector, self.dataset)
            knn_indices = np.argsort(distances)[:k]
            rangeM = distances[knn_indices[-1]]
            rangeE = scaling_factor * rangeM * growth_factor
            search_radius, effective_stage = self._resolve_progressive_radius(
                rangeM,
                rangeE,
                range_stage=requested_stage,
                quick_radius_ratio=quick_ratio
            )

            _vprint(f"   rangeM (k-th distance): {rangeM:.6f}")
            _vprint(f"   rangeE: {rangeE:.6f}")
            _vprint(f"   rangeSearch ({effective_stage}): {search_radius:.6f}")

            cand_indices, _ = euclidean_range_search(
                query_vector, self.dataset, search_radius
            )

        _vprint(f"   Candidates found: {len(cand_indices)}")

        t_reconstruct = time.perf_counter()
        if len(cand_indices) == 0:
            _vprint("   [WARN] No candidates in range, falling back to kNN")
            cand_indices = knn_indices
            cand_vectors = self.dataset[cand_indices]
            reconstruct_ms = (time.perf_counter() - t_reconstruct) * 1000
            t_refine = time.perf_counter()
            maha_distances = vectorized_mahalanobis_distances(
                query_vector, cand_vectors, metric_matrix
            )
            refined_order = np.argsort(maha_distances)[:k]
            final_indices = cand_indices[refined_order]
            final_distances = maha_distances[refined_order]
        else:
            cand_vectors = self.dataset[cand_indices]
            reconstruct_ms = (time.perf_counter() - t_reconstruct) * 1000
            t_refine = time.perf_counter()
            maha_distances = vectorized_mahalanobis_distances(
                query_vector, cand_vectors, metric_matrix
            )
            refined_mask = maha_distances <= rangeM
            refined_indices = cand_indices[refined_mask]
            refined_distances = maha_distances[refined_mask]

            _vprint(f"   Refined (d_M <= r_M): {len(refined_indices)} candidates")

            if len(refined_indices) == 0:
                _vprint("   [WARN] No candidates passed d_M <= r_M filter, using all candidates")
                refined_order = np.argsort(maha_distances)[:k]
                final_indices = cand_indices[refined_order]
                final_distances = maha_distances[refined_order]
            else:
                sorted_order = np.argsort(refined_distances)
                final_indices = refined_indices[sorted_order][:k]
                final_distances = refined_distances[sorted_order][:k]

        _vprint(f"   Final results: {len(final_indices)}")
        if len(final_distances) > 0:
            _vprint(f"   Min Mahalanobis distance: {final_distances[0]:.6f}")
            _vprint(f"   Max Mahalanobis distance: {final_distances[-1]:.6f}")
        refine_ms = (time.perf_counter() - t_refine) * 1000

        meta = {
            'requested_stage': requested_stage,
            'stage': effective_stage,
            'range_m': float(rangeM),
            'range_e': float(rangeE),
            'range_search': float(search_radius),
            'num_results_candidate_set': int(len(cand_indices)),
            'num_results_refined_set': int(len(final_indices)),
            'faiss_knn_ms': round(faiss_knn_ms, 3) if faiss_knn_ms is not None else None,
            'faiss_range_ms': round(faiss_range_ms, 3) if faiss_range_ms is not None else None,
            'reconstruct_ms': round(reconstruct_ms, 3) if reconstruct_ms is not None else None,
            'refine_ms': round(refine_ms, 3) if refine_ms is not None else None,
            'quick_radius_ratio': float(quick_ratio),
            'progressive_pending': bool(effective_stage == 'quick' and search_radius < rangeE),
        }
        if return_meta:
            return final_indices, final_distances, meta
        return final_indices, final_distances

    def search_with_filter_refine(self, query_vector, metric_matrix, scaling_factor,
                                   num_results=20, growth_factor=1.0, distance_mode='euclidean',
                                   range_stage='full', quick_radius_ratio=0.35):
        """
        Search using filter-and-refine and return formatted results.

        Args:
            query_vector: Query embedding vector
            metric_matrix: User's metric matrix
            scaling_factor: Scaling factor from matrix eigenvalues
            num_results: Number of results to return
            growth_factor: Optional multiplier for range
            distance_mode: 'euclidean' or 'dot_product'
            range_stage: 'full' or 'quick' (progressive first pass)
            quick_radius_ratio: interpolation ratio in [0,1] for quick stage

        Returns:
            dict: Dictionary containing search results
        """
        indices, distances, meta = self.filter_and_refine_search(
            query_vector=query_vector,
            metric_matrix=metric_matrix,
            scaling_factor=scaling_factor,
            num_results=num_results,
            growth_factor=growth_factor,
            distance_mode=distance_mode,
            range_stage=range_stage,
            quick_radius_ratio=quick_radius_ratio,
            return_meta=True
        )

        results = []
        name_metric = 'cosine' if distance_mode == 'dot_product' else 'euclidean'
        for idx, dist in zip(indices, distances):
            results.append({
                'index': int(idx),
                'image_name': self._get_image_name_for_metric(int(idx), distance_metric=name_metric),
                'distance': float(dist)
            })

        if distance_mode == 'dot_product':
            base_metric = 'cosine'
        else:
            base_metric = 'euclidean'

        return {
            'query_type': 'vector',
            'distance_metric': 'mahalanobis',
            'base_metric': base_metric,
            'search_pipeline': 'filter+refine',
            'distance_mode': distance_mode,
            'progressive_stage': meta.get('stage', 'full'),
            'progressive_pending': bool(meta.get('progressive_pending', False)),
            'range_m': float(meta.get('range_m', 0.0)),
            'range_e': float(meta.get('range_e', 0.0)),
            'range_search': float(meta.get('range_search', 0.0)),
            'range_r_m_full': float(meta.get('range_m', 0.0)),
            'range_r_e': float(meta.get('range_e', 0.0)),
            'range_progressive': (
                float(meta.get('range_search', 0.0))
                if str(meta.get('stage', 'full')).strip().lower() == 'quick'
                else None
            ),
            'num_results_candidate_set': int(meta.get('num_results_candidate_set', len(results))),
            'num_results_candidate_set_progressive_filter': (
                int(meta.get('num_results_candidate_set', len(results)))
                if str(meta.get('stage', 'full')).strip().lower() == 'quick'
                else None
            ),
            'num_results_candidate_set_full_mahalanobis_filter': (
                int(meta.get('num_results_candidate_set', len(results)))
                if str(meta.get('stage', 'full')).strip().lower() == 'full'
                else None
            ),
            'num_results_refined_set': int(meta.get('num_results_refined_set', len(results))),
            'faiss_knn_ms': meta.get('faiss_knn_ms'),
            'faiss_range_ms': meta.get('faiss_range_ms'),
            'reconstruct_ms': meta.get('reconstruct_ms'),
            'refine_ms': meta.get('refine_ms'),
            'results': results
        }

    def get_image_name_by_index(self, index, distance_metric='euclidean', distance_mode=None):
        if not self.dataset_loaded:
            raise ValueError("Dataset not loaded.")
        return self._get_image_name_for_metric(
            index=index,
            distance_metric=distance_metric,
            distance_mode=distance_mode
        )

    def get_image_by_index(self, index, distance_metric='euclidean'):
        """
        Get image name and embedding by index.

        Args:
            index: Index of the image in the dataset
            distance_metric: Mapping/index family to use ('euclidean' => L2, 'cosine' => IP)

        Returns:
            tuple: (image_name, embedding_vector)
        """
        if not self.dataset_loaded:
            raise ValueError("Dataset not loaded.")

        idx = int(index)
        image_name = self._get_image_name_for_metric(idx, distance_metric=distance_metric)

        if self.feature_store is not None:
            vector = self.feature_store.get_vector_by_index(idx)
        elif self.faiss_enabled:
            vector = self._reconstruct_vectors_by_indices([idx], distance_metric=distance_metric)[0]
        else:
            vector = self.dataset[idx]

        return image_name, vector

    def search_similar_images(self, anchor_index, num_results=50, distance_metric='euclidean', metric_matrix=None):
        """
        Search for similar images given an anchor image index.

        Args:
            anchor_index: Index of the anchor image
            num_results: Number of similar images to return
            distance_metric: Distance metric to use ('euclidean', 'cosine', 'mahalanobis')
            metric_matrix: Metric matrix (required for mahalanobis)

        Returns:
            dict: Dictionary containing image info and distances
        """
        if not self.dataset_loaded:
            raise ValueError("Dataset not loaded.")

        anchor_name, anchor_vector = self.get_image_by_index(anchor_index, distance_metric=distance_metric)

        if distance_metric == 'euclidean':
            indices, distances = self.nearest_indices_euclidean(anchor_vector, num_results)
        elif distance_metric == 'cosine':
            indices, distances = self.nearest_indices_cosine(anchor_vector, num_results)
        elif distance_metric == 'mahalanobis':
            if metric_matrix is None:
                raise ValueError("Metric matrix required for Mahalanobis distance.")
            indices, distances = self.nearest_indices_mahalanobis(anchor_vector, metric_matrix, num_results)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        results = []
        for idx, dist in zip(indices, distances):
            results.append({
                'index': int(idx),
                'image_name': self._get_image_name_for_metric(int(idx), distance_metric=distance_metric),
                'distance': float(dist)
            })

        return {
            'anchor_index': int(anchor_index),
            'anchor_name': anchor_name,
            'distance_metric': distance_metric,
            'results': results
        }

    def search_by_vector(
        self,
        query_vector,
        num_results=50,
        distance_metric='cosine',
        metric_matrix=None,
        base_metric='euclidean',
    ):
        """
        Search for similar images given a query vector (e.g., from text or image).

        Args:
            query_vector: The query embedding vector (numpy array)
            num_results: Number of similar images to return
            distance_metric: Distance metric to use ('euclidean', 'cosine', 'mahalanobis')
            metric_matrix: Metric matrix (required for mahalanobis)
            base_metric: Candidate metric for Mahalanobis search ('euclidean' or 'cosine')

        Returns:
            dict: Dictionary containing search results
        """
        if not self.dataset_loaded:
            raise ValueError("Dataset not loaded.")

        query_vector = np.asarray(query_vector, dtype=np.float32)

        if distance_metric == 'euclidean':
            indices, distances = self.nearest_indices_euclidean(query_vector, num_results)
        elif distance_metric == 'cosine':
            indices, distances = self.nearest_indices_cosine(query_vector, num_results)
        elif distance_metric == 'mahalanobis':
            if metric_matrix is None:
                raise ValueError("Metric matrix required for mahalanobis distance")
            indices, distances = self.nearest_indices_mahalanobis(
                query_vector,
                metric_matrix,
                num_results,
                base_metric=base_metric,
            )
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        result_mapping_metric = base_metric if distance_metric == 'mahalanobis' else distance_metric
        results = []
        for idx, dist in zip(indices, distances):
            results.append({
                'index': int(idx),
                'image_name': self._get_image_name_for_metric(
                    int(idx),
                    distance_metric=result_mapping_metric,
                ),
                'distance': float(dist)
            })

        return {
            'query_type': 'vector',
            'distance_metric': distance_metric,
            'base_metric': result_mapping_metric if distance_metric == 'mahalanobis' else None,
            'results': results
        }

    def get_random_images(self, num_images=20):
        """
        Get random images from the dataset.

        Args:
            num_images: Number of random images to return

        Returns:
            list: List of dictionaries containing image info
        """
        if not self.dataset_loaded:
            raise ValueError("Dataset not loaded.")

        num_available = len(self.image_names)
        num_to_return = min(int(num_images), num_available)
        if num_to_return <= 0:
            return []

        # numpy.choice(replace=False) is expensive on very large populations.
        # random.sample(range(N), k) is O(k) and much faster for small k (e.g. 20 of 20M).
        random_indices = random.sample(range(num_available), num_to_return)

        results = []
        for idx in random_indices:
            results.append({
                'index': int(idx),
                'image_name': self.image_names[int(idx)]
            })

        return results


# Create a singleton instance
image_similarity_service = ImageSimilarityService()
