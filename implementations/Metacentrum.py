from __future__ import annotations
from disa_dataset_loader import load_dataset
from query_dataset_loader import write_vectors_for_images
from dataset_loader import load_dataset
from utils import *
import timeit

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import math
import pandas as pd
import csv
import os
import glob
import json
import shutil
import random
import faiss
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- NEW: knobs for big runs (place near imports) ---
CHUNK_SIZE = 2_000_000          # how many vectors to load per batch when building/querying
TRAIN_SAMPLE = 1_000_000        # how many vectors to train IVFPQ on (cap)
EMB_MEMMAP_DIR = "/storage/plzen1/home/xsikyna/PhD/laion_large/AutofaissL2/memmap"
EMB_MEMMAP_DTYPE = np.float16   # use float16 on disk; upcast to float32 per chunk
FORCE_INT_KEYS = True           # store keys as int indices for huge runs (saves tens of GB)
FAISS_THREADS = int(os.environ.get("FAISS_THREADS", "32"))

os.environ.setdefault("OMP_NUM_THREADS", str(FAISS_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(FAISS_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(FAISS_THREADS))

try:
    faiss.omp_set_num_threads(FAISS_THREADS)
except Exception:
    pass

# --- NEW: create/reuse a disk-backed memmap with the first `target_size` rows ---
def build_or_open_memmap_from_shards(dirpath: str, target_size: int, dtype=np.float16):
    os.makedirs(EMB_MEMMAP_DIR, exist_ok=True)
    # file name encodes size + dtype so we can reuse
    tag = f"laion_first_{target_size}_{np.dtype(dtype).name}"
    mmap_path = os.path.join(EMB_MEMMAP_DIR, f"{tag}.mmap")
    shape = (target_size, 768)

    if os.path.exists(mmap_path):
        emb_mm = np.memmap(mmap_path, mode="r", dtype=dtype, shape=shape)
        return emb_mm, mmap_path

    # create and fill once, streaming shards
    pairs = _index_shards(dirpath)
    emb_mm = np.memmap(mmap_path, mode="w+", dtype=dtype, shape=shape)
    written = 0
    for _, emb_path, meta_path in pairs:
        if written >= target_size:
            break
        emb_shard = np.load(emb_path, mmap_mode="r")   # (Ns,768), float32 or float16
        n_take = min(len(emb_shard), target_size - written)
        # write in sub-batches to reduce peak RAM
        step = 5_000_000
        for s in range(0, n_take, step):
            e = min(n_take, s + step)
            chunk = emb_shard[s:e]
            if chunk.dtype != dtype:
                chunk = chunk.astype(dtype, copy=False)
            emb_mm[written + s: written + e] = chunk
        written += n_take

    emb_mm.flush()
    emb_mm = np.memmap(mmap_path, mode="r", dtype=dtype, shape=shape)
    return emb_mm, mmap_path

log_file_path = "/storage/brno2/home/xsikyna/PhD/metric_learning_exp_3/results/laion_large/AutofaissL2/output_log.txt"
log_file = open(log_file_path, "w", buffering=1)
sys.stdout = log_file
sys.stderr = log_file

os.dup2(log_file.fileno(), sys.stdout.fileno())
os.dup2(log_file.fileno(), sys.stderr.fileno())

# Optional: For newer Python (3.7+), make sys.stdout/stderr line-buffered in Python layer too
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass  # older Python, fine

print("=== Logging started ===")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os, re, glob, random
from typing import List, Tuple, Optional, Sequence
import numpy as np

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas (with a parquet engine) is required to read metadata_*.parquet") from e

_SHARD_RE_EMB   = re.compile(r"^img_emb_(?P<tok>.+)\.npy$", re.IGNORECASE)
_SHARD_RE_META  = re.compile(r"^metadata_(?P<tok>.+)\.parquet$", re.IGNORECASE)

# fields we prefer (first that exists wins) to create stable IDs
_DEFAULT_ID_FIELDS: Sequence[str] = ("sha256", "id", "uid", "image_id", "URL", "url")

def _index_shards(dirpath: str) -> List[Tuple[str, str, str]]:
    """
    Discover shard pairs inside dirpath and return a list of tuples:
      (token, emb_path, meta_path), sorted by token.
    A token is whatever {x} matches in your filenames.
    """
    emb_paths  = glob.glob(os.path.join(dirpath, "img_emb_*.npy"))
    meta_paths = glob.glob(os.path.join(dirpath, "metadata_*.parquet"))

    emb_by_tok = {}
    for p in emb_paths:
        m = _SHARD_RE_EMB.match(os.path.basename(p))
        if m:
            emb_by_tok[m.group("tok")] = p

    meta_by_tok = {}
    for p in meta_paths:
        m = _SHARD_RE_META.match(os.path.basename(p))
        if m:
            meta_by_tok[m.group("tok")] = p

    tokens = sorted(set(emb_by_tok.keys()) & set(meta_by_tok.keys()),
                    key=lambda t: (int(t) if t.isdigit() else t))

    pairs = []
    missing = []
    for tok in sorted(set(emb_by_tok.keys()) ^ set(meta_by_tok.keys())):
        missing.append(tok)
    if missing:
        # Not fatal; we’ll just use the intersection.
        # You can print/log this if you want a stricter check.
        pass

    for tok in tokens:
        pairs.append((tok, emb_by_tok[tok], meta_by_tok[tok]))
    if not pairs:
        raise FileNotFoundError(
            f"No shard pairs found in {dirpath}. "
            "Expect files like img_emb_{x}.npy and metadata_{x}.parquet"
        )
    return pairs

def _make_ids_from_meta(df: "pd.DataFrame",
                        shard_token: str,
                        preferred_fields: Sequence[str]) -> np.ndarray:
    """
    Build string IDs from metadata with a preferred field order.
    Falls back to row index if none are present.
    """
    field = None
    for f in preferred_fields:
        if f in df.columns:
            field = f
            break

    if field is not None:
        vals = df[field].astype(str).to_numpy()
        # Prepend shard token to keep IDs unique across shards (and short if URL repeats)
        ids = np.array([f"{shard_token}:{v}" for v in vals], dtype=object)
    else:
        # deterministic fallback
        n = len(df)
        ids = np.array([f"{shard_token}:{i}" for i in range(n)], dtype=object)
    return ids

def _ensure_float32_768(emb: np.ndarray, where: str) -> np.ndarray:
    if emb.ndim != 2 or emb.shape[1] != 768:
        raise ValueError(f"{where}: expected embeddings of shape (N,768), got {emb.shape}")
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32, copy=False)
    return emb

def load_laion_for_pipeline(
    dirpath: str,
    target_size: int,
    pick_queries: int = 128,
    assume_unit_norm: bool = True,
    normalize_if_needed: bool = True,
    rng_seed: int = 123,
    id_fields: Sequence[str] = _DEFAULT_ID_FIELDS,
    mmap_embeddings: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Returns
    -------
    feature_vectors_keys   : np.ndarray (N,)
    feature_vectors_values : np.ndarray (N,768) or np.memmap
    selected_indices       : List[int]
    """
    pairs = _index_shards(dirpath)

    # -------- FAST PATH FOR HUGE DATASETS (avoid building all_ids/all_emb) --------
    LARGE_THRESHOLD = 10_000_000  # 10M+: go straight to disk-backed memmap
    if target_size > LARGE_THRESHOLD:
        emb_mm, _mmap_path = build_or_open_memmap_from_shards(
            dirpath, target_size, dtype=EMB_MEMMAP_DTYPE
        )
        # Use compact numeric keys to avoid hundreds of GB of Python strings.
        ids_arr = np.arange(len(emb_mm), dtype=np.int64) if FORCE_INT_KEYS else np.arange(len(emb_mm)).astype(object)

        rng = random.Random(rng_seed)
        pick = min(pick_queries, len(emb_mm))
        selected_indices = sorted(rng.sample(range(len(emb_mm)), k=pick))
        return ids_arr, emb_mm, selected_indices
    # -----------------------------------------------------------------------------

    # SMALL/MEDIUM PATH (unchanged behavior)
    all_ids: List[np.ndarray] = []
    all_emb: List[np.ndarray] = []
    total = 0

    for tok, emb_path, meta_path in pairs:
        if total >= target_size:
            break

        emb = np.load(emb_path, mmap_mode="r" if mmap_embeddings else None)
        emb = _ensure_float32_768(emb, emb_path)

        meta_df = pd.read_parquet(meta_path)

        n = min(len(emb), len(meta_df))
        if n == 0:
            continue

        ids = _make_ids_from_meta(meta_df.iloc[:n], tok, id_fields)
        emb = emb[:n]

        need = target_size - total
        if n > need:
            ids = ids[:need]
            emb = emb[:need]
            n = need

        all_ids.append(ids)
        all_emb.append(emb)
        total += n

    if total == 0:
        raise RuntimeError(f"Collected zero rows from {dirpath}. Check shard contents.")

    SMALL_THRESHOLD = 100_000
    if target_size <= SMALL_THRESHOLD:
        ids_arr = np.concatenate(all_ids, axis=0).astype(object)
        emb_arr = np.vstack(all_emb).astype(np.float32, copy=False)

        if normalize_if_needed:
            eps = 1e-12
            emb_arr /= np.maximum(np.linalg.norm(emb_arr, axis=1, keepdims=True), eps)

        rng = random.Random(rng_seed)
        pick = min(pick_queries, len(emb_arr))
        selected_indices = sorted(rng.sample(range(len(emb_arr)), k=pick))
        return ids_arr, emb_arr, selected_indices

    # (For target_size in (SMALL_THRESHOLD, LARGE_THRESHOLD]] you can either
    # fall back to concatenation or add an intermediate memmap path if you like.)
    ids_arr = np.concatenate(all_ids, axis=0).astype(object)
    emb_arr = np.vstack(all_emb).astype(np.float32, copy=False)

    rng = random.Random(rng_seed)
    pick = min(pick_queries, len(emb_arr))
    selected_indices = sorted(rng.sample(range(len(emb_arr)), k=pick))
    return ids_arr, emb_arr, selected_indices



def peek_laion(
    dirpath: str,
    shards: int = 3,
    rows: int = 5,
    id_fields: Sequence[str] = _DEFAULT_ID_FIELDS,
) -> None:
    """
    Print a quick summary of the first few shard pairs so you can see which
    metadata columns are present and what ID will be used.
    """
    pairs = _index_shards(dirpath)
    print(f"Found {len(pairs)} shard pair(s) under: {dirpath}\n")

    for tok, emb_path, meta_path in pairs[:max(1, shards)]:
        emb = np.load(emb_path, mmap_mode="r")
        shape, dtype = emb.shape, emb.dtype

        meta_df = pd.read_parquet(meta_path)
        n = min(len(meta_df), shape[0])

        chosen_field = next((f for f in id_fields if f in meta_df.columns), None)
        present_pref = [f for f in id_fields if f in meta_df.columns]

        print("=" * 80)
        print(f"Shard token      : {tok}")
        print(f"Emb shape/dtype  : {shape} / {dtype}")
        print(f"Rows (aligned)   : {n}")
        print(f"Columns          : {list(meta_df.columns)}")
        print(f"Preferred present: {present_pref if present_pref else 'None'}")
        print(f"Chosen ID field  : {chosen_field if chosen_field else '(fallback to index)'}")

        if n > 0:
            ids = _make_ids_from_meta(meta_df.iloc[:min(rows, n)], tok, id_fields)
            print("\nSample rows:")
            print(meta_df.head(rows))
            print("\nSample IDs:")
            for s in ids:
                print(f"  {s}")
        print()

peek_laion("/storage/plzen1/home/xsikyna/PhD/laion/laionDir", shards=2, rows=5)

def euclidean_distances(target_vector, feature_vectors_keys, feature_vectors_values):
    # Compute differences all at once
    delta = feature_vectors_values - target_vector
    
    # Calculate Euclidean distances all at once
    distances = np.linalg.norm(delta, axis=1)
    
    return dict(zip(feature_vectors_keys, distances))

def mahalanobis_distances(target_vector, feature_vectors_keys, feature_vectors_values, inv_cov_matrix):
    # Compute differences all at once
    delta = feature_vectors_values - target_vector

    # Vectorized computation of Mahalanobis distances all at once
    distances = np.sqrt(np.sum(np.dot(delta, inv_cov_matrix) * delta, axis=1))

    return dict(zip(feature_vectors_keys, distances))

def euclidean_distance_two_points(v, u):
    return np.linalg.norm(v - u)

def calculate_scaling_factor(A):
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    min_eigenvalue = np.min(eigenvalues)
    small_cap = 1e-20
    scaling_factor = 10
    if min_eigenvalue >= small_cap:
        scaling_factor = 1 / np.sqrt(min_eigenvalue)
    if scaling_factor > 20:
        print(min_eigenvalue)

    return np.real(scaling_factor)


k_list = [10, 100]
growth_factors = [1.0]
# dataset_sizes = [1000000, 5000000, 10000000, 20000000, 50000000, 100000000, 200000000, 300000000]
dataset_sizes = [100_000_000, 200_000_000, 400_000_000]
# indices = ["AutofaissL2", "AutofaissIP", "FlatL2", "FlatIP", "IVFPQ"]
indices = ["AutofaissL2"]
# indices = ["FlatL2", "FlatIP"]
methods = [
    ("Balance", np.load(open("/storage/brno2/home/xsikyna/PhD/metric_learning_exp_3/inputs/Balance.npy", 'rb'))),
    # ("Balance+ALT", np.load(open("/storage/brno2/home/xsikyna/PhD/metric_learning_exp_3/inputs/Balance+ALT.npy", 'rb'))),
    # ("Balance+Corr", np.load(open("/storage/brno2/home/xsikyna/PhD/metric_learning_exp_3/inputs/Balance+Corr.npy", 'rb'))),
    ("Min", np.load(open("/storage/brno2/home/xsikyna/PhD/metric_learning_exp_3/inputs/Min.npy", 'rb'))),
    # ("Min+ALT", np.load(open("/storage/brno2/home/xsikyna/PhD/metric_learning_exp_3/inputs/Min+ALT.npy", 'rb'))),
    # ("Min+Corr", np.load(open("/storage/brno2/home/xsikyna/PhD/metric_learning_exp_3/inputs/Min+Corr.npy", 'rb'))),
    # ("Max", np.load(open("/storage/brno2/home/xsikyna/PhD/metric_learning_exp_3/inputs/Max.npy", 'rb')))
    ("Max+ALT", np.load(open("/storage/brno2/home/xsikyna/PhD/metric_learning_exp_3/inputs/Max+ALT.npy", 'rb')))
    # ("Max+Corr", np.load(open("/storage/brno2/home/xsikyna/PhD/metric_learning_exp_3/inputs/Max+Corr.npy", 'rb')))
]

IVFPQ_PARAM_GRID = {
    "nlist":  [4096, 8192, 16384, 32768, 65536],
    "m":      [32, 64, 128],
    "nbits":  [8, 10, 12, 14],
    "nprobe": [16, 32, 48],
}

# AUTOFAISS_PARAM_GRID = {
#     "max_index_memory_usage":      ["16G", "32G", "64G"],
#     "current_memory_available":    ["72G", "128G", "192G"],
#     "max_index_query_time_ms":     [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
#     "min_nearest_neighbors_to_retrieve": [5, 10, 20, 50, 100, 200, 500, 1000],
# }

AUTOFAISS_PARAM_GRID = {
    "max_index_memory_usage":      ["64G"],
    "current_memory_available":    ["512G"],
    "max_index_query_time_ms":     [10.0],
    "min_nearest_neighbors_to_retrieve": [100],
}

dataset_name = "LAION2B-en"  # will appear in CSV and index filenames
# Point this to the folder that contains both file types
LAION_DIR = "/storage/plzen1/home/xsikyna/PhD/laion/laionDir"

feature_vectors_keys, feature_vectors_values, selected_indices = load_laion_for_pipeline(
    dirpath=LAION_DIR,
    target_size=400_000_000,  # e.g., 20_000_000 if you have the resources; start smaller
    pick_queries=1000,
    assume_unit_norm=False,
    normalize_if_needed=True,
    rng_seed=42,
    mmap_embeddings=True,           # set True if your .npy shards are huge
)

import numpy as np
from scipy import stats

import csv
import numpy as np
from autofaiss import build_index
import os
from scipy import stats

def make_subset_emb_dir(src_dir: str, size: int, out_dir: str) -> str:
    """
    Create a directory that contains exactly `size` embeddings rows as .npy shards:
      - Full shards are hardlinked (cheap; no data copy).
      - The final partial shard is physically written, but only its needed rows.
    File names are preserved (img_emb_{tok}.npy) to keep the same shard order.
    """
    os.makedirs(out_dir, exist_ok=True)
    pairs = _index_shards(src_dir)  # [(tok, emb_path, meta_path)] sorted like your loader
    remain = size

    for tok, emb_path, _ in pairs:
        if remain <= 0:
            break
        dst = os.path.join(out_dir, f"img_emb_{tok}.npy")
        if os.path.exists(dst):
            # Already created (resume-friendly)
            n = len(np.load(dst, mmap_mode="r"))
            remain -= n
            continue

        emb = np.load(emb_path, mmap_mode="r")  # (N, 768)
        n = len(emb)

        if n <= remain:
            # Full shard fits -> hardlink (or symlink)
            try:
                os.link(emb_path, dst)
            except OSError:
                os.symlink(emb_path, dst)
            remain -= n
        else:
            # Need only a prefix of this shard -> write a trimmed .npy without loading all to RAM
            need = remain
            tmp = np.lib.format.open_memmap(dst, mode="w+",
                                            dtype=emb.dtype,
                                            shape=(need, emb.shape[1]))
            step = 5_000_000
            for s in range(0, need, step):
                e = min(need, s + step)
                tmp[s:e] = emb[s:e]
            del tmp  # flush header + data
            remain = 0

    if remain > 0:
        raise RuntimeError(f"Requested {size} rows but only {size - remain} available under {src_dir}.")
    return out_dir


def params_json_str(params: dict) -> str:
    """Compact, deterministic JSON-like string."""
    if not params:
        return "{}"
    # sort keys for stability
    return json.dumps(params, sort_keys=True, separators=(',', ':'))

def params_tag(params: dict) -> str:
    """Create a safe filename tag."""
    if not params:
        return ""
    parts = []
    for k in sorted(params.keys()):
        v = params[k]
        if isinstance(v, float):
            # use compact float formatting, preserve small values
            parts.append(f"{k}-{v:g}")
        else:
            parts.append(f"{k}-{v}")
    return "_".join(parts)
    
# add at top of file near imports
from datetime import datetime
import json
import platform

def save_autofaiss_metadata(base_index_path: str,
                            dataset_name: str,
                            size: int,
                            index_name: str,
                            metric: str,
                            input_params: dict,
                            index_infos: dict,
                            embedding_file: str):
    """
    Writes a compact JSON sidecar containing your grid params + AutoFaiss index_infos.
    """
    meta = {
        "dataset_name": dataset_name,
        "dataset_size": size,
        "index_name": index_name,
        "metric": metric,                       # "l2" or "ip"
        "input_params": input_params,           # your grid-search inputs
        "index_path": base_index_path,          # .faiss
        "embedding_file": embedding_file,       # saved embeddings .npy
        "timestamp": datetime.now().isoformat(),
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "autofaiss_index_infos": index_infos,   # what AutoFaiss discovered/built
    }
    out = base_index_path + ".meta.json"
    with open(out, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

import gc
import shutil

def safe_remove(path: str):
    try:
        if os.path.isdir(path):
            # use rmdir only if empty; otherwise leave it alone
            if not os.listdir(path):
                os.rmdir(path)
        elif os.path.isfile(path):
            os.remove(path)
    except FileNotFoundError:
        pass

def cleanup_autofaiss_artifacts(index_path: str, emb_file: str = None):
    """
    Remove the heavy artifacts created for AutoFaiss:
    - index_path ('.faiss')
    - index_path + '.json'      (AutoFaiss index_infos)
    - index_path + '.meta.json' (your merged metadata)
    - emb_file                  (<size>_embeddings.npy) if provided
    - parent embedding dir if now empty
    """
    # Make sure no one holds the index in memory
    gc.collect()

    # Delete index + sidecars
    for fp in [index_path]:
        safe_remove(fp)

    # Delete the specific embeddings file, not the whole folder (other sizes may live there)
    if emb_file:
        parent = os.path.dirname(emb_file)
        safe_remove(emb_file)
        # clean the parent folder if it became empty
        safe_remove(parent)

from collections import Counter
from threading import Lock

def index_identity_key(dataset_name, index_name, size, params):
    # params that define the physical index on disk
    tag_needed = index_name in ("IVFPQ", "AutofaissL2", "AutofaissIP")
    tag = params_tag(params) if tag_needed else ""
    return (dataset_name, index_name, size, tag)

# global-ish counters guarded by a lock
_index_use_counts = Counter()
_index_use_lock = Lock()
_built_idents = set()
_emb_file_for_ident = {}

def make_or_load_index(dataset_name: str, index_name: str, size: int, params: dict):
    base_dir = "/storage/brno2/home/xsikyna/PhD/metric_learning_exp_3/results/laion_large/AutofaissL2"
    os.makedirs(base_dir, exist_ok=True)

    tag_needed = (index_name in ("IVFPQ", "AutofaissL2", "AutofaissIP"))
    tag = params_tag(params) if tag_needed else ""
    fname = f"{dataset_name}_{index_name}_{size}" + (f"_{tag}" if tag else "") + ".faiss"
    fpath = os.path.join(base_dir, fname)

    ident = index_identity_key(dataset_name, index_name, size, params)

    if os.path.exists(fpath):
        return faiss.read_index(fpath), fpath, None, False

    # IMPORTANT: never materialize the whole array in RAM
    data_view = feature_vectors_values[:size]   # can be ndarray or memmap
    D = data_view.shape[1]

    if index_name in ("AutofaissL2", "AutofaissIP"):
        # Avoid writing a huge single .npy; let AutoFaiss stream from your shard dir.
        # Keep your existing directory layout and tell AutoFaiss to save on disk.
        if index_name == "AutofaissL2":
            metric = "l2"
        else:
            metric = "ip"
        
        # NEW: build a subset directory that exposes exactly `size` rows
        subset_root = os.path.join(base_dir, "subset_views")
        subset_dir  = os.path.join(subset_root, f"{dataset_name}_{size}")
        subset_dir  = make_subset_emb_dir(LAION_DIR, size, subset_dir)

        index, index_infos = build_index(
            embeddings=subset_dir,                  # <--- directory with img_emb_*.npy
            index_path=fpath,
            index_infos_path=fpath + ".json",
            metric_type=metric,
            max_index_memory_usage=params.get("max_index_memory_usage", "4G"),
            current_memory_available=params.get("current_memory_available", "8G"),
            max_index_query_time_ms=params.get("max_index_query_time_ms", 1.0),
            min_nearest_neighbors_to_retrieve=params.get("min_nearest_neighbors_to_retrieve", 20),
            use_gpu=False,
            save_on_disk=True                      # <--- IMPORTANT for big builds
        )
        
        faiss.write_index(index, fpath)
        save_autofaiss_metadata(
            base_index_path=fpath, dataset_name=dataset_name, size=size,
            index_name=index_name, metric=metric, input_params=params,
            index_infos=index_infos, embedding_file=subset_dir
        )
        with _index_use_lock:
            _built_idents.add(ident)
            _emb_file_for_ident[ident] = subset_dir
        return faiss.read_index(fpath), fpath, subset_dir, True

    if index_name == "FlatL2":
        index = faiss.IndexFlatL2(D)
    elif index_name == "FlatIP":
        index = faiss.IndexFlatIP(D)
    elif index_name == "IVFPQ":
        nlist = int(params["nlist"])
        m     = int(params["m"])
        nbits = int(params["nbits"])
        if D % m != 0:
            raise ValueError(f"Invalid IVFPQ config: D={D} not divisible by m={m}.")
        quantizer = faiss.IndexFlatL2(D)
        index = faiss.IndexIVFPQ(quantizer, D, nlist, m, nbits)

        # Train on a bounded random sample
        train_n = min(TRAIN_SAMPLE, size)
        rng = np.random.RandomState(123)
        train_idx = rng.choice(size, size=train_n, replace=False)
        # stream the training selection in sub-batches to avoid large gathers
        step = 1_000_000
        train_buf = np.empty((train_n, D), dtype=np.float32)
        w = 0
        for s in range(0, train_n, step):
            e = min(train_n, s + step)
            idx_slice = train_idx[s:e]
            # group by contiguous blocks to reduce random I/O
            idx_slice_sorted = np.sort(idx_slice)
            # pull in smaller chunks
            substep = 200_000
            cur = 0
            while cur < len(idx_slice_sorted):
                cur_e = min(len(idx_slice_sorted), cur + substep)
                block = idx_slice_sorted[cur:cur_e]
                X = data_view[block]
                if X.dtype != np.float32:
                    X = X.astype(np.float32, copy=False)
                train_buf[w:w+len(block)] = X
                w += len(block)
                cur = cur_e
        index.train(train_buf[:w])

    else:
        raise ValueError(f"Unknown index: {index_name}")

    # Add vectors in batches for any index type
    for start in range(0, size, CHUNK_SIZE):
        end = min(size, start + CHUNK_SIZE)
        X = data_view[start:end]
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        index.add(X)

    faiss.write_index(index, fpath)
    with _index_use_lock:
        _built_idents.add(ident)

    return index, fpath, None, True        


def knn_all(q_vec, values, keys, M, k):
    d = mahalanobis_distances(q_vec, keys, values, M)
    return sorted(d.items(), key=lambda x: x[1])[:k]

def describe_array(data, which):
    arr = np.asarray(data, dtype=np.float64)
    # keep only finite values
    arr = arr[np.isfinite(arr)]
    out = {which + '_count': int(arr.size)}
    if arr.size == 0:
        # fill with NaNs when no valid data
        for name in ['min','Q1','median','Q3','Q90','Q95','Q99','max',
                     'mean','variance','std_dev','skewness','kurtosis','IQR']:
            out[f"{which}_{name}"] = float('nan')
        return out

    desc = stats.describe(arr, ddof=1)
    pct  = np.percentile(arr, [0,25,50,75,90,95,99,100])
    out.update({
        which + '_min':      pct[0],
        which + '_Q1':       pct[1],
        which + '_median':   pct[2],
        which + '_Q3':       pct[3],
        which + '_Q90':      pct[4],
        which + '_Q95':      pct[5],
        which + '_Q99':      pct[6],
        which + '_max':      pct[7],
        which + '_mean':     desc.mean,
        which + '_variance': desc.variance,
        which + '_std_dev':  np.sqrt(desc.variance),
        which + '_skewness': desc.skewness,
        which + '_kurtosis': desc.kurtosis,
        which + '_IQR':      pct[3] - pct[1],
    })
    return out


CSV_PATH = '/storage/brno2/home/xsikyna/PhD/metric_learning_exp_3/results/laion_large/AutofaissL2/results.csv'
FIELDNAMES = [
    'dataset','method','k','growth', 'dataset_size', 'scaling_factor', 'index', 'index_params',
    'count_superset',
    'count_sameset',
    'count_subset',
    'candidate_count','candidate_min','candidate_Q1','candidate_median','candidate_Q3',
    'candidate_Q90', 'candidate_Q95', 'candidate_Q99', 'candidate_max',
    'candidate_mean','candidate_variance','candidate_std_dev','candidate_skewness','candidate_kurtosis','candidate_IQR',
    'refined_count','refined_min','refined_Q1','refined_median','refined_Q3',
    'refined_Q90', 'refined_Q95', 'refined_Q99', 'refined_max',
    'refined_mean','refined_variance','refined_std_dev','refined_skewness','refined_kurtosis','refined_IQR',
    # NEW: r_E (“range for filter”)
    'range_for_filter_count','range_for_filter_min','range_for_filter_Q1','range_for_filter_median','range_for_filter_Q3',
    'range_for_filter_Q90','range_for_filter_Q95','range_for_filter_Q99','range_for_filter_max',
    'range_for_filter_mean','range_for_filter_variance','range_for_filter_std_dev','range_for_filter_skewness',
    'range_for_filter_kurtosis','range_for_filter_IQR',
    # NEW: distance to k-th item in refined (“range from refined”)
    'range_from_refined_count','range_from_refined_min','range_from_refined_Q1','range_from_refined_median','range_from_refined_Q3',
    'range_from_refined_Q90','range_from_refined_Q95','range_from_refined_Q99','range_from_refined_max',
    'range_from_refined_mean','range_from_refined_variance','range_from_refined_std_dev','range_from_refined_skewness',
    'range_from_refined_kurtosis','range_from_refined_IQR',
    'time_candidates'
]

done = set()
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # cast back to the right types
            dataset = row['dataset']
            method = row['method']
            k      = int(row['k'])
            growth = float(row['growth'])
            dataset_size = int(row['dataset_size'])
            index = row['index']
            index_params_str = row.get('index_params', '{}')
            done.add((dataset, method, k, growth, dataset_size, index, index_params_str))

if not done:
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

csv_lock = threading.Lock()

import re
import matplotlib
matplotlib.use("Agg")              # headless safe
import matplotlib.pyplot as plt

def safe_tag_text(s: str) -> str:
    # keep it filename-friendly
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', str(s))

def make_result_stem(dataset_name: str, method_name: str, k: int, growth: float,
                     size: int, index_name: str, index_params: dict) -> str:
    gf = f"{growth:g}"
    stem = (
        f"{safe_tag_text(dataset_name)}_"
        f"{safe_tag_text(method_name)}_"
        f"k{k}_g{gf}_size{size}_"
    )
    return stem

def _finite_array(values):
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]

def save_hist(values, out_path: str, title: str, xlabel: str):
    vals = _finite_array(values)
    if vals.size == 0:
        return None
    fig = plt.figure(figsize=(6, 4))
    plt.hist(vals, bins='auto')
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(f"{title} (n={len(vals)})")
    plt.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path

def save_ecdf(values, out_path: str, title: str, xlabel: str):
    vals = _finite_array(values)
    if vals.size == 0:
        return None
    x = np.sort(vals)
    y = np.arange(1, len(x) + 1) / len(x)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(x, y, drawstyle='steps-post')
    plt.xlabel(xlabel)
    plt.ylabel("ECDF")
    plt.title(f"{title} (n={len(vals)})")
    # helpful reference lines
    med = np.median(vals)
    mean = np.mean(vals)
    plt.axvline(med, linestyle='--', linewidth=1, label=f"median={med:g}")
    plt.axvline(mean, linestyle=':',  linewidth=1, label=f"mean={mean:g}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', frameon=False)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def run_one(dataset_name, method_name, M, k, growth_factor, dataset_size, index_name, index_params, cleanup_after=True):
    print(f"[RUNNING] dataset_name={dataset_name}, method={method_name}, k={k}, growth={growth_factor}, dataset_size={dataset_size}, index={index_name}, params={index_params}")
    print()
    scaling_factor = calculate_scaling_factor(M)
    count_superset = 0   # times len(refined) >= k
    count_sameset = 0
    count_subset    = 0   # times len(refined) <  k
    refined_sizes = []
    candidate_sizes = []
    range_for_filter_vals = []     # r_E per query
    range_from_refined_vals = []   # k-th Mahalanobis distance in refined (NaN if <k)

    
    # identity for the physical index on disk
    ident = index_identity_key(dataset_name, index_name, dataset_size, index_params)

    # build / load the index
    ind, index_path, emb_file, built_here = make_or_load_index(dataset_name, index_name, dataset_size, index_params)

    total_time_cand = 0.0
    total_time_knn  = 0.0

    # If IVFPQ, set nprobe from params; otherwise ignore
    if index_name == "IVFPQ":
        ind.nprobe = int(index_params["nprobe"])

    for idx in selected_indices:
        q_vec = feature_vectors_values[idx]
        q_key = feature_vectors_keys[idx]

        t0 = timeit.default_timer()
        if index_name == "FlatL2" or index_name == "AutofaissL2":
            q = np.asarray(q_vec, dtype=np.float32).reshape(1, -1)
            D2, I = ind.search(q, k)
            dE_k = np.sqrt(D2[0, -1])    
            r_M = dE_k
            r_E = r_M * scaling_factor * growth_factor
            thresh2 = np.float32(r_E * r_E)
            lims, dist2, idxs = ind.range_search(q, thresh2)
        elif index_name == "FlatIP" or index_name == "AutofaissIP":
            q = q_vec.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(q)
            # 1) find the k-th neighbor’s *inner-product* score
            scores, I = ind.search(q, k)
            s_k = scores[0, -1]                        # highest = best match
            # 2) recover the *Euclidean* distance of that neighbor:
            #    d_E^2 = 2 − 2·<q,x>   ⇒   dE_k = sqrt(max(0,2-2*s_k))
            r_M = np.sqrt(max(0.0, 2.0 - 2.0 * s_k))
            r_E = r_M * scaling_factor * growth_factor            
            ip_thresh = 1.0 - (r_E * r_E) / 2.0
            ip_thresh = np.clip(ip_thresh, -1.0, 1.0)
            lims, dist2, idxs = ind.range_search(q, ip_thresh)
        elif index_name == "IVFPQ":   # assuming the stored index was built with METRIC_L2
            q = np.asarray(q_vec, dtype=np.float32).reshape(1, -1)
            D2, I = ind.search(q, k)          # D2 are approximate squared L2 distances
            dE_k = np.sqrt(D2[0, -1])
            r_M = dE_k
            r_E = r_M * scaling_factor * growth_factor
            thresh2 = np.float32(r_E * r_E)   # squared L2 radius
            lims, dist2, idxs = ind.range_search(q, thresh2)

        start, end     = lims[0], lims[1]
        cand_idxs      = idxs[start:end]
        cand_keys   = [feature_vectors_keys[i] for i in cand_idxs]
        cand_values = feature_vectors_values[cand_idxs]
        
        # Mahalanobis refinement ------------------
        refined = []
        # for key in cand_keys:
        #     diff = feature_vectors[key] - q_vec
        #     dM = np.sqrt(diff @ M @ diff)
        #     if dM <= r_M:
        #         refined.append((key, dM))
        dists = mahalanobis_distances(q_vec, cand_keys, cand_values, M)  # {key: d_M}
        refined = sorted(
            ((k, d) for k, d in dists.items() if d <= r_M),
            key=lambda x: x[1]
        )
        
        # sort refined by true Mahalanobis distance
        # refined_sorted = sorted(refined, key=lambda x: x[1])
        refined_keys   = [key for key, _ in refined]
        refined_sizes.append(len(refined_keys))
        candidate_sizes.append(len(cand_keys))
        total_time_cand += timeit.default_timer() - t0
        
        if len(refined) >= k:
            kth_dist = float(refined[k-1][1])
            range_for_filter_vals.append(float(r_E))
            range_from_refined_vals.append(kth_dist)

        # t1 = timeit.default_timer()
#         # get the pure-Mahalanobis k-NN
#         knn100 = knn_all(q_vec,
#                          feature_vectors_values,
#                          feature_vectors_keys,
#                          M,
#                          k)
#         knn_keys = [key for key, _ in knn100]
#         total_time_knn += timeit.default_timer() - t1
        
        if len(refined_keys) > k:
            count_superset += 1
        elif len(refined_keys) < k:
            count_subset += 1
        else:
            count_sameset += 1

    candidate_stats = describe_array(candidate_sizes, "candidate")
    refined_stats = describe_array(refined_sizes, "refined")
    
    # NEW summaries
    rE_stats        = describe_array(range_for_filter_vals,   "range_for_filter")
    rRefined_stats  = describe_array(range_from_refined_vals, "range_from_refined")
    
    # Build filename stem & output dir (same folder as the index/CSV)
    out_dir = os.path.dirname(index_path) if index_path else os.path.dirname(CSV_PATH)
    stem = make_result_stem(dataset_name, method_name, k, growth_factor,
                            dataset_size, index_name, index_params)

    # Save plots
    save_hist(range_for_filter_vals,
              os.path.join(out_dir, stem + "_range_for_filter_hist.png"),
              title="Filter radius r_E", xlabel="r_E")

    save_ecdf(range_for_filter_vals,
              os.path.join(out_dir, stem + "_range_for_filter_ecdf.png"),
              title="Filter radius r_E", xlabel="r_E")

    save_hist(range_from_refined_vals,
              os.path.join(out_dir, stem + "_range_from_refined_hist.png"),
              title=f"Refined k-th distance (k={k})",
              xlabel="Mahalanobis distance at rank k")

    save_ecdf(range_from_refined_vals,
              os.path.join(out_dir, stem + "_range_from_refined_ecdf.png"),
              title=f"Refined k-th distance (k={k})",
              xlabel="Mahalanobis distance at rank k")
    
    row = {
        'dataset':      dataset_name,
        'method':       method_name,
        'k':            k,
        'growth':       growth_factor,
        'scaling_factor': scaling_factor,
        'index':        index_name,
        'index_params': params_json_str(index_params),
        'dataset_size': dataset_size,
        'count_superset': count_superset,
        'count_sameset':  count_sameset,
        'count_subset':   count_subset,
        **candidate_stats,
        **refined_stats,
        **rE_stats,               # <--- NEW
        **rRefined_stats,         # <--- NEW
        'time_candidates': total_time_cand
    }
    
    with csv_lock:
        with open(CSV_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(row)

    if cleanup_after and index_name in ("AutofaissL2", "AutofaissIP"):
        with _index_use_lock:
            _index_use_counts[ident] -= 1
            remaining = _index_use_counts[ident]
            built_in_this_run = ident in _built_idents
            # if this call didn't build, recover the emb_file we saved earlier
            if emb_file is None:
                emb_file = _emb_file_for_ident.get(ident, None)

        if built_in_this_run and remaining == 0:
            try:
                del ind
            except Exception:
                pass
            gc.collect()
            cleanup_autofaiss_artifacts(index_path, emb_file)
    
    return row

def build_all_needed_indices():
    built = set()
    for size in dataset_sizes:
        data = feature_vectors_values[:size].astype(np.float32)
        D = data.shape[1]

        # figure out param sets to build for each index/size
        for ind in indices:
            if ind == "IVFPQ":
                # filter m by divisibility
                m_list = [m for m in IVFPQ_PARAM_GRID["m"] if D % m == 0]
                for nlist in IVFPQ_PARAM_GRID["nlist"]:
                    for m in m_list:
                        for nbits in IVFPQ_PARAM_GRID["nbits"]:
                            # Note: nprobe is a query-time param, not needed for training/build filename
                            params = {'nlist': nlist, 'm': m, 'nbits': nbits}
                            key = (ind, size, params_json_str(params))
                            if key in built:
                                continue
                            # build/train/add/save
                            make_or_load_index(dataset_name, ind, size, params)
                            built.add(key)
            elif ind in ("AutofaissL2", "AutofaissIP"):
                # Expand full AutoFaiss grid
                for max_mem in AUTOFAISS_PARAM_GRID["max_index_memory_usage"]:
                    for cur_mem in AUTOFAISS_PARAM_GRID["current_memory_available"]:
                        for t_ms in AUTOFAISS_PARAM_GRID["max_index_query_time_ms"]:
                            for min_nn in AUTOFAISS_PARAM_GRID["min_nearest_neighbors_to_retrieve"]:
                                params = {
                                    "max_index_memory_usage": max_mem,
                                    "current_memory_available": cur_mem,
                                    "max_index_query_time_ms": float(t_ms),
                                    "min_nearest_neighbors_to_retrieve": int(min_nn),
                                }
                                key = (ind, size, params_json_str(params))
                                if key not in built:
                                    make_or_load_index(dataset_name, ind, size, params)
                                    built.add(key)
            else:
                # flat indices get empty params
                params = {}
                key = (ind, size, params_json_str(params))
                if key in built:
                    continue
                make_or_load_index(dataset_name, ind, size, params)
                built.add(key)

if __name__ == "__main__":
    tasks = []
    for d in dataset_sizes:
        for k in k_list:
            for g in growth_factors:
                for (mn, M) in methods:
                    for i in indices:
                        if i == "IVFPQ":
                            # Full grid INCLUDING nprobe (query-time)
                            D = feature_vectors_values[:d].shape[1]
                            m_list = [m for m in IVFPQ_PARAM_GRID["m"] if D % m == 0]
                            for nlist in IVFPQ_PARAM_GRID["nlist"]:
                                for m in m_list:
                                    for nbits in IVFPQ_PARAM_GRID["nbits"]:
                                        for nprobe in IVFPQ_PARAM_GRID["nprobe"]:
                                            params = {'nlist': nlist, 'm': m, 'nbits': nbits, 'nprobe': nprobe}
                                            pstr = params_json_str(params)
                                            key = (dataset_name, mn, k, g, d, i, pstr)
                                            if key not in done:
                                                tasks.append((dataset_name, mn, M, k, g, d, i, params))
                        elif i in ("AutofaissL2", "AutofaissIP"):
                            for max_mem in AUTOFAISS_PARAM_GRID["max_index_memory_usage"]:
                                for cur_mem in AUTOFAISS_PARAM_GRID["current_memory_available"]:
                                    for t_ms in AUTOFAISS_PARAM_GRID["max_index_query_time_ms"]:
                                        for min_nn in AUTOFAISS_PARAM_GRID["min_nearest_neighbors_to_retrieve"]:
                                            params = {
                                                "max_index_memory_usage": max_mem,
                                                "current_memory_available": cur_mem,
                                                "max_index_query_time_ms": float(t_ms),
                                                "min_nearest_neighbors_to_retrieve": int(min_nn),
                                            }
                                            pstr = params_json_str(params)
                                            key = (dataset_name, mn, k, g, d, i, pstr)
                                            if key not in done:
                                                tasks.append((dataset_name, mn, M, k, g, d, i, params))
                        else:
                            params = {}
                            pstr = params_json_str(params)
                            key = (dataset_name, mn, k, g, d, i, pstr)
                            if key not in done:
                                tasks.append((dataset_name, mn, M, k, g, d, i, params))

    with _index_use_lock:
        for (_, mn, _M, k, g, d, i, params) in tasks:
            ident = index_identity_key(dataset_name, i, d, params)
            _index_use_counts[ident] += 1

    with ThreadPoolExecutor(max_workers=1) as exe:
        for _ in exe.map(lambda args: run_one(*args), tasks):
            pass

    print("=== All done ===")