import os
import random

import numpy as np
from django.core.management.base import BaseCommand, CommandError

from base import config


def _resolve_index_path(path_or_dir):
    if not path_or_dir:
        return None

    candidate = os.path.abspath(path_or_dir)
    if os.path.isfile(candidate):
        return candidate

    if not os.path.isdir(candidate):
        return None

    faiss_files = sorted(
        os.path.join(candidate, name)
        for name in os.listdir(candidate)
        if name.lower().endswith(".faiss") and os.path.isfile(os.path.join(candidate, name))
    )
    if faiss_files:
        return faiss_files[0]
    return None


def _ensure_direct_map_for_reconstruct(index, faiss_module, stdout):
    ivf_index = None
    try:
        if hasattr(faiss_module, "extract_index_ivf"):
            ivf_index = faiss_module.extract_index_ivf(index)
    except Exception:
        ivf_index = None

    if ivf_index is None:
        return False

    try:
        stdout.write("Initializing FAISS direct map for reconstruct()...")
        if hasattr(ivf_index, "set_direct_map_type") and hasattr(faiss_module, "DirectMap"):
            try:
                ivf_index.set_direct_map_type(faiss_module.DirectMap.Hashtable)
            except Exception:
                pass

        if hasattr(ivf_index, "make_direct_map"):
            ivf_index.make_direct_map()
        else:
            return False
        stdout.write("Direct map initialized.")
        return True
    except Exception:
        return False


def _reconstruct_vector(index, vector_id, faiss_module, stdout):
    try:
        return index.reconstruct(vector_id)
    except Exception as e:
        if "direct map not initialized" in str(e).lower():
            if _ensure_direct_map_for_reconstruct(index, faiss_module, stdout):
                return index.reconstruct(vector_id)
        raise


def _pick_default_index_path(kind):
    if kind == "ip":
        return getattr(config, "FAISS_IP_INDEX_PATH", "")
    return getattr(config, "FAISS_L2_INDEX_PATH", "")


def _pick_default_mapping_path():
    image_names_file = (getattr(config, "IMAGE_NAMES_FILE", "") or "").strip()
    if image_names_file:
        return image_names_file
    return (getattr(config, "IMAGE_NAMES_CACHE_PATH", "") or "").strip()


def _load_mapping_file(mapping_path):
    path = os.path.abspath(mapping_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Mapping file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="strict") as handle:
        return [line.strip() for line in handle if line.strip()]


def _infer_kind_from_index(index, faiss_module):
    metric_type = getattr(index, "metric_type", None)
    if metric_type is None:
        return "l2"
    if metric_type == getattr(faiss_module, "METRIC_INNER_PRODUCT", -1):
        return "ip"
    return "l2"


class Command(BaseCommand):
    help = (
        "Pick a random vector from a FAISS index and print top-k retrieved results "
        "(IDs and scores/distances). Useful for quick demo checks."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--index-path",
            default="",
            help="Path to .faiss index (or directory containing one).",
        )
        parser.add_argument(
            "--kind",
            choices=["auto", "l2", "ip"],
            default="auto",
            help="Index metric type (default: auto-detect from index metric_type).",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=10,
            help="Number of nearest neighbors to retrieve (default: 10).",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for reproducible query-id selection.",
        )
        parser.add_argument(
            "--query-id",
            type=int,
            default=None,
            help="Use this index id as query instead of random.",
        )
        parser.add_argument(
            "--exclude-self",
            action="store_true",
            help="Exclude the query id itself from printed results.",
        )
        parser.add_argument(
            "--with-names",
            action="store_true",
            help="Also print image names by loading the mapping file.",
        )
        parser.add_argument(
            "--mapping-file",
            default="",
            help="Path to mapping file used with --with-names (default: IMAGE_NAMES_FILE/cache).",
        )

    def handle(self, *args, **options):
        try:
            import faiss
        except Exception as e:
            raise CommandError(f"faiss is not available in this environment: {e}") from e

        requested_kind = options.get("kind", "auto")
        requested_index = (options.get("index_path") or "").strip()
        if not requested_index:
            requested_index = (_pick_default_index_path("ip" if requested_kind == "ip" else "l2") or "").strip()

        resolved_index = _resolve_index_path(requested_index)
        if not resolved_index:
            raise CommandError(f"Could not resolve index path from: {requested_index}")

        self.stdout.write(f"Loading index: {resolved_index}")
        try:
            index = faiss.read_index(resolved_index)
        except Exception as e:
            raise CommandError(f"Failed to load FAISS index: {e}") from e

        try:
            ntotal = int(index.ntotal)
            dim = int(index.d)
        except Exception as e:
            raise CommandError(f"Could not read ntotal/d from index: {e}") from e

        if ntotal <= 0:
            raise CommandError("Index is empty (ntotal=0).")

        kind = requested_kind
        if kind == "auto":
            kind = _infer_kind_from_index(index, faiss)

        top_k = max(1, int(options.get("top_k", 10)))
        exclude_self = bool(options.get("exclude_self"))
        seed = options.get("seed")
        query_id = options.get("query_id")

        if query_id is None:
            rng = random.Random(seed)
            query_id = rng.randrange(ntotal)
        else:
            query_id = int(query_id)

        if query_id < 0 or query_id >= ntotal:
            raise CommandError(f"--query-id {query_id} out of range [0, {ntotal - 1}]")

        self.stdout.write(f"Index ntotal={ntotal}, dim={dim}, kind={kind}")
        self.stdout.write(f"Query id: {query_id}")

        try:
            query_vec = _reconstruct_vector(index, query_id, faiss, self.stdout)
        except Exception as e:
            raise CommandError(
                f"Failed to reconstruct query vector id {query_id}: {e}"
            ) from e

        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != dim:
            raise CommandError(f"Unexpected query shape {q.shape}, expected (*, {dim})")

        if kind == "ip":
            faiss.normalize_L2(q)

        search_k = min(ntotal, top_k + (1 if exclude_self else 0))
        D, I = index.search(q, search_k)
        scores = D[0]
        ids = I[0]

        mapping = None
        if options.get("with_names"):
            mapping_path = (options.get("mapping_file") or "").strip()
            if not mapping_path:
                mapping_path = _pick_default_mapping_path()
            if not mapping_path:
                raise CommandError("--with-names requested but no mapping file is configured.")
            self.stdout.write(f"Loading mapping file: {mapping_path}")
            mapping = _load_mapping_file(mapping_path)
            self.stdout.write(f"Mapping entries: {len(mapping)}")

            if 0 <= query_id < len(mapping):
                self.stdout.write(f"Query image name: {mapping[query_id]}")

        self.stdout.write("Top results:")
        rank = 0
        for idx, score in zip(ids, scores):
            idx_int = int(idx)
            if idx_int < 0:
                continue
            if exclude_self and idx_int == query_id:
                continue
            rank += 1

            if kind == "l2":
                value_text = f"d2={float(score):.6f}, d={float(np.sqrt(max(score, 0.0))):.6f}"
            else:
                value_text = f"ip={float(score):.6f}"

            line = f"{rank:>2}. id={idx_int}  {value_text}"
            if mapping is not None:
                name = mapping[idx_int] if 0 <= idx_int < len(mapping) else "<out-of-range>"
                line += f"  name={name}"
            self.stdout.write(line)

            if rank >= top_k:
                break

        if rank == 0:
            self.stdout.write(self.style.WARNING("No valid neighbors returned by FAISS search."))
