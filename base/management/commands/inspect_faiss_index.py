import os

from django.core.management.base import BaseCommand, CommandError

from base import config


def _resolve_index_path(path_or_dir):
    """Resolve a file or directory into a concrete .faiss index path."""
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


class Command(BaseCommand):
    help = (
        "Inspect FAISS index structure and check whether original IDs are stored "
        "(IndexIDMap/IndexIDMap2 via id_map)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--index-path",
            default="",
            help="Path to .faiss file (or directory containing one).",
        )
        parser.add_argument(
            "--kind",
            choices=["l2", "ip"],
            default="l2",
            help="If --index-path is not set, use configured L2 or IP index path.",
        )
        parser.add_argument(
            "--sample-size",
            type=int,
            default=10,
            help="How many leading IDs to print from id_map (default: 10).",
        )

    def _pick_default_path(self, kind):
        if kind == "ip":
            return getattr(config, "FAISS_IP_INDEX_PATH", "")
        return getattr(config, "FAISS_L2_INDEX_PATH", "")

    @staticmethod
    def _unwrap_chain(index, faiss_module):
        chain = [type(index).__name__]
        current = index
        while isinstance(current, faiss_module.IndexPreTransform):
            current = faiss_module.downcast_index(current.index)
            chain.append(type(current).__name__)
        return chain, current

    @staticmethod
    def _read_id(id_map_obj, idx):
        # SWIG vector bindings vary by version; try common access methods.
        try:
            return int(id_map_obj.at(idx))
        except Exception:
            pass
        try:
            return int(id_map_obj[idx])
        except Exception:
            pass
        return None

    def handle(self, *args, **options):
        try:
            import faiss
        except Exception as e:
            raise CommandError(f"faiss is not available in this environment: {e}") from e

        requested = (options.get("index_path") or "").strip()
        if not requested:
            requested = (self._pick_default_path(options.get("kind", "l2")) or "").strip()

        resolved = _resolve_index_path(requested)
        if not resolved:
            raise CommandError(f"Could not resolve index path from: {requested}")

        self.stdout.write(f"Loading index: {resolved}")
        try:
            index = faiss.read_index(resolved)
        except Exception as e:
            raise CommandError(f"Failed to load FAISS index: {e}") from e

        chain, core = self._unwrap_chain(index, faiss)
        self.stdout.write(f"Type chain: {' -> '.join(chain)}")
        self.stdout.write(f"Core type: {type(core).__name__}")

        try:
            self.stdout.write(f"ntotal: {int(index.ntotal)}")
        except Exception:
            self.stdout.write("ntotal: <unavailable>")

        try:
            self.stdout.write(f"dimension (d): {int(index.d)}")
        except Exception:
            self.stdout.write("dimension (d): <unavailable>")

        metric_type = getattr(index, "metric_type", None)
        self.stdout.write(f"metric_type: {metric_type if metric_type is not None else '<unavailable>'}")

        has_id_map = hasattr(core, "id_map")
        self.stdout.write(f"has id_map: {has_id_map}")
        if not has_id_map:
            self.stdout.write(
                self.style.WARNING(
                    "Index does not expose id_map. IDs are likely implicit row order (0..ntotal-1)."
                )
            )
            return

        id_map_obj = core.id_map
        size = None
        try:
            size = int(id_map_obj.size())
        except Exception:
            pass

        self.stdout.write(f"id_map size: {size if size is not None else '<unavailable>'}")
        sample_size = max(1, int(options.get("sample_size", 10)))

        if size is None:
            self.stdout.write("Could not sample id_map (size unavailable in this FAISS build).")
            return

        take = min(sample_size, size)
        sample_ids = []
        for i in range(take):
            value = self._read_id(id_map_obj, i)
            if value is None:
                break
            sample_ids.append(value)

        if sample_ids:
            self.stdout.write(f"id_map first {len(sample_ids)} IDs: {sample_ids}")
            sequential_prefix = all(value == i for i, value in enumerate(sample_ids))
            self.stdout.write(f"sequential prefix: {sequential_prefix}")
        else:
            self.stdout.write("Could not read id_map entries with this FAISS Python binding.")
