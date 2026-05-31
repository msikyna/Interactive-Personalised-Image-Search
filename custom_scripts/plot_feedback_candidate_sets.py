#!/usr/bin/env python3
"""
Plot scaling factor across feedback iterations for again_<method> experiments.

Expected input layout:

    experiment_root/
      again_OMDML/
      again_OASIS/
      again_OPML/
      again_SORS/
      again_MLOML/
      again_RobustODML/

Each method folder is traversed recursively. The script reads feedback_log.json
entries produced by the app and uses their performance metadata for the plot.
Each matrix_<n> folder is treated as one hyperparameter configuration, and
the entries inside that folder's feedback_log.json become feedback iterations.
Candidate-set size is annotated at each plotted point.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METHOD_ORDER = [
    "OASIS",
    "OMDML",
    "SORS",
    "OPML",
    "MLOML",
    "RobustODML",
]

PREFERRED_PARAM_BY_METHOD = {
    "OASIS": "C",
    "OMDML": "C",
    "SORS": "eta",
    "OPML": "gamma",
    "MLOML": "gamma",
    "RobustODML": "C",
}

DEFAULT_PARAM_VALUES_FEEDBACK_TYPE_3 = {
    "OASIS": {"C": 1e-3},
    "OMDML": {"C": 4e-3},
    "SORS": {"eta": 1e-3},
    "OPML": {"gamma": 4e-4},
    "MLOML": {"gamma": 0.0004},
    "RobustODML": {"C": 7e-4},
}

CANDIDATE_KEY_FALLBACKS = [
    "num_results_candidate_set",
    "num_results_candidate_set_full_mahalanobis_filter",
    "num_results_candidate_set_progressive_filter",
    "candidate_set_size",
]

SCALING_KEYS = [
    "scaling_factor_after",
    "scaling_factor",
    "scaling_factor_before",
]


@dataclass(frozen=True)
class Record:
    method: str
    source: Path
    source_index: int
    config_hint: str
    params_key: str
    model_params: tuple[tuple[str, Any], ...]
    candidate_set_size: int
    scaling_factor: float | None
    iteration: int | None
    timestamp: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a multi-panel scaling-factor plot from again_<method> "
            "experiment folders."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root folder containing again_<method> folders. Default: current directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="scaling_factor_by_feedback_iteration.png",
        help="Output image path. Default: scaling_factor_by_feedback_iteration.png",
    )
    parser.add_argument(
        "--title",
        default="Scaling factor across feedback iterations",
        help="Figure title.",
    )
    parser.add_argument(
        "--candidate-key",
        default="num_results_candidate_set",
        help=(
            "Candidate-set metadata key to prefer. Falls back to related keys "
            "when this key is absent."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum feedback iterations to plot per configuration. Default: 3.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output DPI. Default: 220.",
    )
    parser.add_argument(
        "--no-point-labels",
        "--no-scaling-labels",
        dest="no_point_labels",
        action="store_true",
        help="Do not annotate points with candidate-set-size labels.",
    )
    return parser.parse_args()


def normalise_method_name(value: str) -> str:
    compact = re.sub(r"[^a-z0-9]", "", value.lower())
    for method in METHOD_ORDER:
        if compact == re.sub(r"[^a-z0-9]", "", method.lower()):
            return method
    return value


def find_method_dirs(root: Path) -> dict[str, Path]:
    method_dirs: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        match = re.match(r"again_(?P<method>.+)$", path.name, flags=re.IGNORECASE)
        if not match:
            continue
        method = normalise_method_name(match.group("method"))
        method_dirs[method] = path
    return method_dirs


def load_json(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Skipping unreadable JSON {path}: {exc}")
        return None


def to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(round(number))


def natural_sort_key(value: Path | str) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def canonical_params(model_params: dict[str, Any] | None, config_hint: str) -> tuple[str, tuple[tuple[str, Any], ...]]:
    if not isinstance(model_params, dict) or not model_params:
        return config_hint, tuple()
    items = tuple(sorted(model_params.items(), key=lambda item: item[0]))
    return json.dumps(items, sort_keys=True, default=str), items


def find_candidate(entry: dict[str, Any], preferred_key: str) -> int | None:
    keys = [preferred_key] + [key for key in CANDIDATE_KEY_FALLBACKS if key != preferred_key]

    for key in keys:
        value = to_int(entry.get(key))
        if value is not None:
            return value

    performance = entry.get("performance_metrics")
    if isinstance(performance, dict):
        for key in keys:
            value = to_int(performance.get(key))
            if value is not None:
                return value

    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            value = to_int(metadata.get(key))
            if value is not None:
                return value

    return None


def find_scaling_factor(entry: dict[str, Any], context: dict[str, Any]) -> float | None:
    for container in (entry, context):
        for key in SCALING_KEYS:
            value = to_float(container.get(key))
            if value is not None:
                return value

    performance = entry.get("performance_metrics")
    if isinstance(performance, dict):
        for key in SCALING_KEYS:
            value = to_float(performance.get(key))
            if value is not None:
                return value

    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        for key in SCALING_KEYS:
            value = to_float(metadata.get(key))
            if value is not None:
                return value

    return None


def find_iteration(entry: dict[str, Any]) -> int | None:
    for key in (
        "feedback_iteration",
        "feedback_iter",
        "iteration",
        "iteration_index",
        "feedback_round",
    ):
        value = to_int(entry.get(key))
        if value is not None:
            return value
    return None


def extract_records_from_json(
    value: Any,
    *,
    method: str,
    source: Path,
    source_index_start: int,
    config_hint: str,
    preferred_candidate_key: str,
    context: dict[str, Any] | None = None,
) -> tuple[list[Record], int]:
    context = dict(context or {})
    records: list[Record] = []
    source_index = source_index_start

    def walk(node: Any, inherited: dict[str, Any], fallback_iteration: int | None = None) -> None:
        nonlocal source_index

        if isinstance(node, list):
            for iteration, child in enumerate(node, start=1):
                walk(child, inherited, iteration)
            return

        if not isinstance(node, dict):
            return

        current = dict(inherited)
        for key in (
            "model",
            "model_params",
            "timestamp",
            "updated_at",
            "scaling_factor",
            "scaling_factor_before",
            "scaling_factor_after",
        ):
            if key in node:
                current[key] = node[key]

        candidate_set_size = find_candidate(node, preferred_candidate_key)
        if candidate_set_size is not None:
            _, model_params = canonical_params(
                current.get("model_params") if isinstance(current.get("model_params"), dict) else None,
                config_hint,
            )
            iteration = find_iteration(node)
            records.append(
                Record(
                    method=normalise_method_name(str(current.get("model") or method)),
                    source=source,
                    source_index=source_index,
                    config_hint=config_hint,
                    params_key=config_hint,
                    model_params=model_params,
                    candidate_set_size=candidate_set_size,
                    scaling_factor=find_scaling_factor(node, current),
                    iteration=iteration if iteration is not None else fallback_iteration,
                    timestamp=str(current.get("timestamp") or current.get("updated_at") or ""),
                )
            )
            source_index += 1
            return

        for child in node.values():
            walk(child, current, fallback_iteration)

    walk(value, context)
    return records, source_index


def config_hint_for_path(method_dir: Path, json_path: Path) -> str:
    try:
        relative_parent = json_path.parent.relative_to(method_dir)
    except ValueError:
        return json_path.parent.name

    if not relative_parent.parts:
        return method_dir.name
    return "/".join(relative_parent.parts)


def collect_records(
    method: str,
    method_dir: Path,
    preferred_candidate_key: str,
) -> list[Record]:
    records: list[Record] = []
    source_index = 0

    for path in sorted(method_dir.rglob("feedback_log.json"), key=natural_sort_key):
        loaded = load_json(path)
        if loaded is None:
            continue
        extracted, source_index = extract_records_from_json(
            loaded,
            method=method,
            source=path,
            source_index_start=source_index,
            config_hint=config_hint_for_path(method_dir, path),
            preferred_candidate_key=preferred_candidate_key,
        )
        records.extend(extracted)

    return records


def group_records(records: list[Record]) -> dict[str, list[Record]]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.params_key].append(record)

    for key, values in groups.items():
        values.sort(key=lambda item: (item.timestamp, str(item.source), item.source_index))
        if all(item.iteration is None for item in values):
            groups[key] = [
                Record(
                    method=item.method,
                    source=item.source,
                    source_index=item.source_index,
                    config_hint=item.config_hint,
                    params_key=item.params_key,
                    model_params=item.model_params,
                    candidate_set_size=item.candidate_set_size,
                    scaling_factor=item.scaling_factor,
                    iteration=index,
                    timestamp=item.timestamp,
                )
                for index, item in enumerate(values, start=1)
            ]

    return groups


def numeric_param_value(params: tuple[tuple[str, Any], ...], name: str | None) -> float | None:
    if not name:
        return None
    for key, value in params:
        if key == name:
            return to_float(value)
    return None


def choose_label_param(method: str, groups: dict[str, list[Record]]) -> str | None:
    preferred = PREFERRED_PARAM_BY_METHOD.get(method)
    param_values: dict[str, set[float]] = defaultdict(set)

    for values in groups.values():
        params = values[0].model_params if values else tuple()
        for key, value in params:
            number = to_float(value)
            if number is not None:
                param_values[key].add(number)

    if preferred in param_values and len(param_values[preferred]) > 1:
        return preferred

    for key in sorted(param_values):
        if len(param_values[key]) > 1:
            return key

    if preferred in param_values:
        return preferred

    if param_values:
        return sorted(param_values)[0]

    return None


def format_number(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return str(value)
    if abs(number) >= 1000 or (0 < abs(number) < 0.01):
        return f"{number:.0e}"
    if float(number).is_integer() and abs(number) >= 1:
        return str(int(number))
    return f"{number:.4g}"


def format_candidate_size(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def infer_config_tag(method: str, param_name: str | None, param_value: float | None, hint: str) -> str:
    hint_lower = hint.lower()
    for token in ("default", "x5", "5x", "x10", "10x", "low", "high"):
        if token in hint_lower:
            if token == "5x":
                return "x5"
            if token == "10x":
                return "x10"
            return token

    if param_name and param_value is not None:
        default = DEFAULT_PARAM_VALUES_FEEDBACK_TYPE_3.get(method, {}).get(param_name)
        default_number = to_float(default)
        if default_number not in (None, 0.0):
            ratio = param_value / default_number
            if math.isclose(ratio, 1.0, rel_tol=0.08, abs_tol=0.08):
                return "default"
            if math.isclose(ratio, 5.0, rel_tol=0.08, abs_tol=0.08):
                return "x5"
            if math.isclose(ratio, 10.0, rel_tol=0.08, abs_tol=0.08):
                return "x10"

    return ""


def make_group_label(method: str, values: list[Record], param_name: str | None) -> tuple[str, str, float | None]:
    first = values[0]
    param_value = numeric_param_value(first.model_params, param_name)
    tag = infer_config_tag(method, param_name, param_value, first.config_hint)

    if param_name and param_value is not None:
        label_core = f"{param_name}={format_number(param_value)}"
    elif first.config_hint:
        label_core = first.config_hint
    else:
        label_core = "configuration"

    if tag:
        return f"{tag} ({label_core})", tag, param_value
    return label_core, tag, param_value


def sort_group_items(
    method: str,
    groups: dict[str, list[Record]],
    param_name: str | None,
) -> list[tuple[str, list[Record], str, float | None]]:
    items = []
    tag_order = {
        "default": 0,
        "low": 1,
        "x5": 1,
        "high": 2,
        "x10": 2,
    }
    for values in groups.values():
        if not values:
            continue
        label, tag, param_value = make_group_label(method, values, param_name)
        order = tag_order.get(tag, 10)
        numeric_order = param_value if param_value is not None else float("inf")
        items.append((label, values, tag, numeric_order, order))

    items.sort(key=lambda item: (item[4], item[3], item[0]))
    return [(label, values, tag, numeric_order) for label, values, tag, numeric_order, _ in items]


def style_for_series(tag: str, index: int) -> dict[str, Any]:
    if tag == "default":
        return {"color": "#2c7fb8", "linestyle": "-", "marker": "s"}
    if tag in {"x5", "low"}:
        return {"color": "#9bd8ef", "linestyle": "--", "marker": "s"}
    if tag in {"x10", "high"}:
        return {"color": "#f03b3b", "linestyle": ":", "marker": "s"}

    fallback = [
        {"color": "#2c7fb8", "linestyle": "-", "marker": "s"},
        {"color": "#9bd8ef", "linestyle": "--", "marker": "s"},
        {"color": "#f03b3b", "linestyle": ":", "marker": "s"},
        {"color": "#31a354", "linestyle": "-.", "marker": "s"},
        {"color": "#756bb1", "linestyle": "-", "marker": "s"},
    ]
    return fallback[index % len(fallback)]


def plot(
    all_records: dict[str, list[Record]],
    *,
    output: Path,
    title: str,
    max_iterations: int | None,
    show_point_labels: bool,
    dpi: int,
) -> None:
    plot_cache_dir = Path(tempfile.gettempdir()) / "simsearch-matplotlib"
    plot_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_names = [method for method in METHOD_ORDER if method in all_records]
    method_names.extend(sorted(method for method in all_records if method not in method_names))
    if not method_names:
        raise SystemExit("No records found in any again_<method> folder.")

    ncols = 3
    nrows = math.ceil(len(method_names) / ncols)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
        }
    )
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.6 * ncols, 4.15 * nrows),
        sharex=False,
        squeeze=False,
    )

    for axis in axes.ravel()[len(method_names):]:
        axis.axis("off")

    for axis, method in zip(axes.ravel(), method_names):
        groups = group_records(all_records[method])
        param_name = choose_label_param(method, groups)
        series_items = sort_group_items(method, groups, param_name)

        axis.set_title(method, fontsize=14)
        axis.set_xlabel("Feedback iteration", fontsize=12)
        axis.set_ylabel("Scaling factor", fontsize=12)
        axis.tick_params(axis="both", labelsize=11)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        if not series_items:
            axis.text(0.5, 0.5, "No metadata found", ha="center", va="center", transform=axis.transAxes)
            continue

        for series_index, (label, values, tag, _) in enumerate(series_items):
            plotted_values = [
                record for record in values
                if record.iteration is not None
                and record.scaling_factor is not None
                and (max_iterations is None or record.iteration <= max_iterations)
            ]
            plotted_values.sort(key=lambda item: item.iteration or 0)
            if not plotted_values:
                continue

            x_values = [int(record.iteration or 0) for record in plotted_values]
            y_values = [float(record.scaling_factor or 0.0) for record in plotted_values]
            style = style_for_series(tag, series_index)
            axis.plot(x_values, y_values, label=label, linewidth=1.4, markersize=4, **style)

            if show_point_labels:
                for x_value, y_value, record in zip(x_values, y_values, plotted_values):
                    axis.annotate(
                        format_candidate_size(record.candidate_set_size),
                        (x_value, y_value),
                        textcoords="offset points",
                        xytext=(4, 5),
                        fontsize=9,
                        color=style["color"],
                    )

        axis.set_xticks(range(1, (max_iterations or 3) + 1))
        axis.margins(x=0.08, y=0.18)
        legend = axis.legend(
            fontsize=9,
            frameon=True,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
            borderpad=0.3,
            handlelength=1.5,
            labelspacing=0.25,
        )
        legend.get_frame().set_alpha(0.95)

    fig.suptitle(title, fontsize=17)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96), h_pad=4.2, w_pad=2.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"Saved {output}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    method_dirs = find_method_dirs(root)
    if not method_dirs:
        raise SystemExit(f"No again_<method> folders found under {root}")

    all_records: dict[str, list[Record]] = {}
    for method, method_dir in method_dirs.items():
        records = collect_records(method, method_dir, args.candidate_key)
        if records:
            all_records[method] = [record for record in records if record.method == method]
            if not all_records[method]:
                all_records[method] = records
        else:
            all_records[method] = []
        print(f"{method}: {len(all_records[method])} points from {method_dir}")

    plot(
        all_records,
        output=Path(args.output).expanduser().resolve(),
        title=args.title,
        max_iterations=args.max_iterations,
        show_point_labels=not args.no_point_labels,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
