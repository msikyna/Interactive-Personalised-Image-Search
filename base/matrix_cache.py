from __future__ import annotations

import pickle
import threading
from collections import OrderedDict
from typing import Any

from . import config


_CACHE_LOCK = threading.Lock()
_MATRIX_CACHE: "OrderedDict[tuple[int, str], Any]" = OrderedDict()


def _max_cache_size() -> int:
    try:
        return max(1, int(getattr(config, 'MATRIX_CACHE_SIZE', 32)))
    except (TypeError, ValueError):
        return 32


def _cache_key(user_matrix_obj) -> tuple[int, str]:
    updated_at = getattr(user_matrix_obj, 'updated_at', None)
    if updated_at is None:
        updated_marker = ''
    else:
        try:
            updated_marker = updated_at.isoformat()
        except Exception:
            updated_marker = str(updated_at)
    return int(user_matrix_obj.pk), updated_marker


def _drop_stale_entries_for_pk(user_matrix_pk: int) -> None:
    stale_keys = [key for key in _MATRIX_CACHE.keys() if key[0] == int(user_matrix_pk)]
    for key in stale_keys:
        _MATRIX_CACHE.pop(key, None)


def _load_matrix_data(user_matrix_obj):
    deferred_fields = set()
    try:
        deferred_fields = set(user_matrix_obj.get_deferred_fields())
    except Exception:
        deferred_fields = set()

    if 'matrix_data' not in deferred_fields:
        return user_matrix_obj.matrix_data

    from .models import UserMetricMatrix

    fresh_obj = UserMetricMatrix.objects.only('matrix_data').get(pk=user_matrix_obj.pk)
    return fresh_obj.matrix_data


def get_cached_matrix_for_user_matrix(user_matrix_obj):
    """
    Return a cached numpy matrix for the given UserMetricMatrix metadata row.
    Cache key is based on (pk, updated_at) so saves naturally invalidate old entries.
    """
    if user_matrix_obj is None:
        return None

    key = _cache_key(user_matrix_obj)
    with _CACHE_LOCK:
        cached = _MATRIX_CACHE.get(key)
        if cached is not None:
            _MATRIX_CACHE.move_to_end(key)
            return cached

    matrix_data = _load_matrix_data(user_matrix_obj)
    if matrix_data:
        matrix = pickle.loads(matrix_data)
    else:
        import numpy as np
        matrix = np.eye(int(getattr(user_matrix_obj, 'matrix_dimension', 768)))

    with _CACHE_LOCK:
        _drop_stale_entries_for_pk(int(user_matrix_obj.pk))
        _MATRIX_CACHE[key] = matrix
        while len(_MATRIX_CACHE) > _max_cache_size():
            _MATRIX_CACHE.popitem(last=False)
    return matrix


def store_cached_matrix_for_user_matrix(user_matrix_obj, matrix) -> None:
    if user_matrix_obj is None or matrix is None:
        return

    key = _cache_key(user_matrix_obj)
    with _CACHE_LOCK:
        _drop_stale_entries_for_pk(int(user_matrix_obj.pk))
        _MATRIX_CACHE[key] = matrix
        _MATRIX_CACHE.move_to_end(key)
        while len(_MATRIX_CACHE) > _max_cache_size():
            _MATRIX_CACHE.popitem(last=False)


def invalidate_cached_matrix(user_matrix_obj_or_pk) -> None:
    if user_matrix_obj_or_pk is None:
        return
    pk = getattr(user_matrix_obj_or_pk, 'pk', user_matrix_obj_or_pk)
    with _CACHE_LOCK:
        _drop_stale_entries_for_pk(int(pk))
