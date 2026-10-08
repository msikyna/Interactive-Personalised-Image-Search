from django.shortcuts import render
from django.http import JsonResponse, FileResponse, Http404, HttpResponse
from django.db import IntegrityError
from functools import lru_cache
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from .services import image_similarity_service
from .clip_service import clip_service
from .image_urls import build_image_url
from django.conf import settings
from . import config
from .matrix_cache import get_cached_matrix_for_user_matrix
try:
    from . import startup_status  # type: ignore
except Exception:
    startup_status = None
import os
import numpy as np
import time
import math
import mimetypes
import re
import tempfile
import zipfile


def _distance_mode_for_metric(distance_metric):
    metric = (distance_metric or '').lower()
    return 'dot_product' if metric in ('cosine', 'dot_product') else 'euclidean'


def _current_dataset_dimension(default=768):
    try:
        dataset = image_similarity_service.dataset
        shape = getattr(dataset, 'shape', None)
        if shape and len(shape) > 1:
            return int(shape[1])
    except Exception:
        pass

    try:
        vector_dim = int(getattr(image_similarity_service, 'vector_dim', 0) or 0)
        if vector_dim > 0:
            return vector_dim
    except Exception:
        pass

    return int(default)


def _should_use_filter_refine(user_matrix_obj, scaling_factor):
    if user_matrix_obj is None:
        return False
    try:
        return float(scaling_factor) > 1.0
    except (TypeError, ValueError):
        return False


def _get_authenticated_search_state(user_id, include_matrix=False):
    """
    Load minimal user + matrix metadata for search paths.
    The matrix payload itself is loaded lazily via in-process cache.
    """
    if not user_id:
        return None, None, None, 1.0

    from .models import User, UserMetricMatrix

    try:
        user = User.objects.only('id', 'username').get(id=user_id)
    except User.DoesNotExist:
        return None, None, None, 1.0

    try:
        user_matrix_obj = UserMetricMatrix.objects.only(
            'id',
            'user_id',
            'matrix_dimension',
            'scaling_factor',
            'updated_at',
            'distance_mode',
            'metric_learning_model',
            'feedback_type',
            'model_hyperparameters',
            'feedback_hyperparameters',
        ).get(user_id=user.id)
    except UserMetricMatrix.DoesNotExist:
        return user, None, None, 1.0

    metric_matrix = get_cached_matrix_for_user_matrix(user_matrix_obj) if include_matrix else None
    try:
        scaling_factor = float(user_matrix_obj.scaling_factor)
    except (TypeError, ValueError):
        scaling_factor = 1.0

    return user, user_matrix_obj, metric_matrix, scaling_factor


def _progressive_filter_refine_params(initial_request=False):
    """
    Return (range_stage, quick_radius_ratio) for filter+refine.
    Quick stage is used when initial_request=True, otherwise full stage.
    """
    quick_enabled = bool(getattr(config, 'PROGRESSIVE_FILTER_REFINE', True))
    stage = 'quick' if (initial_request and quick_enabled) else 'full'
    try:
        ratio = float(getattr(config, 'PROGRESSIVE_QUICK_RANGE_RATIO', 0.35))
    except (TypeError, ValueError):
        ratio = 0.35
    ratio = min(1.0, max(0.0, ratio))
    return stage, ratio


def _to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_feedback_timing_fields(
    results,
    total_flow_time_ms=None,
    feedback_processing_matrix_save_ms=None,
    model_learning_ms=None
):
    results = results or {}
    refined_set_size = int(len(results.get('results', [])))
    candidate_set_size = results.get('num_results_candidate_set')
    if candidate_set_size is None:
        candidate_set_size = refined_set_size
    else:
        try:
            candidate_set_size = int(candidate_set_size)
        except (TypeError, ValueError):
            candidate_set_size = refined_set_size

    clip_encode_ms = _to_float_or_none(results.get('clip_encode_ms'))
    faiss_knn_ms = _to_float_or_none(results.get('faiss_knn_ms'))
    faiss_range_ms = _to_float_or_none(results.get('faiss_range_ms'))
    reconstruct_ms = _to_float_or_none(results.get('reconstruct_ms'))
    refine_ms = _to_float_or_none(results.get('refine_ms'))

    distance_metric = results.get('distance_metric')
    base_metric = results.get('base_metric')
    search_pipeline = results.get('search_pipeline')
    distance_mode = results.get('distance_mode')
    progressive_stage = str(results.get('progressive_stage') or '').strip().lower()
    if not distance_mode:
        metric_lower = str(distance_metric or '').strip().lower()
        base_lower = str(base_metric or '').strip().lower()
        if metric_lower in ('cosine', 'dot_product'):
            distance_mode = 'dot_product'
        elif metric_lower == 'euclidean':
            distance_mode = 'euclidean'
        elif metric_lower == 'mahalanobis':
            if base_lower in ('cosine', 'dot_product'):
                distance_mode = 'dot_product'
            elif base_lower == 'euclidean':
                distance_mode = 'euclidean'

    feedback_processing_matrix_save_ms = _to_float_or_none(feedback_processing_matrix_save_ms)
    model_learning_ms = _to_float_or_none(model_learning_ms)

    # Candidate-set split for progressive vs full stage
    num_results_candidate_set_progressive_filter = _to_int_or_none(
        results.get('num_results_candidate_set_progressive_filter')
    )
    num_results_candidate_set_full_mahalanobis_filter = _to_int_or_none(
        results.get('num_results_candidate_set_full_mahalanobis_filter')
    )
    if num_results_candidate_set_progressive_filter is None and progressive_stage == 'quick':
        num_results_candidate_set_progressive_filter = candidate_set_size
    if num_results_candidate_set_full_mahalanobis_filter is None and progressive_stage == 'full':
        num_results_candidate_set_full_mahalanobis_filter = candidate_set_size

    # Ranges
    range_r_e = _to_float_or_none(results.get('range_r_e'))
    if range_r_e is None:
        range_r_e = _to_float_or_none(results.get('range_e'))

    range_progressive = _to_float_or_none(results.get('range_progressive'))
    if range_progressive is None and progressive_stage == 'quick':
        range_progressive = _to_float_or_none(results.get('range_search'))

    range_r_m_full = _to_float_or_none(results.get('range_r_m_full'))
    if range_r_m_full is None:
        range_r_m_full = _to_float_or_none(results.get('range_m'))

    # Compute total flow deterministically as the sum of known measured stages.
    # Client/browser wall-clock values can drift from server stage timings.
    processing_sum = 0.0
    for value in (
        feedback_processing_matrix_save_ms,
        model_learning_ms,
        clip_encode_ms,
        faiss_knn_ms,
        faiss_range_ms,
        reconstruct_ms,
        refine_ms,
    ):
        if value is not None:
            processing_sum += value

    total_ms = processing_sum if processing_sum > 0.0 else _to_float_or_none(total_flow_time_ms)

    return {
        'feedback_processing_matrix_save_ms': feedback_processing_matrix_save_ms,
        'model_learning_ms': model_learning_ms,
        'total_flow_time_ms': total_ms,
        'num_results_candidate_set': candidate_set_size,
        'num_results_candidate_set_progressive_filter': num_results_candidate_set_progressive_filter,
        'num_results_candidate_set_full_mahalanobis_filter': num_results_candidate_set_full_mahalanobis_filter,
        'num_results_refined_set': refined_set_size,
        'range_r_e': range_r_e,
        'range_progressive': range_progressive,
        'range_r_m_full': range_r_m_full,
        'clip_encode_ms': clip_encode_ms,
        'faiss_knn_ms': faiss_knn_ms,
        'faiss_range_ms': faiss_range_ms,
        'reconstruct_ms': reconstruct_ms,
        'refine_ms': refine_ms,
        'distance_mode': distance_mode,
        'distance_metric': distance_metric,
        'base_metric': base_metric,
        'search_pipeline': search_pipeline,
    }


def _normalize_query_text(text):
    return ' '.join(str(text or '').strip().lower().split())


def _build_query_signature(request, text_query='', image_index='', upload_search=''):
    """
    Build a stable signature for "same vs different query" detection.
    """
    if text_query:
        return {
            'type': 'text',
            'value': _normalize_query_text(text_query),
        }

    if image_index not in (None, ''):
        raw = str(image_index).strip()
        if raw:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                value = raw
            return {
                'type': 'image',
                'value': value,
            }

    if upload_search:
        upload_token = request.session.get('uploaded_query_token')
        if not upload_token:
            upload_token = request.session.get('uploaded_image_name', 'uploaded_image')
        return {
            'type': 'uploaded_image',
            'value': str(upload_token),
        }

    return None


def _remember_query_signature(request, query_signature):
    if query_signature is None:
        return
    request.session['last_query_signature'] = query_signature
    request.session.modified = True


def _remember_last_logged_query_ref(request, query_log_ref):
    """Store file+index reference for the latest logged query entry."""
    if not isinstance(query_log_ref, dict):
        return
    queries_file = query_log_ref.get('queries_file')
    query_index = query_log_ref.get('query_index')
    if not queries_file or query_index is None:
        return
    request.session['last_logged_query_ref'] = {
        'queries_file': str(queries_file),
        'query_index': int(query_index),
    }
    request.session.modified = True


def _mark_last_query_results_delivered(request):
    """
    Mark the moment when the user received results for the latest logged query.
    This timestamp is used to compute query_ranking_time_ms on next search/rerun.
    """
    if 'last_logged_query_ref' not in request.session:
        return
    request.session['last_query_results_delivered_at_ms'] = round(time.time() * 1000.0, 3)
    request.session.modified = True


def _extract_client_query_ranking_time_ms(request):
    """
    Read client-measured ranking time from request, if provided.
    This is browser-side think time (results shown -> next action trigger),
    so it excludes server processing and network transit.
    """
    raw_value = None
    if request.method == 'POST':
        raw_value = request.POST.get('query_ranking_time_ms')
        if raw_value is None:
            raw_value = request.GET.get('query_ranking_time_ms')
    else:
        raw_value = request.GET.get('query_ranking_time_ms')
        if raw_value is None:
            raw_value = request.POST.get('query_ranking_time_ms')

    if raw_value in (None, ''):
        return None

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value) or value < 0.0:
        return None

    # Guardrail against corrupt values.
    max_reasonable_ms = 24.0 * 60.0 * 60.0 * 1000.0
    return min(value, max_reasonable_ms)


def _finalize_previous_query_ranking_time(request):
    """
    Finalize ranking time for the previously delivered query.
    Called when the user triggers another query/rerun.
    """
    log_ref = request.session.get('last_logged_query_ref')
    delivered_at_ms = request.session.get('last_query_results_delivered_at_ms')
    if not isinstance(log_ref, dict) or delivered_at_ms is None:
        return None

    queries_file = log_ref.get('queries_file')
    query_index = log_ref.get('query_index')
    if not queries_file or query_index is None:
        return None

    client_ranking_time_ms = _extract_client_query_ranking_time_ms(request)
    if client_ranking_time_ms is not None:
        ranking_time_ms = float(client_ranking_time_ms)
    else:
        try:
            ranking_time_ms = max(0.0, (time.time() * 1000.0) - float(delivered_at_ms))
        except (TypeError, ValueError):
            return None

    try:
        from .feedback import QueryLogger
        updated = QueryLogger.update_query_ranking_time(
            queries_file=queries_file,
            query_index=query_index,
            query_ranking_time_ms=ranking_time_ms
        )
    except Exception as e:
        print(f"Warning: Could not finalize previous query ranking time: {e}")
        updated = False

    if updated:
        request.session.pop('last_logged_query_ref', None)
        request.session.pop('last_query_results_delivered_at_ms', None)
        request.session.modified = True
        return round(ranking_time_ms, 2)

    return None


def _build_results_for_query_logging(results, fetch_more_results_fn=None):
    """
    Ensure query logging can store up to configured number of results.
    If current result list is shorter, optionally rerun the same search for logging only.
    """
    if not isinstance(results, dict):
        return results

    current_results = results.get('results')
    if not isinstance(current_results, list):
        return results

    try:
        target_count = max(1, int(getattr(config, 'QUERY_LOG_RESULTS_COUNT', 100)))
    except (TypeError, ValueError):
        target_count = 100

    if len(current_results) >= target_count or fetch_more_results_fn is None:
        return results

    # Optional latency optimization: avoid a second in-request search used only for logging expansion.
    if not bool(getattr(config, 'QUERY_LOG_EXPAND_RESULTS', True)):
        return results

    try:
        expanded = fetch_more_results_fn(target_count)
        if isinstance(expanded, dict):
            expanded_results = expanded.get('results')
            if isinstance(expanded_results, list) and expanded_results:
                merged = dict(results)
                merged['results'] = expanded_results[:target_count]
                return merged
    except Exception as e:
        print(f"Warning: Could not expand query results for logging: {e}")

    return results


def _annotate_result_rank_changes(results, previous_indices):
    """
    Annotate current results with rank movement relative to a previous result list.
    """
    if not isinstance(results, dict):
        return results

    current_results = results.get('results')
    if not isinstance(current_results, list):
        return results

    previous_positions = {}
    for rank, index in enumerate(previous_indices or [], start=1):
        try:
            idx_int = int(index)
        except (TypeError, ValueError):
            continue
        if idx_int not in previous_positions:
            previous_positions[idx_int] = rank

    annotated = []
    for current_rank, result in enumerate(current_results, start=1):
        item = dict(result)
        try:
            idx_int = int(item.get('index'))
        except (TypeError, ValueError):
            idx_int = None

        previous_rank = previous_positions.get(idx_int) if idx_int is not None else None
        if previous_rank is None:
            rank_change = None
            rank_change_abs = None
            rank_change_direction = 'new'
        else:
            rank_change = int(previous_rank - current_rank)
            rank_change_abs = abs(rank_change)
            if rank_change > 0:
                rank_change_direction = 'improved'
            elif rank_change < 0:
                rank_change_direction = 'worsened'
            else:
                rank_change_direction = 'same'

        item['current_rank'] = int(current_rank)
        item['previous_rank'] = int(previous_rank) if previous_rank is not None else None
        item['rank_change'] = rank_change
        item['rank_change_abs'] = rank_change_abs
        item['rank_change_direction'] = rank_change_direction
        annotated.append(item)

    merged = dict(results)
    merged['results'] = annotated
    merged['rank_comparison_available'] = bool(previous_positions)
    return merged


def _maybe_reset_matrix_on_query_change(request, query_signature, previous_session_id=None):
    """
    Optionally auto-reset personalized matrix when the user switches query.
    Controlled by config.RESET_MATRIX_ON_QUERY_CHANGE.
    """
    if not bool(getattr(config, 'RESET_MATRIX_ON_QUERY_CHANGE', False)):
        return False
    if query_signature is None:
        return False

    previous_signature = request.session.get('last_query_signature')
    if previous_signature is None or previous_signature == query_signature:
        return False

    user_id = request.session.get('user_id')
    if not user_id or not image_similarity_service.dataset_loaded:
        return False

    from .models import User, FeedbackItem
    from .feedback import MatrixManager

    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return False

    dataset_dim = _current_dataset_dimension()
    MatrixManager.reset_user_matrix(user, dataset_dim)

    # Drop pending linkage/feedback for previous query session.
    request.session.pop('pending_feedback_search_id', None)
    if previous_session_id:
        request.session.pop(f'session_results_{previous_session_id}', None)
        FeedbackItem.objects.filter(user=user, session_id=previous_session_id).delete()

    print("[INFO] Query changed: personalized matrix reset automatically.")
    return True


def home(request):
    """Main page for image similarity search."""
    import uuid

    dataset_loaded = image_similarity_service.dataset_loaded

    text_query = request.GET.get('text_query', '')
    image_index = request.GET.get('image_index', '')
    upload_search = request.GET.get('upload_search', '')
    distance_metric = request.GET.get('metric', 'cosine')
    num_results = int(request.GET.get('num_results', 20))

    previous_session_id = request.session.get('feedback_session_id')
    user_id = request.session.get('user_id')

    # Only apply feedback if we have a new query (text, image, or upload) and there's a previous session
    is_new_query = bool(text_query or image_index or upload_search)
    current_query_signature = _build_query_signature(
        request,
        text_query=text_query,
        image_index=image_index,
        upload_search=upload_search,
    )

    import time
    flow_start_time = time.perf_counter()
    feedback_processing_matrix_save_ms = 0.0
    model_learning_ms = 0.0
    applied_feedback_search_id = None
    query_change_reset_triggered = False

    if is_new_query:
        query_change_reset_triggered = _maybe_reset_matrix_on_query_change(
            request,
            current_query_signature,
            previous_session_id=previous_session_id,
        )
        _finalize_previous_query_ranking_time(request)

    if (
        is_new_query and
        not query_change_reset_triggered and
        previous_session_id and
        user_id and
        dataset_loaded
    ):
        from .models import User
        from .feedback import FeedbackManager, MatrixManager
        try:
            user = User.objects.get(id=user_id)
            # Check if there's any feedback in the previous session
            feedback_items = FeedbackManager.get_session_feedback(user, previous_session_id)
            if feedback_items:
                print(f"\n[INFO] APPLYING FEEDBACK: Found {len(feedback_items)} feedback items from previous session")
                print(f"   Previous session: {previous_session_id}")
                print(f"   User: {user.username}")

                # Get session results for implicit feedback (when user only marks positives or only negatives)
                session_results_key = f'session_results_{previous_session_id}'
                session_results = None
                if session_results_key in request.session:
                    session_results = {previous_session_id: request.session[session_results_key]}
                    print(f"   Session results available: {len(session_results[previous_session_id])} images for implicit feedback")
                else:
                    # Fallback: use current_results if available
                    current_results = request.session.get('current_results', [])
                    if current_results:
                        session_results = {previous_session_id: current_results}
                        print(f"   Using current_results fallback: {len(current_results)} images for implicit feedback")

                # Apply feedback to update the metric matrix
                # Note: total_flow_time_ms and result counts will be updated via
                # a separate log_feedback_timing call after the search completes
                (
                    _,
                    _,
                    feedback_processing_matrix_save_ms,
                    model_learning_ms,
                    _,
                    applied_feedback_search_id
                ) = MatrixManager.update_metric_matrix(
                    user=user,
                    dataset=image_similarity_service.dataset,
                    session_id=previous_session_id,
                    session_results=session_results,
                    image_names=image_similarity_service.image_names
                )
                print(f"[OK] FEEDBACK APPLIED: Matrix updated for user '{user.username}'")
            else:
                print(f"\n[INFO] No feedback to apply from previous session {previous_session_id}")
        except Exception as e:
            import traceback
            print("[ERROR] Error applying feedback:", e)
            traceback.print_exc()

    # Generate NEW session ID for each query (feedback is exclusive per query as per zadanie.txt)
    existing_session_id = request.session.get('feedback_session_id')
    if is_new_query or not existing_session_id:
        session_id = str(uuid.uuid4())
        request.session['feedback_session_id'] = session_id
    else:
        session_id = existing_session_id
    force_sidebar_collapsed = request.session.pop('force_sidebar_collapsed', False)

    context = {
        'dataset_loaded': dataset_loaded,
        'text_query': text_query,
        'distance_metric': distance_metric,
        'num_results': num_results,
        'session_id': session_id,
        'user_logged_in': 'user_id' in request.session,
        'force_sidebar_collapsed': force_sidebar_collapsed,
    }
    startup_snapshot = startup_status.get_snapshot() if startup_status is not None else {}
    context['startup_phase'] = startup_snapshot.get('phase', 'idle')
    context['startup_error'] = startup_snapshot.get('error')
    context['startup_logs'] = startup_snapshot.get('logs', [])[-80:]
    pending_feedback_search_id = request.session.get('pending_feedback_search_id')
    query_feedback_search_id = applied_feedback_search_id or pending_feedback_search_id
    latest_query_log_ref = None

    if dataset_loaded and upload_search and 'uploaded_image_vector' in request.session:
        # Handle uploaded image search from session
        try:
            image_vector = np.array(request.session['uploaded_image_vector'])

            # Get user's metric matrix if logged in
            metric_matrix = None
            scaling_factor = 1.0
            distance_mode = _distance_mode_for_metric(distance_metric)
            use_filter_refine = False
            user_matrix_obj = None
            user = None
            user_id = request.session.get('user_id')

            if user_id:
                user, user_matrix_obj, metric_matrix, scaling_factor = _get_authenticated_search_state(
                    user_id,
                    include_matrix=True
                )
                if user_matrix_obj is not None and metric_matrix is not None:
                    is_identity = np.allclose(metric_matrix, np.eye(metric_matrix.shape[0]))

                    print(f"\n[SEARCH] Uploaded image: loading matrix for user '{user.username}'")
                    print(f"   Matrix is identity: {is_identity}")
                    print(f"   Matrix diagonal (first 5): {np.diag(metric_matrix)[:5]}")
                    print(f"   Scaling factor: {scaling_factor}")
                    print(f"   Distance mode: {distance_mode}")

                    if is_identity:
                        print(f"   Matrix is identity")
                else:
                    print(f"\n[WARN] No matrix found for user {user_id}, using default")

            if _should_use_filter_refine(user_matrix_obj, scaling_factor) and metric_matrix is not None:
                use_filter_refine = True
                distance_mode = _distance_mode_for_metric(distance_metric)
                print(f"   Using Filter-and-Refine ({distance_mode})")

            # Perform search
            if use_filter_refine and metric_matrix is not None:
                range_stage, quick_radius_ratio = _progressive_filter_refine_params(initial_request=True)
                results = image_similarity_service.search_with_filter_refine(
                    query_vector=image_vector,
                    metric_matrix=metric_matrix,
                    scaling_factor=scaling_factor,
                    num_results=num_results,
                    growth_factor=1.0,
                    distance_mode=distance_mode,
                    range_stage=range_stage,
                    quick_radius_ratio=quick_radius_ratio
                )
            else:
                results = image_similarity_service.search_by_vector(
                    query_vector=image_vector,
                    num_results=num_results,
                    distance_metric=distance_metric,
                    metric_matrix=metric_matrix
                )

            results['query_type'] = 'uploaded_image'
            results['uploaded_image_name'] = request.session.get('uploaded_image_name', 'uploaded_image')
            results['uploaded_image_data'] = request.session.get('uploaded_image_data', '')

            context['results'] = results
            context['total_images'] = len(image_similarity_service.image_names)

            # Store query vector and results for feedback
            request.session['current_query_vector'] = image_vector.tolist()
            request.session['current_query_type'] = 'uploaded_image'
            request.session['current_results'] = [r['index'] for r in results.get('results', [])]

            # Log query if user is logged in
            if user_id:
                from .feedback import QueryLogger
                try:
                    if use_filter_refine and metric_matrix is not None:
                        log_stage, log_quick_ratio = _progressive_filter_refine_params(initial_request=True)
                        fetch_more_results_fn = lambda k: image_similarity_service.search_with_filter_refine(
                            query_vector=image_vector,
                            metric_matrix=metric_matrix,
                            scaling_factor=scaling_factor,
                            num_results=k,
                            growth_factor=1.0,
                            distance_mode=distance_mode,
                            range_stage=log_stage,
                            quick_radius_ratio=log_quick_ratio
                        )
                    else:
                        fetch_more_results_fn = lambda k: image_similarity_service.search_by_vector(
                            query_vector=image_vector,
                            num_results=k,
                            distance_metric=distance_metric,
                            metric_matrix=metric_matrix
                        )

                    results_for_logging = _build_results_for_query_logging(
                        results,
                        fetch_more_results_fn=fetch_more_results_fn
                    )
                    latest_query_log_ref = QueryLogger.log_query(
                        user=user,
                        user_matrix_obj=user_matrix_obj,
                        query_info={
                            'type': 'uploaded_image',
                            'text': '',
                            'image_index': None,
                            'session_id': session_id,
                            'feedback_search_id': query_feedback_search_id,
                        },
                        results=results_for_logging
                    )
                except Exception as e:
                    print(f"Warning: Could not log query: {e}")

            # Don't clear uploaded image data yet - keep for feedback
        except Exception as e:
            context['error'] = str(e)
            import traceback
            traceback.print_exc()
    elif dataset_loaded and text_query:
        try:

            # Get user's metric matrix if logged in
            metric_matrix = None
            scaling_factor = 1.0
            distance_mode = _distance_mode_for_metric(distance_metric)
            use_filter_refine = False
            user_matrix_obj = None
            user = None
            user_id = request.session.get('user_id')

            if user_id:
                user, user_matrix_obj, metric_matrix, scaling_factor = _get_authenticated_search_state(
                    user_id,
                    include_matrix=True
                )
                if user_matrix_obj is not None and metric_matrix is not None:
                    is_identity = np.allclose(metric_matrix, np.eye(metric_matrix.shape[0]))

                    print(f"\n[SEARCH] Text query: loading matrix for user '{user.username}'")
                    print(f"   Matrix is identity: {is_identity}")
                    print(f"   Scaling factor: {scaling_factor}")
                    print(f"   Distance mode: {distance_mode}")

                    if is_identity:
                        print(f"   Matrix is identity")

            if _should_use_filter_refine(user_matrix_obj, scaling_factor) and metric_matrix is not None:
                use_filter_refine = True
                distance_mode = _distance_mode_for_metric(distance_metric)
                print(f"   Using Filter-and-Refine ({distance_mode})")

            if use_filter_refine and metric_matrix is not None:
                normalize_text_query = (distance_mode == 'dot_product')
            else:
                normalize_text_query = (distance_metric != 'euclidean')

            clip_encode_start = time.perf_counter()
            text_vector = clip_service.text_to_vector(
                text_query,
                normalize=normalize_text_query
            )
            clip_encode_ms = (time.perf_counter() - clip_encode_start) * 1000

            # Perform search
            if use_filter_refine and metric_matrix is not None:
                range_stage, quick_radius_ratio = _progressive_filter_refine_params(initial_request=True)
                results = image_similarity_service.search_with_filter_refine(
                    query_vector=text_vector,
                    metric_matrix=metric_matrix,
                    scaling_factor=scaling_factor,
                    num_results=num_results,
                    growth_factor=1.0,
                    distance_mode=distance_mode,
                    range_stage=range_stage,
                    quick_radius_ratio=quick_radius_ratio
                )
            else:
                results = image_similarity_service.search_by_vector(
                    query_vector=text_vector,
                    num_results=num_results,
                    distance_metric=distance_metric,
                    metric_matrix=metric_matrix
                )

            results['query_text'] = text_query
            results['query_type'] = 'text'
            results['clip_encode_ms'] = round(clip_encode_ms, 3)
            context['results'] = results
            context['total_images'] = len(image_similarity_service.image_names)

            # Store query vector and info for feedback
            request.session['current_query_vector'] = text_vector.tolist()
            request.session['current_query_type'] = 'text'
            request.session['current_query_text'] = text_query
            request.session['current_results'] = [r['index'] for r in results.get('results', [])]

            # Log query if user is logged in
            if user_id:
                from .feedback import QueryLogger
                try:
                    if use_filter_refine and metric_matrix is not None:
                        log_stage, log_quick_ratio = _progressive_filter_refine_params(initial_request=True)
                        fetch_more_results_fn = lambda k: image_similarity_service.search_with_filter_refine(
                            query_vector=text_vector,
                            metric_matrix=metric_matrix,
                            scaling_factor=scaling_factor,
                            num_results=k,
                            growth_factor=1.0,
                            distance_mode=distance_mode,
                            range_stage=log_stage,
                            quick_radius_ratio=log_quick_ratio
                        )
                    else:
                        fetch_more_results_fn = lambda k: image_similarity_service.search_by_vector(
                            query_vector=text_vector,
                            num_results=k,
                            distance_metric=distance_metric,
                            metric_matrix=metric_matrix
                        )

                    results_for_logging = _build_results_for_query_logging(
                        results,
                        fetch_more_results_fn=fetch_more_results_fn
                    )
                    latest_query_log_ref = QueryLogger.log_query(
                        user=user,
                        user_matrix_obj=user_matrix_obj,
                        query_info={
                            'type': 'text',
                            'text': text_query,
                            'image_index': None,
                            'session_id': session_id,
                            'feedback_search_id': query_feedback_search_id,
                        },
                        results=results_for_logging
                    )
                except Exception as e:
                    print(f"Warning: Could not log query: {e}")
        except Exception as e:
            context['error'] = str(e)
            import traceback
            traceback.print_exc()
    elif dataset_loaded and image_index:
        try:
            idx = int(image_index)

            # Get user's metric matrix if logged in
            metric_matrix = None
            scaling_factor = 1.0
            distance_mode = _distance_mode_for_metric(distance_metric)
            use_filter_refine = False
            user_matrix_obj = None
            user = None

            if 'user_id' in request.session:
                user, user_matrix_obj, metric_matrix, scaling_factor = _get_authenticated_search_state(
                    request.session.get('user_id'),
                    include_matrix=True
                )
                if user_matrix_obj is not None and metric_matrix is not None:
                    is_identity = np.allclose(metric_matrix, np.eye(metric_matrix.shape[0]))

                    print(f"\n[SEARCH] Image index {idx}: loading matrix for user '{user.username}'")
                    print(f"   Matrix is identity: {is_identity}")
                    print(f"   Matrix diagonal (first 5): {np.diag(metric_matrix)[:5]}")
                    print(f"   Scaling factor: {scaling_factor}")
                    print(f"   Distance mode: {distance_mode}")

                    if is_identity:
                        print(f"   Matrix is identity")
                else:
                    print(f"\n[WARN] No matrix found for user, using default")

            if _should_use_filter_refine(user_matrix_obj, scaling_factor) and metric_matrix is not None:
                use_filter_refine = True
                distance_mode = _distance_mode_for_metric(distance_metric)
                print(f"   Using Filter-and-Refine ({distance_mode})")

            if use_filter_refine:
                anchor_metric = 'cosine' if distance_mode == 'dot_product' else 'euclidean'
            else:
                anchor_metric = distance_metric
            anchor_name, anchor_vector = image_similarity_service.get_image_by_index(
                idx, distance_metric=anchor_metric
            )

            # Perform search
            if use_filter_refine and metric_matrix is not None:
                range_stage, quick_radius_ratio = _progressive_filter_refine_params(initial_request=True)
                results = image_similarity_service.search_with_filter_refine(
                    query_vector=anchor_vector,
                    metric_matrix=metric_matrix,
                    scaling_factor=scaling_factor,
                    num_results=num_results,
                    growth_factor=1.0,
                    distance_mode=distance_mode,
                    range_stage=range_stage,
                    quick_radius_ratio=quick_radius_ratio
                )
                results['anchor_index'] = idx
                results['anchor_name'] = anchor_name
            else:
                results = image_similarity_service.search_similar_images(
                    anchor_index=idx,
                    num_results=num_results,
                    distance_metric=distance_metric,
                    metric_matrix=metric_matrix
                )

            results['query_type'] = 'image'
            context['results'] = results
            context['total_images'] = len(image_similarity_service.image_names)

            # Store query vector and info for feedback
            request.session['current_query_vector'] = anchor_vector.tolist()
            request.session['current_query_type'] = 'image'
            request.session['current_query_image_index'] = idx
            request.session['current_results'] = [r['index'] for r in results.get('results', [])]

            # Log query if user is logged in
            if 'user_id' in request.session and user_matrix_obj:
                from .feedback import QueryLogger
                try:
                    if use_filter_refine and metric_matrix is not None:
                        log_stage, log_quick_ratio = _progressive_filter_refine_params(initial_request=True)
                        fetch_more_results_fn = lambda k: image_similarity_service.search_with_filter_refine(
                            query_vector=anchor_vector,
                            metric_matrix=metric_matrix,
                            scaling_factor=scaling_factor,
                            num_results=k,
                            growth_factor=1.0,
                            distance_mode=distance_mode,
                            range_stage=log_stage,
                            quick_radius_ratio=log_quick_ratio
                        )
                    else:
                        fetch_more_results_fn = lambda k: image_similarity_service.search_similar_images(
                            anchor_index=idx,
                            num_results=k,
                            distance_metric=distance_metric,
                            metric_matrix=metric_matrix
                        )

                    results_for_logging = _build_results_for_query_logging(
                        results,
                        fetch_more_results_fn=fetch_more_results_fn
                    )
                    latest_query_log_ref = QueryLogger.log_query(
                        user=user,
                        user_matrix_obj=user_matrix_obj,
                        query_info={
                            'type': 'image',
                            'text': '',
                            'image_index': idx,
                            'session_id': session_id,
                            'feedback_search_id': query_feedback_search_id,
                        },
                        results=results_for_logging
                    )
                except Exception as e:
                    print(f"Warning: Could not log query: {e}")
        except Exception as e:
            context['error'] = str(e)
            import traceback
            traceback.print_exc()
    elif dataset_loaded:
        # Show 20 random images on page load
        context['total_images'] = len(image_similarity_service.image_names)
        context['random_images'] = image_similarity_service.get_random_images(20)

    # Update feedback log with total flow time and result counts (if feedback was applied)
    if is_new_query and (feedback_processing_matrix_save_ms > 0 or model_learning_ms > 0) and 'results' in context:
        timing_fields = _extract_feedback_timing_fields(
            context['results'],
            total_flow_time_ms=(time.perf_counter() - flow_start_time) * 1000,
            feedback_processing_matrix_save_ms=feedback_processing_matrix_save_ms,
            model_learning_ms=model_learning_ms
        )
        try:
            from .models import User, UserMetricMatrix
            from .feedback import QueryLogger
            user = User.objects.get(id=request.session.get('user_id'))
            user_matrix_obj = UserMetricMatrix.objects.get(user=user)
            QueryLogger.update_feedback_log_timing(
                user=user,
                user_matrix_obj=user_matrix_obj,
                total_flow_time_ms=timing_fields['total_flow_time_ms'],
                feedback_processing_matrix_save_ms=timing_fields['feedback_processing_matrix_save_ms'],
                model_learning_ms=timing_fields['model_learning_ms'],
                feedback_search_id=applied_feedback_search_id,
                num_results_requested=num_results,
                num_results_candidate_set=timing_fields['num_results_candidate_set'],
                num_results_candidate_set_progressive_filter=timing_fields['num_results_candidate_set_progressive_filter'],
                num_results_candidate_set_full_mahalanobis_filter=timing_fields['num_results_candidate_set_full_mahalanobis_filter'],
                num_results_refined_set=timing_fields['num_results_refined_set'],
                range_r_e=timing_fields['range_r_e'],
                range_progressive=timing_fields['range_progressive'],
                range_r_m_full=timing_fields['range_r_m_full'],
                clip_encode_ms=timing_fields['clip_encode_ms'],
                faiss_knn_ms=timing_fields['faiss_knn_ms'],
                faiss_range_ms=timing_fields['faiss_range_ms'],
                reconstruct_ms=timing_fields['reconstruct_ms'],
                refine_ms=timing_fields['refine_ms'],
                distance_mode=timing_fields['distance_mode'],
                distance_metric=timing_fields['distance_metric'],
                base_metric=timing_fields['base_metric'],
                search_pipeline=timing_fields['search_pipeline'],
            )
            print(f"⏱️ Total flow time: {timing_fields['total_flow_time_ms']:.1f} ms "
                  f"(feedback+save: {feedback_processing_matrix_save_ms:.1f} ms, "
                  f"model learning: {model_learning_ms:.1f} ms, "
                  f"candidate/refined: {timing_fields['num_results_candidate_set']}/{timing_fields['num_results_refined_set']}, requested: {num_results})")
        except Exception as e:
            print(f"⚠️ Could not update feedback log timing: {e}")

    if is_new_query and 'results' in context and query_feedback_search_id and pending_feedback_search_id:
        request.session.pop('pending_feedback_search_id', None)

    if is_new_query and 'results' in context:
        if latest_query_log_ref:
            _remember_last_logged_query_ref(request, latest_query_log_ref)
            _mark_last_query_results_delivered(request)
        _remember_query_signature(request, current_query_signature)

    # Get user's current feedback for this session if logged in
    if 'user_id' in request.session and 'results' in context:
        try:
            from .models import User
            from .feedback import FeedbackManager
            user = User.objects.get(id=request.session['user_id'])
            feedback_items = FeedbackManager.get_session_feedback(user, session_id)

            # Create a dictionary of feedback by image index
            feedback_map = {}
            for item in feedback_items:
                feedback_map[item.result_image_index] = item.feedback_type

            context['feedback_map'] = feedback_map
        except:
            pass

    return render(request, 'home.html', context)


def startup_status_api(request):
    """API endpoint with startup/warmup status and recent log lines."""
    snapshot = startup_status.get_snapshot() if startup_status is not None else {}
    dataset_loaded = bool(image_similarity_service.dataset_loaded)
    clip_loaded = bool(clip_service.model_loaded)
    clip_required = bool(getattr(config, 'AUTO_LOAD_CLIP_MODEL', True))
    ready = dataset_loaded and (clip_loaded or not clip_required)

    return JsonResponse({
        'success': True,
        'phase': snapshot.get('phase', 'idle'),
        'error': snapshot.get('error'),
        'started_at': snapshot.get('started_at'),
        'updated_at': snapshot.get('updated_at'),
        'logs': snapshot.get('logs', []),
        'dataset_loaded': dataset_loaded,
        'clip_model_loaded': clip_loaded,
        'ready': ready,
    })


def load_dataset_view(request):
    """API endpoint to load the dataset."""
    if request.method == 'POST':
        directory = request.POST.get('directory', config.DATASET_DIRECTORY)
        files_list = request.POST.getlist('files[]') or config.DATASET_FILES

        try:
            image_names, dataset = image_similarity_service.load_dataset_from_files(directory, files_list)

            dataset_dim = len(dataset[0]) if len(dataset) > 0 else 768
            print(f"Dataset dimension: {dataset_dim}")
            print(f"Dataset loaded with {len(image_names)} images")

            return JsonResponse({
                'success': True,
                'num_images': len(image_names),
                'num_dimensions': dataset_dim
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)

    return JsonResponse({'error': 'POST required'}, status=405)


MAHALANOBIS_API_RESULT_COUNT = 100
MAHALANOBIS_MATRIX_SYMMETRY_TOLERANCE = 1e-5
MAHALANOBIS_MATRIX_PSD_TOLERANCE = 1e-6


def _parse_mahalanobis_api_payload(request):
    """Parse and validate the JSON body for the standalone search API."""
    import json

    try:
        payload = json.loads(request.body.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('Request body must be valid UTF-8 JSON.') from exc

    if not isinstance(payload, dict):
        raise ValueError('Request body must be a JSON object.')

    raw_text_query = payload.get('query_text', payload.get('query'))
    raw_image_index = payload.get('query_image_index', payload.get('image_index'))
    has_text_query = isinstance(raw_text_query, str) and bool(raw_text_query.strip())
    has_image_query = raw_image_index not in (None, '')
    if has_text_query == has_image_query:
        raise ValueError('Provide exactly one of query_text or query_image_index.')

    raw_distance_metric = payload.get('distance_metric', payload.get('base_metric'))
    if not isinstance(raw_distance_metric, str) or not raw_distance_metric.strip():
        raise ValueError("distance_metric is required ('euclidean' or 'cosine').")
    base_metric = raw_distance_metric.strip().lower()
    if base_metric in {'cosine', 'inner_product', 'dot_product'}:
        base_metric = 'cosine'
    elif base_metric != 'euclidean':
        raise ValueError(
            "distance_metric must be 'euclidean', 'cosine', 'inner_product', or 'dot_product'."
        )

    raw_matrix = payload.get('metric_matrix', payload.get('matrix'))
    if raw_matrix is None:
        raise ValueError('metric_matrix is required.')

    try:
        metric_matrix = np.asarray(raw_matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('metric_matrix must be a rectangular array of numbers.') from exc

    expected_dimension = int(image_similarity_service.vector_dim)
    expected_shape = (expected_dimension, expected_dimension)
    if metric_matrix.ndim != 2 or metric_matrix.shape != expected_shape:
        raise ValueError(
            f'metric_matrix must have shape {expected_shape}; received {metric_matrix.shape}.'
        )
    if not np.all(np.isfinite(metric_matrix)):
        raise ValueError('metric_matrix may contain only finite numbers.')
    if not np.allclose(
        metric_matrix,
        metric_matrix.T,
        rtol=MAHALANOBIS_MATRIX_SYMMETRY_TOLERANCE,
        atol=MAHALANOBIS_MATRIX_SYMMETRY_TOLERANCE,
    ):
        raise ValueError('metric_matrix must be symmetric.')

    # Work with an exactly symmetric matrix after accepting normal serialization noise.
    metric_matrix = (metric_matrix + metric_matrix.T) / 2.0
    try:
        minimum_eigenvalue = float(np.linalg.eigvalsh(metric_matrix)[0])
    except np.linalg.LinAlgError as exc:
        raise ValueError('metric_matrix eigenvalue calculation did not converge.') from exc
    if minimum_eigenvalue < -MAHALANOBIS_MATRIX_PSD_TOLERANCE:
        raise ValueError(
            'metric_matrix must be positive semidefinite; '
            f'minimum eigenvalue is {minimum_eigenvalue:.6g}.'
        )
    if minimum_eigenvalue < 0.0:
        # Remove a tiny negative eigenvalue caused by floating-point serialization.
        metric_matrix += np.eye(expected_dimension, dtype=np.float64) * (-minimum_eigenvalue)
        minimum_eigenvalue = 0.0

    query = {
        'type': 'text' if has_text_query else 'image',
    }
    if has_text_query:
        query['text'] = raw_text_query.strip()
    else:
        if isinstance(raw_image_index, bool):
            raise ValueError('query_image_index must be an integer.')
        try:
            query['image_index'] = int(raw_image_index)
        except (TypeError, ValueError) as exc:
            raise ValueError('query_image_index must be an integer.') from exc
        if isinstance(raw_image_index, float) and not raw_image_index.is_integer():
            raise ValueError('query_image_index must be an integer.')
        if query['image_index'] < 0:
            raise ValueError('query_image_index must be zero or greater.')

    return query, metric_matrix, minimum_eigenvalue, base_metric


@csrf_exempt
def mahalanobis_search_api(request):
    """Search with a caller-supplied Mahalanobis matrix and return result embeddings."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'}, status=405)
    if not image_similarity_service.dataset_loaded:
        return JsonResponse({'success': False, 'error': 'Dataset not loaded'}, status=503)

    try:
        query, metric_matrix, minimum_eigenvalue, base_metric = _parse_mahalanobis_api_payload(request)

        if query['type'] == 'text':
            query_vector = clip_service.text_to_vector(
                query['text'],
                normalize=(base_metric == 'cosine'),
            )
        else:
            query_name, query_vector = image_similarity_service.get_image_by_index(
                query['image_index'],
                distance_metric=base_metric,
            )
            query['image_name'] = query_name

        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        query['embedding'] = query_vector.tolist()

        # Keep this endpoint aligned with the UI's personalized-search path.
        from .feedback.matrix_manager import MatrixManager

        scaling_factor = float(MatrixManager.calculate_scaling_factor(metric_matrix))
        distance_mode = _distance_mode_for_metric(base_metric)
        use_filter_refine = scaling_factor > 1.0
        if use_filter_refine:
            range_stage, quick_radius_ratio = _progressive_filter_refine_params(
                initial_request=True
            )
            search_results = image_similarity_service.search_with_filter_refine(
                query_vector=query_vector,
                metric_matrix=metric_matrix,
                scaling_factor=scaling_factor,
                num_results=MAHALANOBIS_API_RESULT_COUNT,
                growth_factor=1.0,
                distance_mode=distance_mode,
                range_stage=range_stage,
                quick_radius_ratio=quick_radius_ratio,
            )
            result_metric_name = 'mahalanobis'
            search_mode = 'filter_and_refine'
        else:
            # This is the same optimization used by the UI for identity/unscaled
            # matrices. It also keeps FAISS k-NN results identical to the UI.
            range_stage = None
            search_results = image_similarity_service.search_by_vector(
                query_vector=query_vector,
                num_results=MAHALANOBIS_API_RESULT_COUNT,
                distance_metric=base_metric,
            )
            result_metric_name = base_metric
            search_mode = 'base_knn'
        raw_results = sorted(
            search_results.get('results', []),
            key=lambda result: float(result['distance']),
        )
        result_indices = [int(result['index']) for result in raw_results]
        result_vectors = image_similarity_service.get_vectors_by_indices(
            result_indices,
            distance_metric=base_metric,
        )

        results = []
        for rank, (result, vector) in enumerate(zip(raw_results, result_vectors), start=1):
            image_name = str(result['image_name'])
            results.append({
                'rank': rank,
                'position': rank,
                'index': int(result['index']),
                'image_name': image_name,
                'image_url': request.build_absolute_uri(build_image_url(image_name)),
                'distance': float(result['distance']),
                'embedding': np.asarray(vector, dtype=np.float32).tolist(),
            })

        return JsonResponse({
            'success': True,
            'query': query,
            'metric': {
                'name': result_metric_name,
                'candidate_distance_metric': base_metric,
                'distance_mode': distance_mode,
                'dimension': int(metric_matrix.shape[0]),
                'minimum_eigenvalue': minimum_eigenvalue,
                'scaling_factor': scaling_factor,
                'matrix_applied': use_filter_refine,
            },
            'ordered_by': 'distance_ascending',
            'search_mode': search_mode,
            'progressive_stage': (
                search_results.get('progressive_stage', range_stage)
                if use_filter_refine
                else None
            ),
            'requested_result_count': MAHALANOBIS_API_RESULT_COUNT,
            'result_count': len(results),
            'results': results,
        })
    except (ValueError, TypeError, IndexError) as exc:
        return JsonResponse({'success': False, 'error': str(exc)}, status=400)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': str(exc),
            'error_type': exc.__class__.__name__,
        }, status=500)


def search_api(request):
    """API endpoint for similarity search."""
    if not image_similarity_service.dataset_loaded:
        return JsonResponse({'error': 'Dataset not loaded'}, status=400)

    try:
        anchor_index = int(request.GET.get('anchor', 0))
        num_results = int(request.GET.get('num_results', 20))
        distance_metric = request.GET.get('metric', 'euclidean')
        query_signature = _build_query_signature(request, image_index=anchor_index)
        _maybe_reset_matrix_on_query_change(
            request,
            query_signature,
            previous_session_id=request.session.get('feedback_session_id'),
        )
        _finalize_previous_query_ranking_time(request)

        # Get user's metric matrix if logged in
        metric_matrix = None
        scaling_factor = 1.0
        distance_mode = _distance_mode_for_metric(distance_metric)
        use_filter_refine = False
        user_matrix_obj = None
        user_id = request.session.get('user_id')
        user = None
        if user_id:
            user, user_matrix_obj, metric_matrix, scaling_factor = _get_authenticated_search_state(
                user_id,
                include_matrix=True
            )

        if _should_use_filter_refine(user_matrix_obj, scaling_factor) and metric_matrix is not None:
            use_filter_refine = True

        if use_filter_refine and metric_matrix is not None:
            range_stage, quick_radius_ratio = _progressive_filter_refine_params(initial_request=False)
            anchor_metric = 'cosine' if distance_mode == 'dot_product' else 'euclidean'
            anchor_name, anchor_vector = image_similarity_service.get_image_by_index(
                anchor_index, distance_metric=anchor_metric
            )
            results = image_similarity_service.search_with_filter_refine(
                query_vector=anchor_vector,
                metric_matrix=metric_matrix,
                scaling_factor=scaling_factor,
                num_results=num_results,
                growth_factor=1.0,
                distance_mode=distance_mode,
                range_stage=range_stage,
                quick_radius_ratio=quick_radius_ratio
            )
            results['anchor_index'] = int(anchor_index)
            results['anchor_name'] = anchor_name
            results['query_type'] = 'image'
        else:
            anchor_metric = distance_metric
            anchor_name, anchor_vector = image_similarity_service.get_image_by_index(
                anchor_index, distance_metric=anchor_metric
            )
            results = image_similarity_service.search_similar_images(
                anchor_index=anchor_index,
                num_results=num_results,
                distance_metric=distance_metric,
                metric_matrix=metric_matrix
            )

        # Store query info in session for feedback
        request.session['current_query_vector'] = anchor_vector.tolist()
        request.session['current_query_type'] = 'image'
        request.session['current_query_image_index'] = anchor_index

        # Log query for authenticated users
        query_log_ref = None
        if user_id and user_matrix_obj:
            try:
                from .feedback import QueryLogger
                if use_filter_refine and metric_matrix is not None:
                    log_stage, log_quick_ratio = _progressive_filter_refine_params(initial_request=False)
                    fetch_more_results_fn = lambda k: image_similarity_service.search_with_filter_refine(
                        query_vector=anchor_vector,
                        metric_matrix=metric_matrix,
                        scaling_factor=scaling_factor,
                        num_results=k,
                        growth_factor=1.0,
                        distance_mode=distance_mode,
                        range_stage=log_stage,
                        quick_radius_ratio=log_quick_ratio
                    )
                else:
                    fetch_more_results_fn = lambda k: image_similarity_service.search_similar_images(
                        anchor_index=anchor_index,
                        num_results=k,
                        distance_metric=distance_metric,
                        metric_matrix=metric_matrix
                    )

                results_for_logging = _build_results_for_query_logging(
                    results,
                    fetch_more_results_fn=fetch_more_results_fn
                )
                query_log_ref = QueryLogger.log_query(
                    user=user,
                    user_matrix_obj=user_matrix_obj,
                    query_info={
                        'type': 'image',
                        'text': '',
                        'image_index': int(anchor_index),
                        'session_id': request.session.get('feedback_session_id'),
                    },
                    results=results_for_logging
                )
            except Exception as e:
                print(f"Warning: Could not log query in search_api: {e}")

        if query_log_ref:
            _remember_last_logged_query_ref(request, query_log_ref)
            _mark_last_query_results_delivered(request)
        _remember_query_signature(request, query_signature)
        return JsonResponse(results)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


def text_search_api(request):
    """API endpoint for text-based similarity search."""
    if not image_similarity_service.dataset_loaded:
        return JsonResponse({'error': 'Dataset not loaded'}, status=400)

    try:
        text_query = request.GET.get('text', '')
        if not text_query:
            return JsonResponse({'error': 'Text query required'}, status=400)
        query_signature = _build_query_signature(request, text_query=text_query)
        _maybe_reset_matrix_on_query_change(
            request,
            query_signature,
            previous_session_id=request.session.get('feedback_session_id'),
        )
        _finalize_previous_query_ranking_time(request)

        num_results = int(request.GET.get('num_results', 20))
        distance_metric = request.GET.get('metric', 'cosine')

        if not clip_service.model_loaded:
            dataset_dim = image_similarity_service.dataset.shape[1]
            clip_service.target_dim = dataset_dim
            print(f"Configuring CLIP to match dataset dimension: {dataset_dim}")

        # Get user's metric matrix if logged in
        metric_matrix = None
        scaling_factor = 1.0
        distance_mode = _distance_mode_for_metric(distance_metric)
        use_filter_refine = False
        user_matrix_obj = None
        user_id = request.session.get('user_id')
        user = None
        if user_id:
            user, user_matrix_obj, metric_matrix, scaling_factor = _get_authenticated_search_state(
                user_id,
                include_matrix=True
            )

        if _should_use_filter_refine(user_matrix_obj, scaling_factor) and metric_matrix is not None:
            use_filter_refine = True

        if use_filter_refine:
            normalize_text_query = (distance_mode == 'dot_product')
        else:
            normalize_text_query = (distance_metric in ('cosine', 'dot_product'))
        clip_encode_start = time.perf_counter()
        text_vector = clip_service.text_to_vector(
            text_query,
            normalize=normalize_text_query
        )
        clip_encode_ms = (time.perf_counter() - clip_encode_start) * 1000

        if use_filter_refine and metric_matrix is not None:
            range_stage, quick_radius_ratio = _progressive_filter_refine_params(initial_request=False)
            results = image_similarity_service.search_with_filter_refine(
                query_vector=text_vector,
                metric_matrix=metric_matrix,
                scaling_factor=scaling_factor,
                num_results=num_results,
                growth_factor=1.0,
                distance_mode=distance_mode,
                range_stage=range_stage,
                quick_radius_ratio=quick_radius_ratio
            )
        else:
            results = image_similarity_service.search_by_vector(
                query_vector=text_vector,
                num_results=num_results,
                distance_metric=distance_metric,
                metric_matrix=metric_matrix
            )

        results['query_text'] = text_query
        results['clip_encode_ms'] = round(clip_encode_ms, 3)

        # Store query info in session for feedback
        request.session['current_query_vector'] = text_vector.tolist()
        request.session['current_query_type'] = 'text'
        request.session['current_query_text'] = text_query

        # Log query for authenticated users
        query_log_ref = None
        if user_id and user_matrix_obj:
            try:
                from .feedback import QueryLogger
                if use_filter_refine and metric_matrix is not None:
                    log_stage, log_quick_ratio = _progressive_filter_refine_params(initial_request=False)
                    fetch_more_results_fn = lambda k: image_similarity_service.search_with_filter_refine(
                        query_vector=text_vector,
                        metric_matrix=metric_matrix,
                        scaling_factor=scaling_factor,
                        num_results=k,
                        growth_factor=1.0,
                        distance_mode=distance_mode,
                        range_stage=log_stage,
                        quick_radius_ratio=log_quick_ratio
                    )
                else:
                    fetch_more_results_fn = lambda k: image_similarity_service.search_by_vector(
                        query_vector=text_vector,
                        num_results=k,
                        distance_metric=distance_metric,
                        metric_matrix=metric_matrix
                    )

                results_for_logging = _build_results_for_query_logging(
                    results,
                    fetch_more_results_fn=fetch_more_results_fn
                )
                query_log_ref = QueryLogger.log_query(
                    user=user,
                    user_matrix_obj=user_matrix_obj,
                    query_info={
                        'type': 'text',
                        'text': text_query,
                        'image_index': None,
                        'session_id': request.session.get('feedback_session_id'),
                    },
                    results=results_for_logging
                )
            except Exception as e:
                print(f"Warning: Could not log query in text_search_api: {e}")

        if query_log_ref:
            _remember_last_logged_query_ref(request, query_log_ref)
            _mark_last_query_results_delivered(request)
        _remember_query_signature(request, query_signature)
        return JsonResponse(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=400)


def image_upload_search(request):
    """Handle image upload and search for similar images."""
    if not image_similarity_service.dataset_loaded:
        return JsonResponse({'error': 'Dataset not loaded'}, status=400)

    if request.method == 'POST' and request.FILES.get('image'):
        try:
            uploaded_image = request.FILES['image']
            num_results = int(request.POST.get('num_results', 20))
            distance_metric = request.POST.get('metric', 'cosine')

            # Load CLIP model if not loaded
            if not clip_service.model_loaded:
                clip_service.load_model()

            # Save uploaded image temporarily for display
            from django.shortcuts import redirect
            from PIL import Image
            import io
            import base64

            # Read image data
            uploaded_image.seek(0)
            image_data = uploaded_image.read()

            # Convert to base64 for display in template
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            image_mime = uploaded_image.content_type or 'image/jpeg'

            # Convert to PIL Image for processing
            image = Image.open(io.BytesIO(image_data)).convert('RGB')

            # Get image embedding
            image_vector = clip_service.image_to_vector(image)

            # Store in session for display (with the search results)
            request.session['uploaded_image_data'] = f"data:{image_mime};base64,{image_base64}"
            request.session['uploaded_image_name'] = uploaded_image.name
            request.session['uploaded_image_vector'] = image_vector.tolist()
            request.session['uploaded_query_token'] = str(uuid.uuid4())

            # Redirect to home with query parameters
            from django.http import QueryDict
            query_params = QueryDict(mutable=True)
            query_params['upload_search'] = 'true'
            query_params['metric'] = distance_metric
            query_params['num_results'] = str(num_results)

            return redirect(f"{reverse('home')}?{query_params.urlencode()}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            from django.contrib import messages
            from django.shortcuts import redirect
            messages.error(request, f'Error processing image: {str(e)}')
            return redirect('home')

    return JsonResponse({'error': 'POST with image file required'}, status=405)


def load_clip_model(request):
    """API endpoint to load CLIP model."""
    if request.method == 'POST':
        try:
            model_name = request.POST.get('model_name', None)

            # If dataset is loaded, auto-select the correct model
            if image_similarity_service.dataset_loaded and not model_name:
                dataset_dim = image_similarity_service.dataset.shape[1]
                if dataset_dim == 768:
                    model_name = 'ViT-L/14'
                    print(f"Auto-selected ViT-L/14 for 768-dim dataset")
                elif dataset_dim == 512:
                    model_name = 'ViT-B/32'
                    print(f"Auto-selected ViT-B/32 for 512-dim dataset")
                else:
                    model_name = 'ViT-B/32'
                    clip_service.target_dim = dataset_dim
            elif not model_name:
                model_name = 'ViT-L/14'  # Default to larger model

            clip_service.model_name = model_name
            clip_service.load_model()

            return JsonResponse({
                'success': True,
                'model': model_name,
                'device': clip_service.device,
                'native_dim': clip_service.native_dim,
                'target_dim': clip_service.target_dim,
                'message': f'Loaded {model_name} ({clip_service.native_dim}-dim) on {clip_service.device}'
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)

    return JsonResponse({'error': 'POST required'}, status=405)


def serve_image(request, image_path):
    """Serve images from the external directory."""
    try:
        base_path = settings.IMAGE_BASE_PATH

        # Optionally restrict serving only to folders selected by DATASET_FILES.
        normalized_path = image_path.replace('\\', '/').lstrip('/')
        path_parts = normalized_path.split('/', 1)
        folder_name = path_parts[0] if path_parts else ''
        allowed_folders = getattr(config, 'ALLOWED_IMAGE_FOLDERS_SET', set())
        if allowed_folders and folder_name not in allowed_folders:
            raise Http404("Image folder not allowed")

        full_path = os.path.join(base_path, image_path)

        full_path = os.path.abspath(full_path)
        base_path = os.path.abspath(base_path)

        if not full_path.startswith(base_path):
            raise Http404("Invalid image path")

        # Prefer proxy/web-server offload when configured so Gunicorn workers
        # are not blocked streaming image files from shared storage.
        accel_prefix = getattr(config, 'IMAGE_X_ACCEL_REDIRECT_PREFIX', '')
        if accel_prefix:
            accel_path = f"{accel_prefix.rstrip('/')}/{normalized_path}"
            response = HttpResponse()
            response['X-Accel-Redirect'] = accel_path
            content_type, _ = mimetypes.guess_type(full_path)
            if content_type:
                response['Content-Type'] = content_type
            response['Cache-Control'] = 'public, max-age=86400'
            return response

        if getattr(config, 'IMAGE_X_SENDFILE', False):
            response = HttpResponse()
            response['X-Sendfile'] = full_path
            content_type, _ = mimetypes.guess_type(full_path)
            if content_type:
                response['Content-Type'] = content_type
            response['Cache-Control'] = 'public, max-age=86400'
            return response

        try:
            response = FileResponse(open(full_path, 'rb'))
        except FileNotFoundError:
            raise Http404("Image not found")

        # Let browser/proxy cache immutable image paths to reduce repeated latency.
        response['Cache-Control'] = 'public, max-age=86400'
        return response
    except Exception as e:
        raise Http404(f"Error serving image: {str(e)}")


# Authentication Views
from django.shortcuts import redirect
from django.contrib import messages
from .models import User

def login_view(request):
    """Simple login view."""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        try:
            user = User.objects.get(username=username)
            if user.check_password(password):
                request.session['user_id'] = user.id
                request.session['username'] = user.username
                request.session['force_sidebar_collapsed'] = True
                messages.success(request, f'Welcome back, {username}!')
                return redirect('home')
            else:
                messages.error(request, 'Invalid username or password')
        except User.DoesNotExist:
            messages.error(request, 'Invalid username or password')

    return render(request, 'login.html')


def register_view(request):
    """Simple registration view."""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')
        email = request.POST.get('email', '')

        # Basic validation
        if not username or not password:
            messages.error(request, 'Username and password are required')
        elif password != password_confirm:
            messages.error(request, 'Passwords do not match')
        elif User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists')
        else:
            # Create new user
            user = User(username=username, email=email)
            user.set_password(password)
            try:
                user.save()
            except IntegrityError:
                # Handle concurrent registration attempts for the same username.
                messages.error(request, 'Username already exists')
                return render(request, 'register.html')

            # Create the initial identity matrix row so first-time settings are
            # fully populated and consistent immediately after registration.
            from .feedback import MatrixManager
            dataset_dim = image_similarity_service.vector_dim
            if not dataset_dim:
                dataset_dim = 768
                try:
                    if image_similarity_service.dataset_loaded and image_similarity_service.dataset is not None:
                        dataset_dim = int(image_similarity_service.dataset.shape[1])
                except Exception:
                    dataset_dim = 768
            MatrixManager.create_default_user_matrix(user, dataset_dim)

            # Auto login after registration
            request.session['user_id'] = user.id
            request.session['username'] = user.username
            request.session['force_sidebar_collapsed'] = True
            messages.success(request, f'Welcome, {username}! Your account has been created.')
            return redirect('home')

    return render(request, 'register.html')


def logout_view(request):
    """Simple logout view."""
    username = request.session.get('username', 'User')
    request.session.flush()
    messages.success(request, f'Goodbye, {username}!')
    return redirect('home')


# Feedback and Metric Learning Views
from .feedback import FeedbackManager, MatrixManager, QueryLogger
from .models import FeedbackApplyJob, UserMetricMatrix
import uuid

FEEDBACK_APPLY_JOB_STALE_SECONDS = 30 * 60


def _feedback_session_results_from_request(request, session_id):
    """Return the result indices needed for implicit feedback, if available."""
    if not session_id:
        return None

    session_results_key = f'session_results_{session_id}'
    if session_results_key in request.session:
        return {session_id: request.session[session_results_key]}

    current_results = request.session.get('current_results', [])
    if current_results:
        return {session_id: current_results}

    return None


def _mark_stale_feedback_apply_job(job):
    if job.status not in {FeedbackApplyJob.STATUS_PENDING, FeedbackApplyJob.STATUS_RUNNING}:
        return False

    from datetime import timedelta
    from django.utils import timezone

    reference_time = job.started_at or job.created_at
    if not reference_time:
        return False
    if timezone.now() - reference_time <= timedelta(seconds=FEEDBACK_APPLY_JOB_STALE_SECONDS):
        return False

    job.status = FeedbackApplyJob.STATUS_FAILED
    job.completed_at = timezone.now()
    job.error = 'Feedback application did not finish. Please try again.'
    job.error_type = 'FeedbackApplyJobTimeout'
    job.save(update_fields=['status', 'completed_at', 'error', 'error_type', 'updated_at'])
    return True


def _serialize_feedback_apply_result(updated_matrix, scaling_factor,
                                     feedback_processing_matrix_save_ms,
                                     model_learning_ms,
                                     feedback_processing_breakdown,
                                     feedback_search_id):
    return {
        'message': 'Metric matrix updated with feedback',
        'matrix_shape': list(updated_matrix.shape),
        'scaling_factor': float(scaling_factor),
        'feedback_search_id': feedback_search_id,
        'feedback_processing_matrix_save_ms': round(feedback_processing_matrix_save_ms, 2),
        'model_learning_ms': round(model_learning_ms, 2),
        'feedback_processing': {
            'feedback_db_ms': round(float(feedback_processing_breakdown.get('feedback_db_ms', 0.0)), 3),
            'feedback_vector_fetch_ms': round(float(feedback_processing_breakdown.get('feedback_vector_fetch_ms', 0.0)), 3),
            'scaling_factor_ms': round(float(feedback_processing_breakdown.get('scaling_factor_ms', 0.0)), 3),
            'matrix_db_save_ms': round(float(feedback_processing_breakdown.get('matrix_db_save_ms', 0.0)), 3),
        }
    }


def _run_feedback_apply_job(job_id, user_id, session_id, session_results):
    """Apply feedback in a worker thread and persist status for polling clients."""
    from django.db import close_old_connections
    from django.utils import timezone
    import traceback

    close_old_connections()
    try:
        FeedbackApplyJob.objects.filter(id=job_id).update(
            status=FeedbackApplyJob.STATUS_RUNNING,
            started_at=timezone.now(),
            error='',
            error_type='',
        )

        if not image_similarity_service.dataset_loaded:
            raise RuntimeError('Dataset not loaded')

        user = User.objects.only('id', 'username').get(id=user_id)
        (
            updated_matrix,
            scaling_factor,
            feedback_processing_matrix_save_ms,
            model_learning_ms,
            feedback_processing_breakdown,
            feedback_search_id
        ) = MatrixManager.update_metric_matrix(
            user=user,
            dataset=image_similarity_service.dataset,
            session_id=session_id,
            session_results=session_results,
            image_names=image_similarity_service.image_names
        )

        result = _serialize_feedback_apply_result(
            updated_matrix,
            scaling_factor,
            feedback_processing_matrix_save_ms,
            model_learning_ms,
            feedback_processing_breakdown,
            feedback_search_id
        )
        FeedbackApplyJob.objects.filter(id=job_id).update(
            status=FeedbackApplyJob.STATUS_COMPLETED,
            result=result,
            completed_at=timezone.now(),
            error='',
            error_type='',
        )
    except Exception as e:
        print("[ERROR] async apply_feedback_learning failed", flush=True)
        traceback.print_exc()
        FeedbackApplyJob.objects.filter(id=job_id).update(
            status=FeedbackApplyJob.STATUS_FAILED,
            completed_at=timezone.now(),
            error=str(e),
            error_type=e.__class__.__name__,
        )
    finally:
        close_old_connections()


def _start_feedback_apply_job(user, session_id, session_results):
    import threading

    existing_job = FeedbackApplyJob.objects.filter(
        user=user,
        session_id=session_id,
        status__in=[FeedbackApplyJob.STATUS_PENDING, FeedbackApplyJob.STATUS_RUNNING]
    ).order_by('-created_at').first()
    if existing_job:
        if not _mark_stale_feedback_apply_job(existing_job):
            return existing_job

    job = FeedbackApplyJob.objects.create(
        user=user,
        session_id=session_id,
        status=FeedbackApplyJob.STATUS_PENDING,
    )
    worker = threading.Thread(
        target=_run_feedback_apply_job,
        args=(job.id, user.id, session_id, session_results),
        name=f'feedback-apply-{job.id}',
        daemon=True,
    )
    worker.start()
    return job


def save_feedback(request):
    """API endpoint to save user feedback on a result image."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    # Check if user is logged in
    user_id = request.session.get('user_id')
    if not user_id:
        return JsonResponse({'error': 'User not logged in'}, status=401)

    try:
        user = User.objects.get(id=user_id)

        # Get feedback data
        result_image_index = int(request.POST.get('image_index'))
        result_image_name = request.POST.get('image_name')
        feedback_type = request.POST.get('feedback_type')  # 'positive' or 'negative'
        session_id = request.POST.get('session_id', str(uuid.uuid4()))

        # Get query_image_index and convert empty string to None
        query_image_index_str = request.POST.get('query_image_index', '')
        query_image_index = int(query_image_index_str) if query_image_index_str and query_image_index_str.strip() else None

        # Get query information from session or POST
        query_info = {
            'type': request.POST.get('query_type', 'unknown'),
            'text': request.POST.get('query_text', ''),
            'image_index': query_image_index,
            'vector': request.session.get('current_query_vector')  # Stored during search
        }

        # Save feedback
        feedback = FeedbackManager.save_feedback(
            user=user,
            session_id=session_id,
            query_info=query_info,
            result_image_index=result_image_index,
            result_image_name=result_image_name,
            feedback_type=feedback_type
        )

        # Store all result indices in a session-specific key for implicit feedback
        session_results_key = f'session_results_{session_id}'
        if session_results_key not in request.session:
            # Initialize with current results from the query
            current_results = request.session.get('current_results', [])
            if current_results:
                request.session[session_results_key] = current_results
                request.session.modified = True
                print(f"[INFO] Stored {len(current_results)} result indices for session {session_id[:8]}... (for implicit feedback)")
            else:
                print(f"[WARN] No current_results found in session for implicit feedback")

        return JsonResponse({
            'success': True,
            'feedback_id': feedback.id,
            'session_id': session_id,
            'message': f'Feedback saved: {feedback_type}'
        })

    except Exception as e:
        import traceback
        print("[ERROR] save_feedback failed", flush=True)
        print(
            "[ERROR] save_feedback payload:",
            {
                'user_id': user_id,
                'session_id': request.POST.get('session_id'),
                'image_index': request.POST.get('image_index'),
                'feedback_type': request.POST.get('feedback_type'),
                'query_type': request.POST.get('query_type'),
                'query_image_index': request.POST.get('query_image_index'),
            },
            flush=True
        )
        traceback.print_exc()
        return JsonResponse({'error': str(e), 'error_type': e.__class__.__name__}, status=400)


def get_session_feedback(request):
    """API endpoint to get feedback for the current session."""
    user_id = request.session.get('user_id')
    if not user_id:
        return JsonResponse({'error': 'User not logged in'}, status=401)

    try:
        user = User.objects.get(id=user_id)
        session_id = request.GET.get('session_id')

        if not session_id:
            return JsonResponse({'error': 'session_id required'}, status=400)

        feedback_items = FeedbackManager.get_session_feedback(user, session_id)

        feedback_data = []
        for item in feedback_items:
            feedback_data.append({
                'id': item.id,
                'image_index': item.result_image_index,
                'image_name': item.result_image_name,
                'feedback_type': item.feedback_type,
                'created_at': item.created_at.isoformat()
            })

        return JsonResponse({
            'success': True,
            'feedback': feedback_data
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


def apply_feedback_learning(request):
    """API endpoint to apply feedback and update metric matrix."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    user_id = request.session.get('user_id')
    if not user_id:
        return JsonResponse({'error': 'User not logged in'}, status=401)

    if not image_similarity_service.dataset_loaded:
        return JsonResponse({'error': 'Dataset not loaded'}, status=400)

    try:
        user = User.objects.get(id=user_id)
        session_id = request.POST.get('session_id')
        session_results = _feedback_session_results_from_request(request, session_id)

        if str(request.POST.get('async', '')).strip() == '1':
            job = _start_feedback_apply_job(user, session_id, session_results)
            return JsonResponse({
                'success': True,
                'async': True,
                'job_id': str(job.id),
                'status': job.status,
            }, status=202)

        # Update metric matrix based on feedback
        (
            updated_matrix,
            scaling_factor,
            feedback_processing_matrix_save_ms,
            model_learning_ms,
            feedback_processing_breakdown,
            feedback_search_id
        ) = MatrixManager.update_metric_matrix(
            user=user,
            dataset=image_similarity_service.dataset,
            session_id=session_id,
            session_results=session_results,
            image_names=image_similarity_service.image_names
        )
        if feedback_search_id:
            request.session['pending_feedback_search_id'] = feedback_search_id

        result = _serialize_feedback_apply_result(
            updated_matrix,
            scaling_factor,
            feedback_processing_matrix_save_ms,
            model_learning_ms,
            feedback_processing_breakdown,
            feedback_search_id
        )
        return JsonResponse({'success': True, **result})

    except Exception as e:
        import traceback
        print("[ERROR] apply_feedback_learning failed", flush=True)
        print(
            "[ERROR] apply_feedback payload:",
            {
                'user_id': user_id,
                'session_id': request.POST.get('session_id'),
            },
            flush=True
        )
        traceback.print_exc()
        return JsonResponse({'error': str(e), 'error_type': e.__class__.__name__}, status=400)


def feedback_apply_job_status(request, job_id):
    """API endpoint to poll an asynchronous feedback application job."""
    user_id = request.session.get('user_id')
    if not user_id:
        return JsonResponse({'error': 'User not logged in'}, status=401)

    try:
        job = FeedbackApplyJob.objects.get(id=job_id, user_id=user_id)
    except FeedbackApplyJob.DoesNotExist:
        return JsonResponse({'error': 'Feedback apply job not found'}, status=404)

    _mark_stale_feedback_apply_job(job)

    response = {
        'success': True,
        'async': True,
        'job_id': str(job.id),
        'status': job.status,
    }

    if job.status == FeedbackApplyJob.STATUS_COMPLETED:
        result = job.result or {}
        response.update(result)
        feedback_search_id = result.get('feedback_search_id')
        if feedback_search_id:
            request.session['pending_feedback_search_id'] = feedback_search_id
    elif job.status == FeedbackApplyJob.STATUS_FAILED:
        response.update({
            'success': False,
            'error': job.error or 'Feedback application failed',
            'error_type': job.error_type,
        })

    return JsonResponse(response)


def reset_metric_matrix(request):
    """API endpoint to reset user's metric matrix to identity."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    user_id = request.session.get('user_id')
    if not user_id:
        return JsonResponse({'error': 'User not logged in'}, status=401)

    if not image_similarity_service.dataset_loaded:
        return JsonResponse({'error': 'Dataset not loaded'}, status=400)

    try:
        user = User.objects.get(id=user_id)
        dataset_dim = _current_dataset_dimension()

        MatrixManager.reset_user_matrix(user, dataset_dim)

        return JsonResponse({
            'success': True,
            'message': 'Metric matrix reset to identity'
        })

    except Exception as e:
        import traceback
        print("[ERROR] reset_metric_matrix failed", flush=True)
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=400)


def download_current_matrix(request):
    """Download the current matrix folder, including an up-to-date snapshot."""
    if request.method != 'GET':
        return JsonResponse({'error': 'GET required'}, status=405)

    user_id = request.session.get('user_id')
    if not user_id:
        return JsonResponse({'error': 'User not logged in'}, status=401)

    try:
        user = User.objects.get(id=user_id)
        user_matrix_obj = UserMetricMatrix.objects.get(user=user)

        # A username is also used as an on-disk directory name by QueryLogger.
        # Reject path-like values before creating or reading an export folder.
        username = str(user.username)
        if not username or os.path.basename(username) != username or username in {'.', '..'}:
            return JsonResponse({'error': 'Invalid username for matrix export'}, status=400)

        # Force a snapshot so matrix.npy and matrix_metadata.json always match
        # the current database state, even when periodic snapshots are disabled.
        matrix_folder = QueryLogger.save_matrix_state(
            user,
            user_matrix_obj,
            force=True,
        )
        if not matrix_folder or not os.path.isdir(matrix_folder):
            return JsonResponse({'error': 'No matrix is available to download'}, status=404)

        matrix_folder = os.path.realpath(matrix_folder)
        folder_name = os.path.basename(matrix_folder)
        if not re.fullmatch(r'matrix_\d+', folder_name):
            return JsonResponse({'error': 'Invalid matrix export folder'}, status=400)

        archive = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024, mode='w+b')
        try:
            with zipfile.ZipFile(
                archive,
                mode='w',
                compression=zipfile.ZIP_DEFLATED,
                allowZip64=True,
            ) as zip_file:
                for root, directories, files in os.walk(matrix_folder, followlinks=False):
                    directories[:] = [
                        directory for directory in directories
                        if not os.path.islink(os.path.join(root, directory))
                    ]
                    for filename in files:
                        file_path = os.path.join(root, filename)
                        if os.path.islink(file_path):
                            continue

                        real_file_path = os.path.realpath(file_path)
                        if os.path.commonpath([matrix_folder, real_file_path]) != matrix_folder:
                            continue

                        relative_path = os.path.relpath(real_file_path, matrix_folder)
                        archive_path = os.path.join(folder_name, relative_path)
                        zip_file.write(real_file_path, archive_path)

            archive.seek(0)
            safe_username = re.sub(r'[^A-Za-z0-9_.-]+', '_', username).strip('._-') or 'user'
            response = FileResponse(
                archive,
                as_attachment=True,
                filename=f'{safe_username}_{folder_name}.zip',
                content_type='application/zip',
            )
            response['Cache-Control'] = 'no-store'
            return response
        except Exception:
            archive.close()
            raise

    except User.DoesNotExist:
        return JsonResponse({'error': 'User not found'}, status=404)
    except UserMetricMatrix.DoesNotExist:
        return JsonResponse({'error': 'No learned matrix is available yet'}, status=404)
    except Exception as e:
        import traceback
        print("[ERROR] download_current_matrix failed", flush=True)
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=400)


def update_learning_settings(request):
    """API endpoint to update metric learning settings."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    user_id = request.session.get('user_id')
    if not user_id:
        return JsonResponse({'error': 'User not logged in'}, status=401)

    if not image_similarity_service.dataset_loaded:
        return JsonResponse({'error': 'Dataset not loaded'}, status=400)

    try:
        user = User.objects.get(id=user_id)
        dataset_dim = image_similarity_service.dataset.shape[1]

        # Get settings from POST data
        model_name = request.POST.get('model_name')
        feedback_type_str = request.POST.get('feedback_type')
        feedback_type = int(feedback_type_str) if feedback_type_str else None

        # Get hyperparameters (if provided as JSON strings)
        import json
        model_params = None
        feedback_params = None

        if request.POST.get('model_params'):
            model_params = json.loads(request.POST.get('model_params'))
        if request.POST.get('feedback_params'):
            feedback_params = json.loads(request.POST.get('feedback_params'))

        MatrixManager.update_learning_settings(
            user=user,
            dataset_dim=dataset_dim,
            model_name=model_name,
            model_params=model_params,
            feedback_type=feedback_type,
            feedback_params=feedback_params
        )

        return JsonResponse({
            'success': True,
            'message': 'Personalization settings updated'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=400)


def get_learning_settings(request):
    """API endpoint to get current metric learning settings."""
    user_id = request.session.get('user_id')
    if not user_id:
        return JsonResponse({'error': 'User not logged in'}, status=401)

    if not image_similarity_service.dataset_loaded:
        # Warmup can still be in progress during startup; return a non-error payload.
        return JsonResponse({
            'success': False,
            'loading': True,
            'error': 'Dataset not loaded yet'
        })

    try:
        user = User.objects.get(id=user_id)
        dataset_dim = image_similarity_service.dataset.shape[1]

        try:
            user_matrix_obj = UserMetricMatrix.objects.get(user=user)

            # Get feedback params - merge defaults with saved custom params
            from .feedback.constants import FEEDBACK_TYPE_PARAMS
            feedback_type = user_matrix_obj.feedback_type
            default_feedback_params = {}
            if feedback_type in FEEDBACK_TYPE_PARAMS:
                for param_name, param_info in FEEDBACK_TYPE_PARAMS[feedback_type].items():
                    default_feedback_params[param_name] = param_info['default']
            saved_feedback_params = user_matrix_obj.feedback_hyperparameters or {}
            allowed_param_names = set(default_feedback_params.keys())
            filtered_saved_feedback_params = {
                k: v for k, v in saved_feedback_params.items() if k in allowed_param_names
            }
            merged_feedback_params = {**default_feedback_params, **filtered_saved_feedback_params}

            settings = {
                'model_name': user_matrix_obj.metric_learning_model,
                'model_params': user_matrix_obj.model_hyperparameters,
                'feedback_type': user_matrix_obj.feedback_type,
                'feedback_params': merged_feedback_params,
                'scaling_factor': user_matrix_obj.scaling_factor,
                'last_updated': user_matrix_obj.updated_at.isoformat()
            }
        except UserMetricMatrix.DoesNotExist:
            # Return defaults
            from .feedback.constants import (
                INITIAL_MODEL,
                INITIAL_FEEDBACK_TYPE,
                INITIAL_MODEL_HYPERPARAMETERS,
                FEEDBACK_TYPE_PARAMS,
            )
            default_feedback_params = {}
            if INITIAL_FEEDBACK_TYPE in FEEDBACK_TYPE_PARAMS:
                for param_name, param_info in FEEDBACK_TYPE_PARAMS[INITIAL_FEEDBACK_TYPE].items():
                    default_feedback_params[param_name] = param_info['default']
            settings = {
                'model_name': INITIAL_MODEL,
                'model_params': INITIAL_MODEL_HYPERPARAMETERS.copy(),
                'feedback_type': INITIAL_FEEDBACK_TYPE,
                'feedback_params': default_feedback_params,
                'scaling_factor': 1.0,
                'last_updated': None
            }

        # Add available models
        from .feedback.constants import METRIC_LEARNING_MODELS
        settings['available_models'] = list(METRIC_LEARNING_MODELS.keys())

        return JsonResponse({
            'success': True,
            'settings': settings
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


def get_available_settings(request):
    """API endpoint to get all available models and feedback types with their parameters."""
    return JsonResponse(_build_available_settings_payload())


@lru_cache(maxsize=1)
def _build_available_settings_payload():
    """Build static settings metadata payload once and reuse it."""
    from .feedback.constants import (
        METRIC_LEARNING_MODELS,
        MODEL_FEEDBACK_DEFAULTS,
        FEEDBACK_TYPE_NAMES,
        FEEDBACK_TYPE_DESCRIPTIONS,
        FEEDBACK_TYPE_PARAMS,
        MODEL_DESCRIPTIONS,
        MODEL_CITATIONS,
        MODEL_PARAM_DESCRIPTIONS
    )

    # Build response with all available options
    models_info = {}
    for model_name in METRIC_LEARNING_MODELS.keys():
        feedback_types_info = {}
        for feedback_type in [1, 2, 3]:
            if feedback_type in MODEL_FEEDBACK_DEFAULTS.get(model_name, {}):
                feedback_types_info[feedback_type] = {
                    'name': FEEDBACK_TYPE_NAMES.get(feedback_type, f'Type {feedback_type}'),
                    'description': FEEDBACK_TYPE_DESCRIPTIONS.get(feedback_type, ''),
                    'default_params': MODEL_FEEDBACK_DEFAULTS[model_name][feedback_type]
                }
        models_info[model_name] = {
            'pair_type': METRIC_LEARNING_MODELS[model_name]['pair_type'],
            'feedback_types': feedback_types_info,
            'description': MODEL_DESCRIPTIONS.get(model_name, ''),
            'citation': MODEL_CITATIONS.get(model_name, ''),
            'param_descriptions': MODEL_PARAM_DESCRIPTIONS.get(model_name, {}),
        }

    # Build feedback type params info
    feedback_type_params_info = {}
    for ft, params in FEEDBACK_TYPE_PARAMS.items():
        feedback_type_params_info[ft] = {}
        for param_name, param_info in params.items():
            feedback_type_params_info[ft][param_name] = {
                'default': param_info['default'],
                'type': param_info['type'],
                'description': param_info['description'],
                'allowed_values': param_info.get('allowed_values'),
                'min': param_info.get('min'),
                'max': param_info.get('max'),
            }

    return {
        'success': True,
        'models': models_info,
        'feedback_type_names': FEEDBACK_TYPE_NAMES,
        'feedback_type_descriptions': FEEDBACK_TYPE_DESCRIPTIONS,
        'feedback_type_params': feedback_type_params_info
    }


def rerun_query(request):
    """API endpoint to rerun the same query after feedback is applied."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    if not image_similarity_service.dataset_loaded:
        return JsonResponse({'error': 'Dataset not loaded'}, status=400)

    try:
        # Get the current query information from session
        query_vector = request.session.get('current_query_vector')
        query_type = request.session.get('current_query_type')
        previous_results = list(request.session.get('current_results', []) or [])
        rerun_rank_baseline = list(request.session.get('rerun_rank_baseline_results', []) or [])

        if not query_vector:
            return JsonResponse({'error': 'No active query to rerun'}, status=400)

        query_vector = np.array(query_vector)
        num_results = int(request.POST.get('num_results', 20))
        distance_metric = request.POST.get('metric', 'cosine')
        background_only = str(request.POST.get('background_only', '')).strip() == '1'
        progressive_full = str(request.POST.get('progressive_full', '')).strip() == '1'
        feedback_search_id = (request.POST.get('feedback_search_id') or '').strip() or None
        if feedback_search_id is None:
            feedback_search_id = request.session.get('pending_feedback_search_id')
        if not background_only:
            _finalize_previous_query_ranking_time(request)
            request.session['rerun_rank_baseline_results'] = previous_results
            request.session.modified = True
        elif progressive_full and rerun_rank_baseline:
            previous_results = rerun_rank_baseline

        # Get user's metric matrix if logged in
        metric_matrix = None
        scaling_factor = 1.0
        distance_mode = _distance_mode_for_metric(distance_metric)
        use_filter_refine = False
        user_id = request.session.get('user_id')
        user = None
        user_matrix_obj = None

        if user_id:
            user, user_matrix_obj, metric_matrix, scaling_factor = _get_authenticated_search_state(
                user_id,
                include_matrix=True
            )

        if _should_use_filter_refine(user_matrix_obj, scaling_factor) and metric_matrix is not None:
            use_filter_refine = True
            distance_mode = _distance_mode_for_metric(distance_metric)

        if query_type == 'text':
            query_text = request.session.get('current_query_text', '')
            if query_text:
                if use_filter_refine:
                    normalize_text_query = (distance_mode == 'dot_product')
                else:
                    normalize_text_query = (distance_metric in ('cosine', 'dot_product'))
                clip_encode_start = time.perf_counter()
                query_vector = clip_service.text_to_vector(
                    query_text,
                    normalize=normalize_text_query
                )
                clip_encode_ms = (time.perf_counter() - clip_encode_start) * 1000
                results_clip_encode_ms = round(clip_encode_ms, 3)
                request.session['current_query_vector'] = query_vector.tolist()
            else:
                results_clip_encode_ms = None
        else:
            results_clip_encode_ms = None

        # Perform the search
        if use_filter_refine and metric_matrix is not None:
            # Rerun now supports progressive two-stage behavior:
            # quick stage by default, full stage only when progressive_full=1.
            range_stage, quick_radius_ratio = _progressive_filter_refine_params(
                initial_request=(not progressive_full)
            )
            results = image_similarity_service.search_with_filter_refine(
                query_vector=query_vector,
                metric_matrix=metric_matrix,
                scaling_factor=scaling_factor,
                num_results=num_results,
                growth_factor=1.0,
                distance_mode=distance_mode,
                range_stage=range_stage,
                quick_radius_ratio=quick_radius_ratio
            )
        else:
            results = image_similarity_service.search_by_vector(
                query_vector=query_vector,
                num_results=num_results,
                distance_metric=distance_metric,
                metric_matrix=metric_matrix
            )

        # Add query info to results
        results['query_type'] = query_type
        if query_type == 'text':
            results['query_text'] = request.session.get('current_query_text', '')
            if results_clip_encode_ms is not None:
                results['clip_encode_ms'] = results_clip_encode_ms
        elif query_type == 'image':
            results['anchor_index'] = request.session.get('current_query_image_index')
            if results['anchor_index'] is not None:
                if use_filter_refine:
                    anchor_metric = 'cosine' if distance_mode == 'dot_product' else 'euclidean'
                else:
                    anchor_metric = distance_metric
                results['anchor_name'] = image_similarity_service.get_image_name_by_index(
                    results['anchor_index'],
                    distance_metric=anchor_metric
                )
        elif query_type == 'uploaded_image':
            results['uploaded_image_name'] = request.session.get('uploaded_image_name', 'uploaded_image')
            results['uploaded_image_data'] = request.session.get('uploaded_image_data', '')

        results = _annotate_result_rank_changes(results, previous_results)

        # Keep session results aligned with what the user currently sees.
        request.session['current_results'] = [r['index'] for r in results.get('results', [])]
        if not results.get('progressive_pending'):
            request.session.pop('rerun_rank_baseline_results', None)
        request.session.modified = True

        # Log rerun query as a regular query entry (plus full payload file).
        query_log_ref = None
        if user and user_matrix_obj and not background_only:
            try:
                from .feedback import QueryLogger
                if use_filter_refine and metric_matrix is not None:
                    log_stage, log_quick_ratio = _progressive_filter_refine_params(
                        initial_request=(not progressive_full)
                    )
                    fetch_more_results_fn = lambda k: image_similarity_service.search_with_filter_refine(
                        query_vector=query_vector,
                        metric_matrix=metric_matrix,
                        scaling_factor=scaling_factor,
                        num_results=k,
                        growth_factor=1.0,
                        distance_mode=distance_mode,
                        range_stage=log_stage,
                        quick_radius_ratio=log_quick_ratio
                    )
                else:
                    fetch_more_results_fn = lambda k: image_similarity_service.search_by_vector(
                        query_vector=query_vector,
                        num_results=k,
                        distance_metric=distance_metric,
                        metric_matrix=metric_matrix
                    )

                results_for_logging = _build_results_for_query_logging(
                    results,
                    fetch_more_results_fn=fetch_more_results_fn
                )
                query_info = {
                    'type': query_type,
                    'text': request.session.get('current_query_text', '') if query_type == 'text' else '',
                    'image_index': request.session.get('current_query_image_index') if query_type == 'image' else None,
                    'session_id': request.session.get('feedback_session_id'),
                    'feedback_search_id': feedback_search_id,
                }
                query_log_ref = QueryLogger.log_query(
                    user=user,
                    user_matrix_obj=user_matrix_obj,
                    query_info=query_info,
                    results=results_for_logging
                )
            except Exception as e:
                print(f"Warning: Could not log rerun query: {e}")
        if query_log_ref:
            _remember_last_logged_query_ref(request, query_log_ref)
            _mark_last_query_results_delivered(request)

        # Update feedback log with result counts and flow time (for rerun flow)
        total_flow_time_ms_str = request.POST.get('total_flow_time_ms')
        total_flow_time_ms = float(total_flow_time_ms_str) if total_flow_time_ms_str else None
        feedback_processing_matrix_save_ms_str = request.POST.get('feedback_processing_matrix_save_ms')
        feedback_processing_matrix_save_ms = (
            float(feedback_processing_matrix_save_ms_str)
            if feedback_processing_matrix_save_ms_str else None
        )
        model_learning_ms_str = request.POST.get('model_learning_ms')
        model_learning_ms = float(model_learning_ms_str) if model_learning_ms_str else None
        timing_fields = _extract_feedback_timing_fields(
            results,
            total_flow_time_ms=total_flow_time_ms,
            feedback_processing_matrix_save_ms=feedback_processing_matrix_save_ms,
            model_learning_ms=model_learning_ms
        )
        if user and user_matrix_obj and (not background_only or feedback_search_id):
            try:
                from .feedback import QueryLogger
                QueryLogger.update_feedback_log_timing(
                    user=user,
                    user_matrix_obj=user_matrix_obj,
                    total_flow_time_ms=timing_fields['total_flow_time_ms'],
                    feedback_processing_matrix_save_ms=timing_fields['feedback_processing_matrix_save_ms'],
                    model_learning_ms=timing_fields['model_learning_ms'],
                    feedback_search_id=feedback_search_id,
                    num_results_requested=num_results,
                    num_results_candidate_set=timing_fields['num_results_candidate_set'],
                    num_results_candidate_set_progressive_filter=timing_fields['num_results_candidate_set_progressive_filter'],
                    num_results_candidate_set_full_mahalanobis_filter=timing_fields['num_results_candidate_set_full_mahalanobis_filter'],
                    num_results_refined_set=timing_fields['num_results_refined_set'],
                    range_r_e=timing_fields['range_r_e'],
                    range_progressive=timing_fields['range_progressive'],
                    range_r_m_full=timing_fields['range_r_m_full'],
                    clip_encode_ms=timing_fields['clip_encode_ms'],
                    faiss_knn_ms=timing_fields['faiss_knn_ms'],
                    faiss_range_ms=timing_fields['faiss_range_ms'],
                    reconstruct_ms=timing_fields['reconstruct_ms'],
                    refine_ms=timing_fields['refine_ms'],
                    distance_mode=timing_fields['distance_mode'],
                    distance_metric=timing_fields['distance_metric'],
                    base_metric=timing_fields['base_metric'],
                    search_pipeline=timing_fields['search_pipeline'],
                )
            except Exception as e:
                print(f"⚠️ Could not update feedback log with result counts: {e}")

        if feedback_search_id and not background_only:
            request.session.pop('pending_feedback_search_id', None)

        return JsonResponse({
            'success': True,
            'results': results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=400)
