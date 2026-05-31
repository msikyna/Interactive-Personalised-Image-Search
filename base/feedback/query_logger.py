"""
Query logger for saving query history and results to disk.
Handles per-user folder structure and JSON logging.
"""

import os
import json
from datetime import datetime
import numpy as np
from .. import config


class QueryLogger:
    """Logs queries and results to disk per user."""

    @staticmethod
    def _file_logging_enabled():
        return bool(getattr(config, 'ENABLE_QUERY_FILE_LOGGING', False))

    @staticmethod
    def _resolve_model_params(user_matrix_obj):
        """Return effective model params for current model + feedback type."""
        from .constants import MODEL_FEEDBACK_DEFAULTS

        model_name = user_matrix_obj.metric_learning_model
        feedback_type = user_matrix_obj.feedback_type
        defaults = MODEL_FEEDBACK_DEFAULTS.get(model_name, {}).get(feedback_type, {})
        custom = user_matrix_obj.model_hyperparameters or {}
        return {**defaults, **custom}

    @staticmethod
    def _resolve_feedback_params(user_matrix_obj):
        """Return effective feedback params filtered to the active feedback type."""
        from .constants import FEEDBACK_TYPE_PARAMS

        feedback_type = user_matrix_obj.feedback_type
        param_defs = FEEDBACK_TYPE_PARAMS.get(feedback_type, {})
        defaults = {name: info['default'] for name, info in param_defs.items() if 'default' in info}
        custom = user_matrix_obj.feedback_hyperparameters or {}
        filtered_custom = {name: value for name, value in custom.items() if name in param_defs}
        return {**defaults, **filtered_custom}

    @staticmethod
    def _infer_runtime_distance_mode(results, fallback=None):
        """
        Infer effective distance_mode from runtime search output.
        Returns 'dot_product', 'euclidean', or fallback.
        """
        if not isinstance(results, dict):
            return fallback

        mode = str(results.get('distance_mode') or '').strip().lower()
        if mode in ('dot_product', 'euclidean'):
            return mode

        metric = str(results.get('distance_metric') or '').strip().lower()
        if metric in ('cosine', 'dot_product'):
            return 'dot_product'
        if metric == 'euclidean':
            return 'euclidean'
        if metric == 'mahalanobis':
            base_metric = str(results.get('base_metric') or '').strip().lower()
            if base_metric in ('cosine', 'dot_product'):
                return 'dot_product'
            if base_metric == 'euclidean':
                return 'euclidean'

        return fallback

    @staticmethod
    def get_user_folder(user, base_path='user_matrices'):
        """
        Get the folder path for a user.

        Args:
            user: User object
            base_path: Base directory for user folders

        Returns:
            Path to user folder
        """
        user_folder = os.path.join(base_path, user.username)
        os.makedirs(user_folder, exist_ok=True)
        return user_folder

    @staticmethod
    def get_matrix_folder(user, user_matrix_obj, base_path='user_matrices'):
        """
        Get the folder path for the current matrix state.
        Creates a new folder when matrix is reset.

        Args:
            user: User object
            user_matrix_obj: UserMetricMatrix object
            base_path: Base directory for user folders

        Returns:
            Path to matrix folder
        """
        user_folder = QueryLogger.get_user_folder(user, base_path)

        # Count existing matrix folders to determine next number
        existing_folders = [d for d in os.listdir(user_folder) if d.startswith('matrix_')]
        if not existing_folders:
            matrix_num = 1
        else:
            # Get the highest number
            nums = [int(f.split('_')[1]) for f in existing_folders if f.split('_')[1].isdigit()]
            matrix_num = max(nums) if nums else 1

            # Check if we should create a new folder (matrix was reset)
            # We check if the current matrix is identity - if so, start a new folder
            M = user_matrix_obj.get_matrix()
            if np.allclose(M, np.eye(M.shape[0])) and user_matrix_obj.scaling_factor == 1.0:
                # Check if there's already a query logged in this folder
                current_folder = os.path.join(user_folder, f'matrix_{matrix_num}')
                queries_file = os.path.join(current_folder, 'queries.json')
                if os.path.exists(queries_file):
                    with open(queries_file, 'r') as f:
                        queries = json.load(f)
                        if queries:  # If there are queries, create new folder
                            matrix_num += 1

        matrix_folder = os.path.join(user_folder, f'matrix_{matrix_num}')
        os.makedirs(matrix_folder, exist_ok=True)
        return matrix_folder

    @staticmethod
    def save_matrix_state(user, user_matrix_obj, base_path='user_matrices'):
        """
        Save the current matrix state to disk.

        Args:
            user: User object
            user_matrix_obj: UserMetricMatrix object
            base_path: Base directory for user folders
        """
        # Matrix snapshots are large; keep them optional for performance.
        if not getattr(config, 'SAVE_MATRIX_SNAPSHOTS', False):
            return

        from .constants import FEEDBACK_TYPE_NAMES

        matrix_folder = QueryLogger.get_matrix_folder(user, user_matrix_obj, base_path)

        M = user_matrix_obj.get_matrix()
        matrix_file = os.path.join(matrix_folder, 'matrix.npy')
        np.save(matrix_file, M)

        model_params = QueryLogger._resolve_model_params(user_matrix_obj)
        feedback_params = QueryLogger._resolve_feedback_params(user_matrix_obj)

        # Save metadata with all required fields
        metadata = {
            'scaling_factor': float(user_matrix_obj.scaling_factor),
            'model': user_matrix_obj.metric_learning_model,
            'feedback_type': user_matrix_obj.feedback_type,
            'feedback_type_name': FEEDBACK_TYPE_NAMES.get(user_matrix_obj.feedback_type, f'Type {user_matrix_obj.feedback_type}'),
            'model_params': model_params,
            'feedback_params': feedback_params,
            'distance_mode': user_matrix_obj.distance_mode,
            'updated_at': user_matrix_obj.updated_at.isoformat(),
            'dimension': M.shape[0]
        }

        metadata_file = os.path.join(matrix_folder, 'matrix_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        if getattr(config, 'VERBOSE_RUNTIME_LOGS', False):
            print(f"    Matrix saved to: {matrix_file}")

    @staticmethod
    def log_query(user, user_matrix_obj, query_info, results, feedback_data=None, base_path='user_matrices'):
        """
        Log a query and its results to disk.

        Args:
            user: User object
            user_matrix_obj: UserMetricMatrix object
            query_info: Dictionary with query details
            results: Dictionary with search results
            feedback_data: Dictionary with feedback information (positives, negatives, implicit)
            base_path: Base directory for user folders
        """
        if not QueryLogger._file_logging_enabled():
            return

        from .constants import FEEDBACK_TYPE_NAMES

        matrix_folder = QueryLogger.get_matrix_folder(user, user_matrix_obj, base_path)

        # Load existing queries
        queries_file = os.path.join(matrix_folder, 'queries.json')
        if os.path.exists(queries_file):
            with open(queries_file, 'r') as f:
                queries = json.load(f)
        else:
            queries = []

        model_params = QueryLogger._resolve_model_params(user_matrix_obj)
        feedback_params = QueryLogger._resolve_feedback_params(user_matrix_obj)

        runtime_distance_mode = QueryLogger._infer_runtime_distance_mode(
            results,
            fallback=user_matrix_obj.distance_mode
        )

        query_session_id = query_info.get('session_id')
        feedback_search_id = query_info.get('feedback_search_id')

        all_results = results.get('results', []) or []
        try:
            target_logged_count = max(1, int(getattr(config, 'QUERY_LOG_RESULTS_COUNT', 100)))
        except (TypeError, ValueError):
            target_logged_count = 100
        logged_results = all_results[:target_logged_count]

        # Create query log entry
        query_log = {
            'query_session_id': query_session_id,
            'feedback_search_id': feedback_search_id,
            'timestamp': datetime.now().isoformat(),
            'query_type': query_info.get('type'),
            'query_text': query_info.get('text', ''),
            'query_image_index': query_info.get('image_index'),
            # Filled when user triggers the next query/rerun action.
            'query_ranking_time_ms': None,
            'model': user_matrix_obj.metric_learning_model,
            'feedback_type': user_matrix_obj.feedback_type,
            'feedback_type_name': FEEDBACK_TYPE_NAMES.get(user_matrix_obj.feedback_type, f'Type {user_matrix_obj.feedback_type}'),
            'model_params': model_params,
            'feedback_params': feedback_params,
            'scaling_factor': float(user_matrix_obj.scaling_factor),
            'distance_mode': runtime_distance_mode,
            'distance_metric': results.get('distance_metric'),
            'base_metric': results.get('base_metric'),
            'search_pipeline': results.get('search_pipeline'),
            'num_results_total': len(all_results),
            'num_results_logged': len(logged_results),
            'results': [
                {
                    'index': r['index'],
                    'image_name': r['image_name'],
                    'distance': float(r['distance'])
                }
                for r in logged_results
            ]
        }

        # Add feedback information if provided
        if feedback_data:
            query_log['feedback'] = {
                'positives': feedback_data.get('positives', []),
                'negatives': feedback_data.get('negatives', []),
                'implicit_positives': feedback_data.get('implicit_positives', []),
                'implicit_negatives': feedback_data.get('implicit_negatives', [])
            }

        queries.append(query_log)

        # Save updated queries
        with open(queries_file, 'w') as f:
            json.dump(queries, f, indent=2)

        if getattr(config, 'VERBOSE_RUNTIME_LOGS', False):
            print(f"    Query logged to: {queries_file}")

        return {
            'queries_file': queries_file,
            'query_index': len(queries) - 1,
        }

    @staticmethod
    def update_query_ranking_time(queries_file, query_index, query_ranking_time_ms):
        """
        Update one query entry in queries.json with user ranking time.

        Args:
            queries_file: Absolute path to queries.json
            query_index: Index of the query entry in the JSON list
            query_ranking_time_ms: Milliseconds from result delivery to next user query action

        Returns:
            True if updated or already set, otherwise False.
        """
        if not QueryLogger._file_logging_enabled():
            return False
        if not queries_file or query_index is None or query_ranking_time_ms is None:
            return False
        if not os.path.exists(queries_file):
            return False

        try:
            idx = int(query_index)
            ranking_ms = max(0.0, float(query_ranking_time_ms))
        except (TypeError, ValueError):
            return False

        try:
            with open(queries_file, 'r') as f:
                queries = json.load(f)

            if not isinstance(queries, list):
                return False
            if idx < 0 or idx >= len(queries):
                return False
            if not isinstance(queries[idx], dict):
                return False

            if queries[idx].get('query_ranking_time_ms') is None:
                queries[idx]['query_ranking_time_ms'] = round(ranking_ms, 2)

            with open(queries_file, 'w') as f:
                json.dump(queries, f, indent=2)
            return True
        except Exception:
            return False

    @staticmethod
    def log_feedback(user, user_matrix_obj, session_id, query_groups,
                     scaling_factor_before=None, image_names=None,
                     feedback_search_id=None,
                     feedback_processing_matrix_save_ms=None,
                     model_learning_ms=None,
                     feedback_processing=None,
                     total_flow_time_ms=None,
                     num_results_requested=None,
                     num_results_candidate_set=None,
                     num_results_candidate_set_progressive_filter=None,
                     num_results_candidate_set_full_mahalanobis_filter=None,
                     num_results_refined_set=None,
                     range_r_e=None,
                     range_progressive=None,
                     range_r_m_full=None,
                     clip_encode_ms=None,
                     faiss_knn_ms=None,
                     faiss_range_ms=None,
                     reconstruct_ms=None,
                     refine_ms=None,
                     base_path='user_matrices'):
        """
        Log feedback data (positives, negatives, implicit selections) to disk.
        This is the most important data we collect from users.

        Args:
            user: User object
            user_matrix_obj: UserMetricMatrix object
            session_id: Session ID for this feedback
            query_groups: Dictionary of query groups with feedback data
            scaling_factor_before: Scaling factor before feedback was applied
            image_names: List of image filenames indexed by dataset position
            feedback_search_id: Unique ID linking this feedback application to its subsequent query
            feedback_processing_matrix_save_ms: Time for feedback processing + matrix save, excluding model update calls (ms)
            model_learning_ms: Time spent in model update calls (ms)
            feedback_processing: Optional dict with detailed feedback-processing timings:
                {
                    'feedback_db_ms': ...,
                    'feedback_vector_fetch_ms': ...,
                    'scaling_factor_ms': ...,
                    'matrix_db_save_ms': ...,
                }
            total_flow_time_ms: Total time for query submission and result return (ms)
            num_results_requested: Number of results the user requested
            num_results_candidate_set: Size of candidate superset
            num_results_candidate_set_progressive_filter: Candidate set size for progressive (quick) range
            num_results_candidate_set_full_mahalanobis_filter: Candidate set size for full Mahalanobis filter range
            num_results_refined_set: Size of refined final set
            range_r_e: Full expansion range rE
            range_progressive: Progressive (quick) range value
            range_r_m_full: Full Mahalanobis base range rM
            clip_encode_ms: Time spent encoding query with CLIP
            faiss_knn_ms: Time spent in FAISS kNN stage
            faiss_range_ms: Time spent in FAISS range-search stage
            reconstruct_ms: Time spent reconstructing candidate vectors
            refine_ms: Time spent Mahalanobis refinement stage
            base_path: Base directory for user folders
        """
        if not QueryLogger._file_logging_enabled():
            return

        from .constants import FEEDBACK_TYPE_NAMES

        matrix_folder = QueryLogger.get_matrix_folder(user, user_matrix_obj, base_path)

        # Load existing feedback log
        feedback_file = os.path.join(matrix_folder, 'feedback_log.json')
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                feedback_logs = json.load(f)
        else:
            feedback_logs = []

        model_params = QueryLogger._resolve_model_params(user_matrix_obj)
        feedback_params = QueryLogger._resolve_feedback_params(user_matrix_obj)

        # Helper: convert index list to [{index, image_name}] for robust storage
        def indices_to_entries(indices):
            entries = []
            for idx in indices:
                entry = {'index': idx}
                if image_names and 0 <= idx < len(image_names):
                    entry['image_name'] = image_names[idx]
                entries.append(entry)
            return entries

        # Log each query group's feedback
        for query_key, group in query_groups.items():
            query_type, query_text, query_image_index = query_key

            # Determine which are explicit vs implicit
            explicit_positives = group.get('positives', []) if not group.get('implicit_positives') else []
            explicit_negatives = group.get('negatives', []) if not group.get('implicit_negatives') else []
            implicit_positives = group.get('positives', []) if group.get('implicit_positives') else []
            implicit_negatives = group.get('negatives', []) if group.get('implicit_negatives') else []

            # Resolve query image name if possible
            query_img_idx = query_image_index if query_image_index != -1 else None
            query_image_name = None
            if query_img_idx is not None and image_names and 0 <= query_img_idx < len(image_names):
                query_image_name = image_names[query_img_idx]

            feedback_processing_obj = {}
            if isinstance(feedback_processing, dict):
                for key in (
                    'feedback_db_ms',
                    'feedback_vector_fetch_ms',
                    'scaling_factor_ms',
                    'matrix_db_save_ms',
                ):
                    value = feedback_processing.get(key)
                    try:
                        feedback_processing_obj[key] = round(float(value), 3) if value is not None else None
                    except (TypeError, ValueError):
                        feedback_processing_obj[key] = None

            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'feedback_search_id': feedback_search_id,
                'query_type': query_type,
                'query_text': query_text if query_text else None,
                'query_image_index': query_img_idx,
                'query_image_name': query_image_name,
                'model': user_matrix_obj.metric_learning_model,
                'feedback_type': user_matrix_obj.feedback_type,
                'feedback_type_name': FEEDBACK_TYPE_NAMES.get(user_matrix_obj.feedback_type, f'Type {user_matrix_obj.feedback_type}'),
                'model_params': model_params,
                'feedback_params': feedback_params,
                'distance_mode': user_matrix_obj.distance_mode,
                'scaling_factor_before': float(scaling_factor_before) if scaling_factor_before is not None else None,
                'scaling_factor_after': float(user_matrix_obj.scaling_factor),
                'performance_metrics': {
                    'feedback_processing_matrix_save_ms': round(feedback_processing_matrix_save_ms, 2) if feedback_processing_matrix_save_ms is not None else None,
                    'model_learning_ms': round(model_learning_ms, 2) if model_learning_ms is not None else None,
                    'feedback_processing': feedback_processing_obj or None,
                    'total_flow_time_ms': round(total_flow_time_ms, 2) if total_flow_time_ms is not None else None,
                    'num_results_requested': num_results_requested,
                    'num_results_candidate_set': num_results_candidate_set,
                    'num_results_candidate_set_progressive_filter': num_results_candidate_set_progressive_filter,
                    'num_results_candidate_set_full_mahalanobis_filter': num_results_candidate_set_full_mahalanobis_filter,
                    'num_results_refined_set': num_results_refined_set,
                    'range_r_e': round(range_r_e, 6) if range_r_e is not None else None,
                    'range_progressive': round(range_progressive, 6) if range_progressive is not None else None,
                    'range_r_m_full': round(range_r_m_full, 6) if range_r_m_full is not None else None,
                    'clip_encode_ms': round(clip_encode_ms, 3) if clip_encode_ms is not None else None,
                    'faiss_knn_ms': round(faiss_knn_ms, 3) if faiss_knn_ms is not None else None,
                    'faiss_range_ms': round(faiss_range_ms, 3) if faiss_range_ms is not None else None,
                    'reconstruct_ms': round(reconstruct_ms, 3) if reconstruct_ms is not None else None,
                    'refine_ms': round(refine_ms, 3) if refine_ms is not None else None,
                },
                'user_feedback': {
                    'selected_positives': indices_to_entries(explicit_positives),
                    'selected_negatives': indices_to_entries(explicit_negatives),
                    'implicit_positives': indices_to_entries(implicit_positives),
                    'implicit_negatives': indices_to_entries(implicit_negatives),
                    'total_positives_used': len(group.get('positives', [])),
                    'total_negatives_used': len(group.get('negatives', []))
                }
            }

            feedback_logs.append(feedback_entry)

        # Save updated feedback log
        with open(feedback_file, 'w') as f:
            json.dump(feedback_logs, f, indent=2)

        if getattr(config, 'VERBOSE_RUNTIME_LOGS', False):
            print(f"    [INFO] Feedback logged to: {feedback_file}")

    @staticmethod
    def update_feedback_log_timing(user, user_matrix_obj, total_flow_time_ms=None,
                                    feedback_processing_matrix_save_ms=None,
                                    model_learning_ms=None,
                                    feedback_search_id=None,
                                    num_results_requested=None,
                                    num_results_candidate_set=None,
                                    num_results_candidate_set_progressive_filter=None,
                                    num_results_candidate_set_full_mahalanobis_filter=None,
                                    num_results_refined_set=None,
                                    range_r_e=None,
                                    range_progressive=None,
                                    range_r_m_full=None,
                                    clip_encode_ms=None,
                                    faiss_knn_ms=None,
                                    faiss_range_ms=None,
                                    reconstruct_ms=None,
                                    refine_ms=None,
                                    distance_mode=None,
                                    distance_metric=None,
                                    base_metric=None,
                                    search_pipeline=None,
                                    base_path='user_matrices'):
        """
        Update the most recent feedback log entry with flow timing and result counts.
        Called after the search completes, since feedback is logged before the search.

        Args:
            user: User object
            user_matrix_obj: UserMetricMatrix object
            total_flow_time_ms: Total time for the entire flow in ms (if omitted, computed as sum of known steps)
            feedback_processing_matrix_save_ms: Time for feedback processing + matrix save, excluding model update calls (ms)
            model_learning_ms: Time spent in model update calls (ms)
            feedback_search_id: Unique ID linking feedback and its subsequent query
            num_results_requested: Number of results the user requested
            num_results_candidate_set: Size of candidate superset
            num_results_candidate_set_progressive_filter: Candidate set size for progressive (quick) range
            num_results_candidate_set_full_mahalanobis_filter: Candidate set size for full Mahalanobis filter range
            num_results_refined_set: Size of refined final set
            range_r_e: Full expansion range rE
            range_progressive: Progressive (quick) range value
            range_r_m_full: Full Mahalanobis base range rM
            clip_encode_ms: Time spent encoding query with CLIP
            faiss_knn_ms: Time spent in FAISS kNN stage
            faiss_range_ms: Time spent in FAISS range-search stage
            reconstruct_ms: Time spent reconstructing candidate vectors
            refine_ms: Time spent Mahalanobis refinement stage
            distance_mode: Effective distance mode for this search ('dot_product' or 'euclidean')
            distance_metric: Returned distance metric label
            base_metric: Base metric used by filter stage (for mahalanobis path)
            search_pipeline: Search pipeline label (e.g., 'filter+refine')
            base_path: Base directory for user folders
        """
        if not QueryLogger._file_logging_enabled():
            return

        matrix_folder = QueryLogger.get_matrix_folder(user, user_matrix_obj, base_path)
        feedback_file = os.path.join(matrix_folder, 'feedback_log.json')

        if not os.path.exists(feedback_file):
            return

        try:
            with open(feedback_file, 'r') as f:
                feedback_logs = json.load(f)

            if not feedback_logs:
                return

            target_entries = []
            if feedback_search_id:
                for entry in feedback_logs:
                    if entry.get('feedback_search_id') == feedback_search_id:
                        target_entries.append(entry)
            if not target_entries:
                target_entries = [feedback_logs[-1]]

            for target_entry in target_entries:
                performance = target_entry.get('performance_metrics')
                if not isinstance(performance, dict):
                    performance = {}
                    target_entry['performance_metrics'] = performance

                if feedback_processing_matrix_save_ms is not None:
                    performance['feedback_processing_matrix_save_ms'] = round(feedback_processing_matrix_save_ms, 2)
                if model_learning_ms is not None:
                    performance['model_learning_ms'] = round(model_learning_ms, 2)
                if num_results_requested is not None:
                    performance['num_results_requested'] = num_results_requested
                if num_results_candidate_set is not None:
                    performance['num_results_candidate_set'] = num_results_candidate_set
                if num_results_candidate_set_progressive_filter is not None:
                    performance['num_results_candidate_set_progressive_filter'] = int(
                        num_results_candidate_set_progressive_filter
                    )
                if num_results_candidate_set_full_mahalanobis_filter is not None:
                    performance['num_results_candidate_set_full_mahalanobis_filter'] = int(
                        num_results_candidate_set_full_mahalanobis_filter
                    )
                if num_results_refined_set is not None:
                    performance['num_results_refined_set'] = num_results_refined_set
                if range_r_e is not None:
                    performance['range_r_e'] = round(float(range_r_e), 6)
                if range_progressive is not None:
                    performance['range_progressive'] = round(float(range_progressive), 6)
                if range_r_m_full is not None:
                    performance['range_r_m_full'] = round(float(range_r_m_full), 6)
                if clip_encode_ms is not None:
                    performance['clip_encode_ms'] = round(clip_encode_ms, 3)
                if faiss_knn_ms is not None:
                    performance['faiss_knn_ms'] = round(faiss_knn_ms, 3)
                if faiss_range_ms is not None:
                    performance['faiss_range_ms'] = round(faiss_range_ms, 3)
                if reconstruct_ms is not None:
                    performance['reconstruct_ms'] = round(reconstruct_ms, 3)
                if refine_ms is not None:
                    performance['refine_ms'] = round(refine_ms, 3)

                # Keep total flow stable unless caller provides explicit total-flow timing.
                if total_flow_time_ms is not None:
                    stage_values = [
                        performance.get('feedback_processing_matrix_save_ms'),
                        performance.get('model_learning_ms'),
                        performance.get('clip_encode_ms'),
                        performance.get('faiss_knn_ms'),
                        performance.get('faiss_range_ms'),
                        performance.get('reconstruct_ms'),
                        performance.get('refine_ms'),
                    ]
                    stage_sum = 0.0
                    has_any_stage = False
                    for value in stage_values:
                        if value is None:
                            continue
                        try:
                            stage_sum += float(value)
                            has_any_stage = True
                        except (TypeError, ValueError):
                            continue
                    if has_any_stage:
                        performance['total_flow_time_ms'] = round(stage_sum, 2)
                    else:
                        performance['total_flow_time_ms'] = round(total_flow_time_ms, 2)

                runtime_mode = QueryLogger._infer_runtime_distance_mode(
                    {
                        'distance_mode': distance_mode,
                        'distance_metric': distance_metric,
                        'base_metric': base_metric,
                    },
                    fallback=None
                )
                if runtime_mode is not None:
                    target_entry['distance_mode'] = runtime_mode
                if distance_metric is not None:
                    target_entry['distance_metric'] = distance_metric
                if base_metric is not None:
                    target_entry['base_metric'] = base_metric
                if search_pipeline is not None:
                    target_entry['search_pipeline'] = search_pipeline

            with open(feedback_file, 'w') as f:
                json.dump(feedback_logs, f, indent=2)
        except Exception as e:
            if getattr(config, 'VERBOSE_RUNTIME_LOGS', False):
                print(f"    [WARNING]]️ Could not update feedback log timing: {e}")
