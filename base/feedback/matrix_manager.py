"""
Matrix operations and management.
Handles metric matrix updates, storage, and user settings.
"""

import numpy as np
import uuid
from ..models import UserMetricMatrix, FeedbackItem
from ..matrix_cache import store_cached_matrix_for_user_matrix
from .. import config
from .constants import (
    MODEL_FEEDBACK_DEFAULTS,
    FEEDBACK_TYPE_PARAMS,
    INITIAL_MODEL,
    INITIAL_FEEDBACK_TYPE,
    INITIAL_MODEL_HYPERPARAMETERS,
)
from .feedback_manager import FeedbackManager
from .triplet_generator import TripletGenerator


def _vprint(*args, **kwargs):
    if getattr(config, 'VERBOSE_RUNTIME_LOGS', False):
        print(*args, **kwargs)


class MatrixManager:
    """Manages metric matrix operations and storage."""

    @staticmethod
    def get_default_params(model_name, feedback_type):
        """
        Get default parameters for a model and feedback type.

        Args:
            model_name: Name of the metric learning model
            feedback_type: Feedback type (1, 2, or 3)

        Returns:
            Dictionary of default parameters
        """
        if model_name in MODEL_FEEDBACK_DEFAULTS:
            if feedback_type in MODEL_FEEDBACK_DEFAULTS[model_name]:
                return MODEL_FEEDBACK_DEFAULTS[model_name][feedback_type].copy()
        # Fallback to feedback type 1 defaults
        return MODEL_FEEDBACK_DEFAULTS.get(model_name, {}).get(1, {}).copy()

    @staticmethod
    def create_default_user_matrix(user, dimension):
        """Create and persist a new identity matrix row with current default settings."""
        dim = int(dimension or 768)
        identity = np.eye(dim)
        user_matrix_obj = UserMetricMatrix(
            user=user,
            matrix_dimension=dim,
            metric_learning_model=INITIAL_MODEL,
            model_hyperparameters=INITIAL_MODEL_HYPERPARAMETERS.copy(),
            feedback_type=INITIAL_FEEDBACK_TYPE,
            feedback_hyperparameters={},
            scaling_factor=1.0,
        )
        user_matrix_obj.set_matrix(identity)
        user_matrix_obj.save()
        store_cached_matrix_for_user_matrix(user_matrix_obj, identity)
        return user_matrix_obj

    @staticmethod
    def calculate_scaling_factor(M):
        """
        Calculate the scaling factor from the metric matrix.
        Based on Metacentrum.py implementation: scaling_factor = 1 / sqrt(min_eigenvalue)

        Args:
            M: Metric matrix

        Returns:
            Scaling factor (reciprocal of square root of minimum eigenvalue)
        """
        eigenvalues, _ = np.linalg.eigh(M)
        min_eigenvalue = np.min(eigenvalues)

        small_cap = 1e-20
        scaling_factor = 10.0  # default cap

        if min_eigenvalue >= small_cap:
            scaling_factor = 1.0 / np.sqrt(min_eigenvalue)

        # Cap at 20 to prevent extreme values
        if scaling_factor > 20:
            scaling_factor = 20.0

        return np.real(scaling_factor)

    @staticmethod
    def update_metric_matrix(user, dataset, session_id=None, session_results=None, image_names=None,
                             total_flow_time_ms=None,
                             num_results_requested=None,
                             num_results_candidate_set=None,
                             num_results_refined_set=None,
                             clip_encode_ms=None,
                             faiss_knn_ms=None,
                             faiss_range_ms=None,
                             reconstruct_ms=None,
                             refine_ms=None):
        """
        Update the user's metric matrix based on collected feedback.

        Args:
            user: User object
            dataset: The image dataset (numpy array)
            session_id: Optional session ID to use only specific feedback
            session_results: Optional dict mapping session_id to list of result indices
                            (used for implicit feedback when user only marks positives or negatives)
            image_names: Optional list of image filenames indexed by dataset position
            total_flow_time_ms: Total time for the entire flow (feedback + search), passed from caller
            num_results_requested: Number of results the user requested
            num_results_candidate_set: Size of candidate superset
            num_results_refined_set: Size of refined final set
            clip_encode_ms: Time spent encoding query with CLIP
            faiss_knn_ms: Time spent in FAISS kNN stage
            faiss_range_ms: Time spent in FAISS range-search stage
            reconstruct_ms: Time spent reconstructing candidate vectors
            refine_ms: Time spent Mahalanobis refinement stage

        Returns:
            Tuple of (
                updated_matrix,
                scaling_factor,
                feedback_processing_matrix_save_ms,
                model_learning_ms,
                feedback_processing_breakdown,
                feedback_search_id
            )
        """
        import time
        feedback_db_ms = 0.0
        feedback_vector_fetch_ms = 0.0
        scaling_factor_ms = 0.0
        matrix_db_save_ms = 0.0
        t_feedback_db = time.perf_counter()

        # Get or create UserMetricMatrix
        try:
            user_matrix_obj = UserMetricMatrix.objects.get(user=user)
        except UserMetricMatrix.DoesNotExist:
            user_matrix_obj = MatrixManager.create_default_user_matrix(user, dataset.shape[1])

        # Get feedback items
        if session_id:
            feedback_items = FeedbackItem.objects.filter(user=user, session_id=session_id)
        else:
            # Get all unapplied feedback (safe because applied items are always deleted)
            feedback_items = FeedbackItem.objects.filter(user=user).order_by('-created_at')

        if not feedback_items.exists():
            # No feedback to process, return current matrix
            M = user_matrix_obj.get_matrix()
            return M, user_matrix_obj.scaling_factor, 0.0, 0.0, {
                'feedback_db_ms': 0.0,
                'feedback_vector_fetch_ms': 0.0,
                'scaling_factor_ms': 0.0,
                'matrix_db_save_ms': 0.0,
            }, None

        # Group feedback by query - pass session_results for implicit feedback
        query_groups = FeedbackManager.group_feedback_by_query(feedback_items, dataset, session_results)

        # Get current matrix
        M = user_matrix_obj.get_matrix()

        # Debug: Print matrix state before update
        _vprint(f"\n{'='*80}")
        _vprint(f"DEBUG: Updating metric matrix for user '{user.username}'")
        _vprint(f"  Session ID: {session_id}")
        _vprint(f"  Matrix shape: {M.shape}")
        _vprint(f"  Matrix diagonal (first 5): {np.diag(M)[:5]}")
        _vprint(f"  Matrix is identity: {np.allclose(M, np.eye(M.shape[0]))}")
        _vprint(f"  Current scaling factor: {user_matrix_obj.scaling_factor}")

        # Get settings
        model_name = user_matrix_obj.metric_learning_model
        feedback_type = user_matrix_obj.feedback_type
        custom_params = user_matrix_obj.model_hyperparameters or {}

        # Get default parameters and merge with custom
        default_params = MatrixManager.get_default_params(model_name, feedback_type)
        model_params = {**default_params, **custom_params}

        _vprint(f"  Using model: {model_name}")
        _vprint(f"  Feedback type: {feedback_type}")
        _vprint(f"  Model params: {model_params}")

        # Get feedback hyperparameters (merge defaults from FEEDBACK_TYPE_PARAMS with user's custom settings)
        default_feedback_params = {}
        if feedback_type in FEEDBACK_TYPE_PARAMS:
            for param_name, param_info in FEEDBACK_TYPE_PARAMS[feedback_type].items():
                default_feedback_params[param_name] = param_info['default']
        custom_feedback_params = user_matrix_obj.feedback_hyperparameters or {}
        feedback_params = {**default_feedback_params, **custom_feedback_params}

        num_iterations = feedback_params.get('num_iterations', 10)
        with_replacement = feedback_params.get('with_replacement', True)
        batch_mode = feedback_params.get('batch_mode', True)
        feedback_db_ms = (time.perf_counter() - t_feedback_db) * 1000.0

        # Capture scaling factor before applying feedback
        scaling_factor_before = float(user_matrix_obj.scaling_factor)
        feedback_search_id = str(uuid.uuid4())
        model_learning_ms_acc = 0.0

        # Apply metric learning for each query with feedback
        for query_key, group in query_groups.items():
            query_vector = group['query_vector']
            positives = group['positives']
            negatives = group['negatives']

            # Debug: Show what we have
            _vprint(f"\n  Query group: {query_key}")
            _vprint(f"    Positives: {len(positives)}, Negatives: {len(negatives)}")
            _vprint(f"    Has query_vector: {query_vector is not None}")
            if group.get('implicit_positives'):
                _vprint(f"    [INFO] Using implicit positives (user only marked negatives)")
            if group.get('implicit_negatives'):
                _vprint(f"    [INFO] Using implicit negatives (user only marked positives)")

            if query_vector is None:
                _vprint(f"    [WARN] Skipping: No query vector available")
                continue
            if not positives:
                _vprint(f"    [WARN] Skipping: No positives (explicit or implicit)")
                continue
            if not negatives:
                _vprint(f"    [WARN] Skipping: No negatives (explicit or implicit)")
                continue

            # Apply feedback based on type
            _vprint(f"    [OK] Processing with {len(positives)} positives, {len(negatives)} negatives")
            M_before = M.copy() if getattr(config, 'VERBOSE_RUNTIME_LOGS', False) else None

            if feedback_type == 1:
                M, model_learning_ms_this_group, vector_fetch_ms_this_group = TripletGenerator.apply_feedback_type_1(
                    M, query_vector, positives, negatives, dataset,
                    model_name, model_params, num_iterations, with_replacement,
                    return_timing=True
                )
            elif feedback_type == 2:
                M, model_learning_ms_this_group, vector_fetch_ms_this_group = TripletGenerator.apply_feedback_type_2(
                    M, query_vector, positives, negatives, dataset,
                    model_name, model_params, batch_mode,
                    return_timing=True
                )
            elif feedback_type == 3:
                M, model_learning_ms_this_group, vector_fetch_ms_this_group = TripletGenerator.apply_feedback_type_3(
                    M, query_vector, positives, negatives, dataset,
                    model_name, model_params,
                    return_timing=True
                )
            else:
                model_learning_ms_this_group = 0.0
                vector_fetch_ms_this_group = 0.0
            model_learning_ms_acc += float(model_learning_ms_this_group)
            feedback_vector_fetch_ms += float(vector_fetch_ms_this_group)

            # Debug: Show matrix change
            if M_before is not None:
                matrix_diff = np.linalg.norm(M - M_before, 'fro')
                _vprint(f"  Matrix changed by (Frobenius norm): {matrix_diff:.6f}")

        # Calculate scaling factor
        t_scaling = time.perf_counter()
        scaling_factor = MatrixManager.calculate_scaling_factor(M)
        scaling_factor_ms = (time.perf_counter() - t_scaling) * 1000.0

        # Debug: Print final matrix state
        _vprint(f"\n  AFTER UPDATE:")
        _vprint(f"  Matrix diagonal (first 5): {np.diag(M)[:5]}")
        _vprint(f"  Matrix is identity: {np.allclose(M, np.eye(M.shape[0]))}")
        _vprint(f"  New scaling factor: {scaling_factor}")
        _vprint(f"  Scaling factor changed: {abs(scaling_factor - user_matrix_obj.scaling_factor) > 1e-6}")
        _vprint(f"{'='*80}\n")

        # Save updated matrix
        t_matrix_save = time.perf_counter()
        user_matrix_obj.set_matrix(M)
        user_matrix_obj.scaling_factor = scaling_factor
        user_matrix_obj.save()
        store_cached_matrix_for_user_matrix(user_matrix_obj, M)
        matrix_db_save_ms = (time.perf_counter() - t_matrix_save) * 1000.0

        feedback_processing_breakdown = {
            'feedback_db_ms': float(feedback_db_ms),
            'feedback_vector_fetch_ms': float(feedback_vector_fetch_ms),
            'scaling_factor_ms': float(scaling_factor_ms),
            'matrix_db_save_ms': float(matrix_db_save_ms),
        }
        feedback_processing_matrix_save_ms = (
            float(feedback_db_ms) +
            float(feedback_vector_fetch_ms) +
            float(scaling_factor_ms) +
            float(matrix_db_save_ms)
        )

        # Save matrix state to disk
        from .query_logger import QueryLogger
        QueryLogger.save_matrix_state(user, user_matrix_obj)

        # Log feedback data
        # It saves: selected positives, selected negatives, implicit positives/negatives
        QueryLogger.log_feedback(user, user_matrix_obj, session_id, query_groups,
                                  scaling_factor_before=scaling_factor_before,
                                  image_names=image_names,
                                  feedback_search_id=feedback_search_id,
                                  feedback_processing_matrix_save_ms=feedback_processing_matrix_save_ms,
                                  model_learning_ms=model_learning_ms_acc,
                                  feedback_processing=feedback_processing_breakdown,
                                  total_flow_time_ms=total_flow_time_ms,
                                  num_results_requested=num_results_requested,
                                  num_results_candidate_set=num_results_candidate_set,
                                  num_results_refined_set=num_results_refined_set,
                                  clip_encode_ms=clip_encode_ms,
                                  faiss_knn_ms=faiss_knn_ms,
                                  faiss_range_ms=faiss_range_ms,
                                  reconstruct_ms=reconstruct_ms,
                                  refine_ms=refine_ms)

        # Delete applied feedback items from the database so they are never reused.
        # Each feedback application is a one-shot input to the model; the learned
        # matrix carries the knowledge forward, and new feedback must be fresh.
        deleted_count, _ = feedback_items.delete()
        _vprint(f"    🗑️ Cleared {deleted_count} applied feedback items from database")

        # Print summary
        for query_key, group in query_groups.items():
            _vprint(f"    Feedback summary: {len(group['positives'])} positives, {len(group['negatives'])} negatives")
        _vprint(
            f"   Feedback timing split: feedback+save={feedback_processing_matrix_save_ms:.1f} ms, "
            f"model_learning={model_learning_ms_acc:.1f} ms"
        )

        return (
            M,
            scaling_factor,
            feedback_processing_matrix_save_ms,
            model_learning_ms_acc,
            feedback_processing_breakdown,
            feedback_search_id,
        )

    @staticmethod
    def reset_user_matrix(user, dimension):
        """
        Reset user's metric matrix to identity matrix.

        Args:
            user: User object
            dimension: Dimension of the identity matrix

        Returns:
            UserMetricMatrix object
        """
        try:
            user_matrix_obj = UserMetricMatrix.objects.get(user=user)
            user_matrix_obj.reset_to_identity()
            store_cached_matrix_for_user_matrix(user_matrix_obj, np.eye(dimension))
        except UserMetricMatrix.DoesNotExist:
            user_matrix_obj = MatrixManager.create_default_user_matrix(user, dimension)

        return user_matrix_obj

    @staticmethod
    def update_learning_settings(user, dataset_dim, model_name=None, model_params=None,
                                  feedback_type=None, feedback_params=None):
        """
        Update user's metric learning settings.

        Args:
            user: User object
            dataset_dim: Dataset dimension
            model_name: Name of the metric learning model (optional)
            model_params: Model hyperparameters (optional)
            feedback_type: Feedback type (1, 2, or 3) (optional)
            feedback_params: Feedback hyperparameters (optional)

        Returns:
            Updated UserMetricMatrix object
        """
        # Get or create UserMetricMatrix
        try:
            user_matrix_obj = UserMetricMatrix.objects.get(user=user)
        except UserMetricMatrix.DoesNotExist:
            user_matrix_obj = MatrixManager.create_default_user_matrix(user, dataset_dim)

        # Update settings if provided
        if model_name is not None:
            user_matrix_obj.metric_learning_model = model_name
            # If model name changed, update to default params for current feedback type
            if model_params is None:
                current_feedback_type = feedback_type if feedback_type is not None else user_matrix_obj.feedback_type
                model_params = MatrixManager.get_default_params(model_name, current_feedback_type)

        if model_params is not None:
            user_matrix_obj.model_hyperparameters = model_params

        if feedback_type is not None:
            user_matrix_obj.feedback_type = feedback_type
            # If feedback type changed, update model params to defaults for new feedback type
            if model_params is None:
                current_model = model_name if model_name is not None else user_matrix_obj.metric_learning_model
                user_matrix_obj.model_hyperparameters = MatrixManager.get_default_params(
                    current_model, feedback_type
                )
            # Also keep only feedback params valid for the newly selected type.
            existing_feedback_params = user_matrix_obj.feedback_hyperparameters or {}
            allowed_feedback_keys = set(FEEDBACK_TYPE_PARAMS.get(feedback_type, {}).keys())
            user_matrix_obj.feedback_hyperparameters = {
                k: v for k, v in existing_feedback_params.items() if k in allowed_feedback_keys
            }

        if feedback_params is not None:
            effective_feedback_type = feedback_type if feedback_type is not None else user_matrix_obj.feedback_type
            allowed_feedback_keys = set(FEEDBACK_TYPE_PARAMS.get(effective_feedback_type, {}).keys())
            user_matrix_obj.feedback_hyperparameters = {
                k: v for k, v in feedback_params.items() if k in allowed_feedback_keys
            }

        user_matrix_obj.save()
        return user_matrix_obj
