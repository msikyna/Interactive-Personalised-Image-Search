"""
Feedback and metric learning module.
Decomposed for better maintainability and separation of concerns.
"""

from .feedback_manager import FeedbackManager
from .matrix_manager import MatrixManager
from .metric_learner import MetricLearner
from .triplet_generator import TripletGenerator
from .query_logger import QueryLogger
from .constants import MODEL_FEEDBACK_DEFAULTS, METRIC_LEARNING_MODELS, FEEDBACK_TYPE_PARAMS

# Main service class for backwards compatibility
class FeedbackService:
    """Unified interface for feedback and metric learning operations."""

    # Expose constants
    MODEL_FEEDBACK_DEFAULTS = MODEL_FEEDBACK_DEFAULTS
    METRIC_LEARNING_MODELS = METRIC_LEARNING_MODELS

    def __init__(self):
        self.feedback_manager = FeedbackManager()
        self.matrix_manager = MatrixManager()
        self.metric_learner = MetricLearner()
        self.triplet_generator = TripletGenerator()

    # Delegate to appropriate managers
    @staticmethod
    def get_default_params(model_name, feedback_type):
        return MatrixManager.get_default_params(model_name, feedback_type)

    @staticmethod
    def calculate_scaling_factor(M):
        return MatrixManager.calculate_scaling_factor(M)

    @staticmethod
    def save_feedback(user, session_id, query_info, result_image_index, result_image_name, feedback_type):
        return FeedbackManager.save_feedback(user, session_id, query_info, result_image_index, result_image_name, feedback_type)

    @staticmethod
    def get_session_feedback(user, session_id):
        return FeedbackManager.get_session_feedback(user, session_id)

    @staticmethod
    def group_feedback_by_query(feedback_items, dataset):
        return FeedbackManager.group_feedback_by_query(feedback_items, dataset)

    @staticmethod
    def update_metric_matrix(user, dataset, session_id=None):
        return MatrixManager.update_metric_matrix(user, dataset, session_id)

    @staticmethod
    def reset_user_matrix(user, dimension):
        return MatrixManager.reset_user_matrix(user, dimension)

    @staticmethod
    def update_learning_settings(user, dataset_dim, model_name=None, model_params=None,
                                  feedback_type=None, feedback_params=None):
        return MatrixManager.update_learning_settings(user, dataset_dim, model_name, model_params,
                                                       feedback_type, feedback_params)

    @staticmethod
    def apply_feedback_type_1(M, query_vector, positives, negatives, dataset, model_name,
                               model_params, num_iterations=10, with_replacement=True):
        return TripletGenerator.apply_feedback_type_1(M, query_vector, positives, negatives, dataset,
                                                       model_name, model_params, num_iterations, with_replacement)

    @staticmethod
    def apply_feedback_type_2(M, query_vector, positives, negatives, dataset, model_name,
                               model_params, batch_mode=True):
        return TripletGenerator.apply_feedback_type_2(M, query_vector, positives, negatives, dataset,
                                                       model_name, model_params, batch_mode)

    @staticmethod
    def apply_feedback_type_3(M, query_vector, positives, negatives, dataset, model_name, model_params):
        return TripletGenerator.apply_feedback_type_3(M, query_vector, positives, negatives, dataset,
                                                       model_name, model_params)


# Create singleton instance for backwards compatibility
feedback_service = FeedbackService()

__all__ = [
    'FeedbackService',
    'FeedbackManager',
    'MatrixManager',
    'MetricLearner',
    'TripletGenerator',
    'feedback_service',
    'MODEL_FEEDBACK_DEFAULTS',
    'METRIC_LEARNING_MODELS',
]

