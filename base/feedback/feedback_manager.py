"""
Feedback manager for CRUD operations on feedback items.
Handles storage, retrieval, and grouping of user feedback.
"""

from ..models import FeedbackItem
from .. import config


def _vprint(*args, **kwargs):
    if getattr(config, 'VERBOSE_RUNTIME_LOGS', False):
        print(*args, **kwargs)


class FeedbackManager:
    """Manages feedback storage and retrieval."""

    @staticmethod
    def save_feedback(user, session_id, query_info, result_image_index, result_image_name, feedback_type):
        """
        Save user feedback for a result image.

        Args:
            user: User object
            session_id: Session identifier to group feedback
            query_info: Dictionary containing query information (type, text, image_index, vector)
            result_image_index: Index of the result image
            result_image_name: Name of the result image
            feedback_type: 'positive' or 'negative'

        Returns:
            FeedbackItem object
        """
        feedback = FeedbackItem(
            user=user,
            query_type=query_info.get('type', 'unknown'),
            query_text=query_info.get('text', ''),
            query_image_index=query_info.get('image_index'),
            result_image_index=result_image_index,
            result_image_name=result_image_name,
            feedback_type=feedback_type,
            session_id=session_id
        )

        # Store the query vector
        if 'vector' in query_info and query_info['vector'] is not None:
            feedback.set_query_vector(query_info['vector'])

        feedback.save()

        # Debug: Print feedback saved
        _vprint(f"[INFO] FEEDBACK SAVED: User '{user.username}', Type: {feedback_type}, Image: {result_image_name}, Session: {session_id[:8]}...")

        return feedback

    @staticmethod
    def get_session_feedback(user, session_id):
        """
        Get all feedback for a specific session.

        Args:
            user: User object
            session_id: Session identifier

        Returns:
            QuerySet of FeedbackItem objects
        """
        return FeedbackItem.objects.filter(user=user, session_id=session_id)

    @staticmethod
    def group_feedback_by_query(feedback_items, dataset, session_results=None):
        """
        Group feedback by query and extract positives/negatives.
        Auto-selects negatives when only positives are provided (and vice versa).

        Args:
            feedback_items: QuerySet or list of FeedbackItem objects
            dataset: Image dataset (numpy array) for validation
            session_results: Optional dict mapping session_id to list of result indices

        Returns:
            Dictionary mapping query info to {'query_vector', 'positives', 'negatives', 'all_result_indices'}
            Format: {(query_type, query_text, query_image_index): {...}}
        """
        query_groups = {}
        session_to_query = {}  # Map session_id to query_key

        for item in feedback_items:
            query_key = (item.query_type, item.query_text or '', item.query_image_index or -1)
            session_to_query[item.session_id] = query_key

            # Initialize group if not exists
            if query_key not in query_groups:
                query_groups[query_key] = {
                    'query_vector': item.get_query_vector(),
                    'positives': [],
                    'negatives': [],
                    'all_result_indices': [],  # Track all results shown to user
                    'session_id': item.session_id
                }

            # Categorize feedback
            if item.feedback_type == 'positive':
                query_groups[query_key]['positives'].append(item.result_image_index)
            else:
                query_groups[query_key]['negatives'].append(item.result_image_index)

            # Track all result indices
            if item.result_image_index not in query_groups[query_key]['all_result_indices']:
                query_groups[query_key]['all_result_indices'].append(item.result_image_index)

        # Use session_results if provided to get all result indices
        if session_results:
            for session_id, result_indices in session_results.items():
                if session_id in session_to_query:
                    query_key = session_to_query[session_id]
                    # Add any missing result indices
                    for idx in result_indices:
                        if idx not in query_groups[query_key]['all_result_indices']:
                            query_groups[query_key]['all_result_indices'].append(idx)

        # Auto-select negatives when only positives provided (and vice versa)
        # Only select the SAME number of implicit items as explicitly selected,
        # chosen randomly from the remaining results. This avoids training on
        # a heavily imbalanced set where a lazy user's unmarked items get wrongly
        # treated as the opposite class.
        import random

        for query_key, group in query_groups.items():
            positives = group['positives']
            negatives = group['negatives']
            all_results = group['all_result_indices']

            if positives and not negatives and len(all_results) > len(positives):
                # Select implicit negatives: same count as explicit positives, randomly chosen
                candidates = [idx for idx in all_results if idx not in positives]
                num_to_select = min(len(positives), len(candidates))
                group['negatives'] = random.sample(candidates, num_to_select)
                group['implicit_negatives'] = True
                _vprint(f"      Auto-selected {len(group['negatives'])} implicit negatives "
                        f"(matching {len(positives)} explicit positives, from {len(candidates)} candidates)")
            elif negatives and not positives and len(all_results) > len(negatives):
                # Select implicit positives: same count as explicit negatives, randomly chosen
                candidates = [idx for idx in all_results if idx not in negatives]
                num_to_select = min(len(negatives), len(candidates))
                group['positives'] = random.sample(candidates, num_to_select)
                group['implicit_positives'] = True
                _vprint(f"      Auto-selected {len(group['positives'])} implicit positives "
                        f"(matching {len(negatives)} explicit negatives, from {len(candidates)} candidates)")

        return query_groups
