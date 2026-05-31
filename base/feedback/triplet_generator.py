"""
Triplet/pair generation strategies for different feedback types.
Implements the three feedback types from the process_query function.
"""

import random
from .constants import METRIC_LEARNING_MODELS
from .metric_learner import MetricLearner


class TripletGenerator:
    """Generates training pairs/triplets based on feedback type."""

    @staticmethod
    def _timed_fetch_vector(dataset, index):
        import time
        start = time.perf_counter()
        vector = dataset[index]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return vector, elapsed_ms

    @staticmethod
    def apply_feedback_type_1(M, query_vector, positives, negatives, dataset, model_name,
                               model_params, num_iterations=10, with_replacement=True,
                               return_timing=False):
        """
        Feedback Mechanism 1: Random Pairing (from Section 4.2 of the paper).

        A single positive image and a single negative image are randomly selected
        to form one triplet. This process repeats for num_iterations times.
        Sampling can be done with or without replacement.

        Args:
            M: Current metric matrix
            query_vector: Query vector
            positives: List of positive example indices
            negatives: List of negative example indices
            dataset: Full dataset for vector lookup
            model_name: Name of metric learning model
            model_params: Model hyperparameters
            num_iterations: Number of random samples to process
            with_replacement: Whether to sample with replacement

        Returns:
            Updated metric matrix, or (matrix, model_learning_ms, vector_fetch_ms)
            when return_timing=True
        """
        import time
        model_config = METRIC_LEARNING_MODELS.get(model_name, {})
        pair_type = model_config.get('pair_type', 'triplets')
        model_learning_ms = 0.0
        vector_fetch_ms = 0.0

        similar_pool = positives.copy()
        dissimilar_pool = negatives.copy()

        for _ in range(num_iterations):
            if not similar_pool or not dissimilar_pool:
                break

            chosen_similar_idx = random.choice(similar_pool)
            chosen_dissimilar_idx = random.choice(dissimilar_pool)

            if not with_replacement:
                similar_pool.remove(chosen_similar_idx)
                dissimilar_pool.remove(chosen_dissimilar_idx)

            similar_vector, t_fetch_ms = TripletGenerator._timed_fetch_vector(dataset, chosen_similar_idx)
            vector_fetch_ms += t_fetch_ms
            dissimilar_vector, t_fetch_ms = TripletGenerator._timed_fetch_vector(dataset, chosen_dissimilar_idx)
            vector_fetch_ms += t_fetch_ms

            t_model = time.perf_counter()
            M = MetricLearner._apply_single_update(
                M, model_name, model_params, pair_type,
                query_vector, similar_vector, dissimilar_vector
            )
            model_learning_ms += (time.perf_counter() - t_model) * 1000.0

        if return_timing:
            return M, model_learning_ms, vector_fetch_ms
        return M

    @staticmethod
    def apply_feedback_type_2(M, query_vector, positives, negatives, dataset, model_name,
                               model_params, batch_mode=True, return_timing=False):
        """
        Feedback Mechanism 2: Single Negative & All Positives (from Section 4.2 of the paper).

        A single negative image is chosen randomly and paired with all positive images,
        resulting in a triplet set containing multiple triplets. This process repeats
        for each negative image (without replacement).

        Args:
            M: Current metric matrix
            query_vector: Query vector
            positives: List of positive example indices
            negatives: List of negative example indices
            dataset: Full dataset for vector lookup
            model_name: Name of metric learning model
            model_params: Model hyperparameters
            batch_mode: Whether to use batch updates (True) or individual updates (False)

        Returns:
            Updated metric matrix, or (matrix, model_learning_ms, vector_fetch_ms)
            when return_timing=True
        """
        import time
        model_config = METRIC_LEARNING_MODELS.get(model_name, {})
        pair_type = model_config.get('pair_type', 'triplets')
        model_learning_ms = 0.0
        vector_fetch_ms = 0.0

        dissimilar_pool = negatives.copy()
        random.shuffle(dissimilar_pool)

        for chosen_dissimilar_idx in dissimilar_pool:
            dissimilar_vector, t_fetch_ms = TripletGenerator._timed_fetch_vector(dataset, chosen_dissimilar_idx)
            vector_fetch_ms += t_fetch_ms

            if batch_mode:
                # Batch update with all similar images
                if pair_type == 'triplets':
                    training_data = []
                    for sim_idx in positives:
                        similar_vector, t_fetch_ms = TripletGenerator._timed_fetch_vector(dataset, sim_idx)
                        vector_fetch_ms += t_fetch_ms
                        training_data.append([query_vector, similar_vector, dissimilar_vector])
                    t_model = time.perf_counter()
                    M = MetricLearner._apply_batch_triplet_update(M, model_name, model_params, training_data)
                    model_learning_ms += (time.perf_counter() - t_model) * 1000.0
                else:  # pairs
                    pairs = []
                    labels = []
                    for sim_idx in positives:
                        similar_vector, t_fetch_ms = TripletGenerator._timed_fetch_vector(dataset, sim_idx)
                        vector_fetch_ms += t_fetch_ms
                        pairs.append((query_vector, similar_vector))
                        labels.append(1)
                        pairs.append((query_vector, dissimilar_vector))
                        labels.append(-1)
                    t_model = time.perf_counter()
                    M = MetricLearner._apply_batch_pair_update(M, model_name, model_params, pairs, labels)
                    model_learning_ms += (time.perf_counter() - t_model) * 1000.0
            else:
                # Individual updates
                for sim_idx in positives:
                    similar_vector, t_fetch_ms = TripletGenerator._timed_fetch_vector(dataset, sim_idx)
                    vector_fetch_ms += t_fetch_ms
                    t_model = time.perf_counter()
                    M = MetricLearner._apply_single_update(
                        M, model_name, model_params, pair_type,
                        query_vector, similar_vector, dissimilar_vector
                    )
                    model_learning_ms += (time.perf_counter() - t_model) * 1000.0

        if return_timing:
            return M, model_learning_ms, vector_fetch_ms
        return M

    @staticmethod
    def apply_feedback_type_3(M, query_vector, positives, negatives, dataset, model_name, model_params,
                              return_timing=False):
        """
        Feedback Mechanism 3: All Possible Pairs (from Section 4.2 of the paper).

        All negative images are paired with all positive images, generating the
        complete set of triplets for comprehensive learning. This creates a larger
        set of triplets that is fed to the model's learning process as a batch.

        Args:
            M: Current metric matrix
            query_vector: Query vector
            positives: List of positive example indices
            negatives: List of negative example indices
            dataset: Full dataset for vector lookup
            model_name: Name of metric learning model
            model_params: Model hyperparameters

        Returns:
            Updated metric matrix, or (matrix, model_learning_ms, vector_fetch_ms)
            when return_timing=True
        """
        import time
        model_config = METRIC_LEARNING_MODELS.get(model_name, {})
        pair_type = model_config.get('pair_type', 'triplets')
        model_learning_ms = 0.0
        vector_fetch_ms = 0.0

        # Create all combinations
        if pair_type == 'triplets':
            training_data = []
            for dis_idx in negatives:
                dissimilar_vector, t_fetch_ms = TripletGenerator._timed_fetch_vector(dataset, dis_idx)
                vector_fetch_ms += t_fetch_ms
                for sim_idx in positives:
                    similar_vector, t_fetch_ms = TripletGenerator._timed_fetch_vector(dataset, sim_idx)
                    vector_fetch_ms += t_fetch_ms
                    training_data.append([query_vector, similar_vector, dissimilar_vector])
            t_model = time.perf_counter()
            M = MetricLearner._apply_batch_triplet_update(M, model_name, model_params, training_data)
            model_learning_ms += (time.perf_counter() - t_model) * 1000.0
        else:  # pairs
            pairs = []
            labels = []
            for dis_idx in negatives:
                dissimilar_vector, t_fetch_ms = TripletGenerator._timed_fetch_vector(dataset, dis_idx)
                vector_fetch_ms += t_fetch_ms
                for sim_idx in positives:
                    similar_vector, t_fetch_ms = TripletGenerator._timed_fetch_vector(dataset, sim_idx)
                    vector_fetch_ms += t_fetch_ms
                    pairs.append((query_vector, similar_vector))
                    labels.append(1)
                    pairs.append((query_vector, dissimilar_vector))
                    labels.append(-1)
            t_model = time.perf_counter()
            M = MetricLearner._apply_batch_pair_update(M, model_name, model_params, pairs, labels)
            model_learning_ms += (time.perf_counter() - t_model) * 1000.0

        if return_timing:
            return M, model_learning_ms, vector_fetch_ms
        return M
