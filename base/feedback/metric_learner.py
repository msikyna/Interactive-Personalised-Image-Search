"""
Metric learning model operations.
Handles instantiation and execution of metric learning algorithms.
"""

import numpy as np
from .constants import (
    OASIS, OMDML, SORS, AdaSORS, RobustODML,
    FactoredAROMA, OPML, MLOML, POLA
)


class MetricLearner:
    """Handles metric learning model operations."""

    @staticmethod
    def _apply_single_update(M, model_name, model_params, pair_type, query_vec, pos_vec, neg_vec):
        """
        Apply a single metric learning update based on the model.

        Args:
            M: Current metric matrix
            model_name: Name of the model to use
            model_params: Parameters for the model
            pair_type: 'triplets' or 'pairs'
            query_vec: Query vector
            pos_vec: Positive example vector
            neg_vec: Negative example vector

        Returns:
            Updated metric matrix
        """
        dim = M.shape[0]
        params = model_params.copy()

        if pair_type == 'triplets':
            training_data = [[query_vec, pos_vec, neg_vec]]

            if model_name == 'OASIS':
                oasis_model = OASIS(**params)
                oasis_model.partial_fit(M, training_data)
                M = oasis_model.get_matrix()
            elif model_name == 'SORS':
                params['M_init'] = M
                sors_model = SORS(**params)
                sors_model.fit(training_data)
                M = sors_model.get_mahalanobis_matrix()
            elif model_name == 'AdaSORS':
                params['M_init'] = M
                adasors_model = AdaSORS(**params)
                adasors_model.fit(training_data)
                M = adasors_model.get_mahalanobis_matrix()
            elif model_name == 'OPML':
                params['initial_matrix'] = M
                opml_model = OPML(**params)
                opml_model.fit(training_data)
                M = opml_model.get_mahalanobis_matrix()

        else:  # pairs
            pairs = [(query_vec, pos_vec), (query_vec, neg_vec)]
            labels = [1, -1]

            if model_name == 'POLA':
                params['dim'] = dim
                pola_model = POLA(**params)
                pola_model.set_mahalanobis_matrix(M)
                pola_model.fit(pairs, labels)
                M = pola_model.get_mahalanobis_matrix()
            elif model_name == 'FactoredAROMA':
                params['m'] = dim
                params['n'] = dim
                params['initial_W'] = M
                aroma_model = FactoredAROMA(**params)
                aroma_model.fit(pairs, labels)
                M = aroma_model.get_mahalanobis_matrix()
            elif model_name == 'OMDML':
                params['prior'] = [M]
                omdml_model = OMDML(**params)
                omdml_model.fit(pairs, labels)
                M = omdml_model.get_mahalanobis_matrix()
            elif model_name == 'MLOML':
                params['d'] = dim
                mloml_model = MLOML(**params)
                mloml_model.set_initial_matrix(M)
                mloml_model.fit(pairs, labels)
                M = mloml_model.get_mahalanobis_matrix()
            elif model_name == 'RobustODML':
                params['init_matrix'] = M
                robust_odml_model = RobustODML(**params)
                robust_odml_model.fit(pairs, labels)
                M = robust_odml_model.get_mahalanobis_matrix()

        return M

    @staticmethod
    def _apply_batch_triplet_update(M, model_name, model_params, training_data):
        """
        Apply batch triplet update.

        Args:
            M: Current metric matrix
            model_name: Name of the model to use
            model_params: Parameters for the model
            training_data: List of triplets [query, positive, negative]

        Returns:
            Updated metric matrix
        """
        dim = M.shape[0]
        params = model_params.copy()

        if model_name == 'OASIS':
            oasis_model = OASIS(**params)
            oasis_model.partial_fit(M, training_data)
            M = oasis_model.get_matrix()
        elif model_name == 'SORS':
            params['M_init'] = M
            sors_model = SORS(**params)
            sors_model.fit(training_data)
            M = sors_model.get_mahalanobis_matrix()
        elif model_name == 'AdaSORS':
            params['M_init'] = M
            adasors_model = AdaSORS(**params)
            adasors_model.fit(training_data)
            M = adasors_model.get_mahalanobis_matrix()
        elif model_name == 'OPML':
            params['initial_matrix'] = M
            opml_model = OPML(**params)
            opml_model.fit(training_data)
            M = opml_model.get_mahalanobis_matrix()

        return M

    @staticmethod
    def _apply_batch_pair_update(M, model_name, model_params, pairs, labels):
        """
        Apply batch pair update.

        Args:
            M: Current metric matrix
            model_name: Name of the model to use
            model_params: Parameters for the model
            pairs: List of (vector1, vector2) tuples
            labels: List of labels (1 for similar, -1 for dissimilar)

        Returns:
            Updated metric matrix
        """
        dim = M.shape[0]
        params = model_params.copy()

        if model_name == 'POLA':
            params['dim'] = dim
            pola_model = POLA(**params)
            pola_model.set_mahalanobis_matrix(M)
            pola_model.fit(pairs, labels)
            M = pola_model.get_mahalanobis_matrix()
        elif model_name == 'FactoredAROMA':
            params['m'] = dim
            params['n'] = dim
            params['initial_W'] = M
            aroma_model = FactoredAROMA(**params)
            aroma_model.fit(pairs, labels)
            M = aroma_model.get_mahalanobis_matrix()
        elif model_name == 'OMDML':
            params['prior'] = [M]
            omdml_model = OMDML(**params)
            omdml_model.fit(pairs, labels)
            M = omdml_model.get_mahalanobis_matrix()
        elif model_name == 'MLOML':
            params['d'] = dim
            mloml_model = MLOML(**params)
            mloml_model.set_initial_matrix(M)
            mloml_model.fit(pairs, labels)
            M = mloml_model.get_mahalanobis_matrix()
        elif model_name == 'RobustODML':
            params['init_matrix'] = M
            robust_odml_model = RobustODML(**params)
            robust_odml_model.fit(pairs, labels)
            M = robust_odml_model.get_mahalanobis_matrix()

        return M
