import json
from unittest.mock import patch

import numpy as np
from django.test import SimpleTestCase
from django.urls import reverse

from . import views
from .services import ImageSimilarityService


class MahalanobisSearchApiTests(SimpleTestCase):
    def setUp(self):
        self.service = views.image_similarity_service
        self.original_state = {
            'dataset_loaded': self.service.dataset_loaded,
            'vector_dim': self.service.vector_dim,
            'faiss_enabled': self.service.faiss_enabled,
            'image_names': self.service.image_names,
        }
        self.service.dataset_loaded = True
        self.service.vector_dim = 2
        self.service.faiss_enabled = False
        self.service.image_names = ['folder/one.jpg', 'folder/two.jpg']

    def tearDown(self):
        for name, value in self.original_state.items():
            setattr(self.service, name, value)

    def post_json(self, payload):
        return self.client.post(
            reverse('mahalanobis_search_api'),
            data=json.dumps(payload),
            content_type='application/json',
        )

    def test_text_query_returns_ranked_results_with_embeddings(self):
        search_payload = {
            'results': [
                {'index': 1, 'image_name': 'folder/two.jpg', 'distance': 0.8},
                {'index': 0, 'image_name': 'folder/one.jpg', 'distance': 0.2},
            ]
        }
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        with (
            patch.object(views.clip_service, 'text_to_vector', return_value=np.asarray([0.5, 0.5])),
            patch.object(self.service, 'search_by_vector', return_value=search_payload) as search_mock,
            patch.object(self.service, 'get_vectors_by_indices', return_value=vectors) as vectors_mock,
        ):
            response = self.post_json({
                'query_text': 'red car',
                'metric_matrix': [[1.0, 0.0], [0.0, 1.0]],
            })

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body['success'])
        self.assertEqual(body['requested_result_count'], 100)
        self.assertEqual(body['result_count'], 2)
        self.assertEqual(body['query']['embedding'], [0.5, 0.5])
        self.assertEqual(body['query_embedding'], [0.5, 0.5])
        self.assertEqual(body['query_embedding'], [0.5, 0.5])
        self.assertEqual([item['distance'] for item in body['results']], [0.2, 0.8])
        self.assertEqual([item['rank'] for item in body['results']], [1, 2])
        self.assertEqual([item['position'] for item in body['results']], [1, 2])
        self.assertEqual(body['results'][0]['embedding'], [1.0, 0.0])
        self.assertIn('/images/folder/one.jpg', body['results'][0]['image_url'])
        self.assertEqual(search_mock.call_args.kwargs['num_results'], 100)
        self.assertEqual(search_mock.call_args.kwargs['distance_metric'], 'mahalanobis')
        vectors_mock.assert_called_once_with([0, 1], distance_metric='euclidean')

    def test_image_index_returns_the_dataset_image_as_query(self):
        search_payload = {
            'results': [
                {'index': 0, 'image_name': 'folder/one.jpg', 'distance': 0.0},
            ]
        }

        with (
            patch.object(
                self.service,
                'get_image_by_index',
                return_value=('folder/two.jpg', np.asarray([0.0, 1.0])),
            ) as image_mock,
            patch.object(self.service, 'search_by_vector', return_value=search_payload),
            patch.object(
                self.service,
                'get_vectors_by_indices',
                return_value=np.asarray([[1.0, 0.0]], dtype=np.float32),
            ),
        ):
            response = self.post_json({
                'query_image_index': 1,
                'metric_matrix': [[2.0, 0.0], [0.0, 1.0]],
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['query']['type'], 'image')
        self.assertEqual(response.json()['query']['image_index'], 1)
        self.assertEqual(response.json()['query']['image_name'], 'folder/two.jpg')
        self.assertEqual(response.json()['query']['embedding'], [0.0, 1.0])
        image_mock.assert_called_once_with(1, distance_metric='euclidean')

    def test_exactly_one_query_source_is_required(self):
        response = self.post_json({
            'query_text': 'red car',
            'query_image_index': 1,
            'metric_matrix': [[1.0, 0.0], [0.0, 1.0]],
        })

        self.assertEqual(response.status_code, 400)
        self.assertIn('exactly one', response.json()['error'])

    def test_matrix_must_match_dataset_dimension(self):
        response = self.post_json({
            'query_text': 'red car',
            'metric_matrix': [[1.0]],
        })

        self.assertEqual(response.status_code, 400)
        self.assertIn('shape (2, 2)', response.json()['error'])

    def test_matrix_must_be_positive_semidefinite(self):
        response = self.post_json({
            'query_text': 'red car',
            'metric_matrix': [[1.0, 0.0], [0.0, -1.0]],
        })

        self.assertEqual(response.status_code, 400)
        self.assertIn('positive semidefinite', response.json()['error'])


class MahalanobisServiceTests(SimpleTestCase):
    def test_faiss_candidate_vectors_use_the_available_l2_index(self):
        service = ImageSimilarityService()
        service.dataset_loaded = True
        service.faiss_enabled = True
        service.faiss_l2_index = object()
        service.image_names = ['one.jpg', 'two.jpg']
        service.vector_dim = 2

        with (
            patch.object(
                service,
                'nearest_indices_euclidean',
                return_value=(np.asarray([0, 1]), np.asarray([0.0, 1.0])),
            ),
            patch.object(
                service,
                '_get_vectors_by_indices',
                return_value=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            ) as vectors_mock,
        ):
            indices, distances = service.nearest_indices_mahalanobis(
                anchor_vector=np.asarray([0.0, 0.0]),
                metric_matrix=np.eye(2),
                num_indices=2,
            )

        self.assertEqual(indices.tolist(), [0, 1])
        self.assertEqual(distances.tolist(), [0.0, 1.0])
        vectors_mock.assert_called_once()
        self.assertEqual(vectors_mock.call_args.kwargs['distance_metric'], 'euclidean')
