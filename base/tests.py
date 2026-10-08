import json
import os
import tempfile
import zipfile
from unittest.mock import patch

import numpy as np
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from . import views
from . import config
from . import image_urls
from .image_urls import (
    build_image_url,
    build_image_url_template,
    expanded_result_count,
    filter_available_image_items,
)
from .models import User, UserMetricMatrix
from .services import ImageSimilarityService


class ImageUrlTests(SimpleTestCase):
    def test_disa_url_preserves_prefix_folder_and_filename(self):
        with (
            patch.object(config, 'USE_DISA_PROFIMEDIA', True),
            patch.object(config, 'DISA_BASE_URL', 'https://primary.example.test/app/'),
        ):
            image_url = build_image_url('558/0077109556.jpg')

        self.assertEqual(
            image_url,
            'https://primary.example.test/app/images/558/0077109556.jpg',
        )

    def test_disa_url_supports_other_prefix_folders(self):
        with (
            patch.object(config, 'USE_DISA_PROFIMEDIA', True),
            patch.object(config, 'DISA_BASE_URL', 'https://primary.example.test/app'),
        ):
            image_url = build_image_url('721/0012345678.jpg')

        self.assertEqual(
            image_url,
            'https://primary.example.test/app/images/721/0012345678.jpg',
        )

    def test_alternative_url_uses_filename_stem_twice(self):
        with (
            patch.object(config, 'USE_DISA_PROFIMEDIA', False),
            patch.object(
                config,
                'ALTERNATIVE_IMAGES_BASE_URL',
                'https://images.example.test/large/1/3/',
            ),
        ):
            image_url = build_image_url('558/0077109556.jpg')

        self.assertEqual(
            image_url,
            'https://images.example.test/large/1/3/0077109556/'
            'profimedia-0077109556.jpg',
        )

    def test_browser_template_matches_selected_url_shape(self):
        with (
            patch.object(config, 'USE_DISA_PROFIMEDIA', False),
            patch.object(
                config,
                'ALTERNATIVE_IMAGES_BASE_URL',
                'https://images.example.test/large/1/3',
            ),
        ):
            image_url_template = build_image_url_template()

        self.assertEqual(
            image_url_template,
            'https://images.example.test/large/1/3/__STEM__/'
            'profimedia-__FILENAME__',
        )

    def test_alternative_mode_expands_requested_count_ten_times(self):
        with patch.object(config, 'USE_DISA_PROFIMEDIA', False):
            self.assertEqual(expanded_result_count(20), 200)
            self.assertEqual(expanded_result_count(20, available_count=125), 125)

    def test_disa_mode_does_not_expand_requested_count(self):
        with patch.object(config, 'USE_DISA_PROFIMEDIA', True):
            self.assertEqual(expanded_result_count(20), 20)

    def test_alternative_filter_keeps_first_successful_images_in_order(self):
        items = [
            {'index': index, 'image_name': f'558/{index:010d}.jpg'}
            for index in range(1, 5)
        ]

        def is_available(image_url):
            return any(image_id in image_url for image_id in ('0000000002', '0000000004'))

        with (
            patch.object(config, 'USE_DISA_PROFIMEDIA', False),
            patch.object(
                config,
                'ALTERNATIVE_IMAGES_BASE_URL',
                'https://images.example.test/large/1/3/',
            ),
            patch.object(image_urls, 'IMAGE_PROBE_WORKERS', 4),
            patch.object(image_urls, '_image_url_is_available', side_effect=is_available),
        ):
            filtered_items = filter_available_image_items(items, requested_count=2)

        self.assertEqual([item['index'] for item in filtered_items], [2, 4])

    def test_disa_filter_does_not_probe_remote_images(self):
        items = [
            {'index': 1, 'image_name': '558/0000000001.jpg'},
            {'index': 2, 'image_name': '558/0000000002.jpg'},
            {'index': 3, 'image_name': '558/0000000003.jpg'},
        ]

        with (
            patch.object(config, 'USE_DISA_PROFIMEDIA', True),
            patch.object(image_urls, '_image_url_is_available') as availability_mock,
        ):
            filtered_items = filter_available_image_items(items, requested_count=2)

        self.assertEqual([item['index'] for item in filtered_items], [1, 2])
        availability_mock.assert_not_called()

    def test_image_probe_rejects_not_found_and_server_error_responses(self):
        for status in (404, 500):
            with self.subTest(status=status):
                error = image_urls.HTTPError(
                    'https://images.example.test/image.jpg',
                    status,
                    'image unavailable',
                    hdrs=None,
                    fp=None,
                )
                with patch.object(image_urls, 'urlopen', side_effect=error):
                    is_available = image_urls._image_url_is_available(
                        'https://images.example.test/image.jpg'
                    )

                self.assertFalse(is_available)


class MatrixDownloadTests(TestCase):
    def setUp(self):
        self.user = User.objects.create(username='matrix-user', password='unused')
        self.user_matrix = UserMetricMatrix(user=self.user, scaling_factor=1.5)
        self.user_matrix.set_matrix(np.asarray([[1.0, 0.0], [0.0, 2.0]]))
        self.user_matrix.save()

        session = self.client.session
        session['user_id'] = self.user.id
        session.save()

    def test_download_contains_the_complete_current_matrix_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            matrix_folder = os.path.join(temp_dir, 'matrix_3')
            os.makedirs(os.path.join(matrix_folder, 'extra'))
            with open(os.path.join(matrix_folder, 'matrix.npy'), 'wb') as matrix_file:
                np.save(matrix_file, self.user_matrix.get_matrix())
            with open(os.path.join(matrix_folder, 'matrix_metadata.json'), 'w') as metadata_file:
                json.dump({'scaling_factor': 1.5}, metadata_file)
            with open(os.path.join(matrix_folder, 'queries.json'), 'w') as queries_file:
                json.dump([{'query': 'red car'}], queries_file)
            with open(os.path.join(matrix_folder, 'extra', 'notes.txt'), 'w') as notes_file:
                notes_file.write('kept in export')

            with patch.object(
                views.QueryLogger,
                'save_matrix_state',
                return_value=matrix_folder,
            ) as save_mock:
                response = self.client.get(reverse('download_current_matrix'))
                archive_bytes = b''.join(response.streaming_content)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/zip')
        self.assertIn('matrix-user_matrix_3.zip', response['Content-Disposition'])
        self.assertEqual(response['Cache-Control'], 'no-store')
        save_mock.assert_called_once_with(self.user, self.user_matrix, force=True)

        with tempfile.SpooledTemporaryFile() as archive_file:
            archive_file.write(archive_bytes)
            archive_file.seek(0)
            with zipfile.ZipFile(archive_file) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {
                        'matrix_3/matrix.npy',
                        'matrix_3/matrix_metadata.json',
                        'matrix_3/queries.json',
                        'matrix_3/extra/notes.txt',
                    },
                )

    def test_download_requires_a_logged_in_user(self):
        self.client.session.flush()

        response = self.client.get(reverse('download_current_matrix'))

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()['error'], 'User not logged in')


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
            patch.object(
                views.clip_service,
                'text_to_vector',
                return_value=np.asarray([0.5, 0.5]),
            ) as clip_mock,
            patch.object(
                self.service,
                'search_with_filter_refine',
                return_value=search_payload,
            ) as search_mock,
            patch.object(self.service, 'get_vectors_by_indices', return_value=vectors) as vectors_mock,
        ):
            response = self.post_json({
                'query_text': 'red car',
                'distance_metric': 'cosine',
                'metric_matrix': [[0.25, 0.0], [0.0, 1.0]],
            })

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body['success'])
        self.assertEqual(body['requested_result_count'], 100)
        self.assertEqual(body['result_count'], 2)
        self.assertEqual(body['query']['embedding'], [0.5, 0.5])
        self.assertNotIn('query_embedding', body)
        self.assertEqual(body['metric']['candidate_distance_metric'], 'cosine')
        self.assertEqual(body['metric']['distance_mode'], 'dot_product')
        self.assertEqual(body['metric']['scaling_factor'], 2.0)
        self.assertEqual(body['search_mode'], 'filter_and_refine')
        self.assertEqual([item['distance'] for item in body['results']], [0.2, 0.8])
        self.assertEqual([item['rank'] for item in body['results']], [1, 2])
        self.assertEqual([item['position'] for item in body['results']], [1, 2])
        self.assertEqual(body['results'][0]['embedding'], [1.0, 0.0])
        self.assertIn('/images/folder/one.jpg', body['results'][0]['image_url'])
        clip_mock.assert_called_once_with('red car', normalize=True)
        self.assertEqual(search_mock.call_args.kwargs['num_results'], 100)
        self.assertEqual(search_mock.call_args.kwargs['distance_mode'], 'dot_product')
        self.assertEqual(search_mock.call_args.kwargs['scaling_factor'], 2.0)
        vectors_mock.assert_called_once_with([0, 1], distance_metric='cosine')

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
            patch.object(
                self.service,
                'search_with_filter_refine',
                return_value=search_payload,
            ) as search_mock,
            patch.object(
                self.service,
                'get_vectors_by_indices',
                return_value=np.asarray([[1.0, 0.0]], dtype=np.float32),
            ),
        ):
            response = self.post_json({
                'query_image_index': 1,
                'distance_metric': 'euclidean',
                'metric_matrix': [[2.0, 0.0], [0.0, 0.25]],
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['query']['type'], 'image')
        self.assertEqual(response.json()['query']['image_index'], 1)
        self.assertEqual(response.json()['query']['image_name'], 'folder/two.jpg')
        self.assertEqual(response.json()['query']['embedding'], [0.0, 1.0])
        self.assertEqual(response.json()['metric']['candidate_distance_metric'], 'euclidean')
        image_mock.assert_called_once_with(1, distance_metric='euclidean')
        self.assertEqual(search_mock.call_args.kwargs['distance_mode'], 'euclidean')
        self.assertEqual(search_mock.call_args.kwargs['scaling_factor'], 2.0)

    def test_identity_matrix_uses_the_ui_base_knn_path_for_both_metrics(self):
        search_payload = {
            'results': [
                {'index': 0, 'image_name': 'folder/one.jpg', 'distance': 0.125},
            ]
        }

        for distance_metric, normalize in (('cosine', True), ('euclidean', False)):
            with self.subTest(distance_metric=distance_metric):
                with (
                    patch.object(
                        views.clip_service,
                        'text_to_vector',
                        return_value=np.asarray([0.5, 0.5]),
                    ) as clip_mock,
                    patch.object(
                        self.service,
                        'search_by_vector',
                        return_value=search_payload,
                    ) as search_mock,
                    patch.object(
                        self.service,
                        'search_with_filter_refine',
                    ) as refine_mock,
                    patch.object(
                        self.service,
                        'get_vectors_by_indices',
                        return_value=np.asarray([[1.0, 0.0]], dtype=np.float32),
                    ),
                ):
                    response = self.post_json({
                        'query_text': 'red car',
                        'distance_metric': distance_metric,
                        'metric_matrix': [[1.0, 0.0], [0.0, 1.0]],
                    })

                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertEqual(body['metric']['name'], distance_metric)
                self.assertFalse(body['metric']['matrix_applied'])
                self.assertEqual(body['search_mode'], 'base_knn')
                self.assertIsNone(body['progressive_stage'])
                self.assertEqual(body['results'][0]['distance'], 0.125)
                clip_mock.assert_called_once_with('red car', normalize=normalize)
                self.assertEqual(
                    search_mock.call_args.kwargs['distance_metric'],
                    distance_metric,
                )
                refine_mock.assert_not_called()

    def test_exactly_one_query_source_is_required(self):
        response = self.post_json({
            'query_text': 'red car',
            'query_image_index': 1,
            'distance_metric': 'euclidean',
            'metric_matrix': [[1.0, 0.0], [0.0, 1.0]],
        })

        self.assertEqual(response.status_code, 400)
        self.assertIn('exactly one', response.json()['error'])

    def test_matrix_must_match_dataset_dimension(self):
        response = self.post_json({
            'query_text': 'red car',
            'distance_metric': 'euclidean',
            'metric_matrix': [[1.0]],
        })

        self.assertEqual(response.status_code, 400)
        self.assertIn('shape (2, 2)', response.json()['error'])

    def test_matrix_must_be_positive_semidefinite(self):
        response = self.post_json({
            'query_text': 'red car',
            'distance_metric': 'euclidean',
            'metric_matrix': [[1.0, 0.0], [0.0, -1.0]],
        })

        self.assertEqual(response.status_code, 400)
        self.assertIn('positive semidefinite', response.json()['error'])

    def test_distance_metric_is_required(self):
        response = self.post_json({
            'query_text': 'red car',
            'metric_matrix': [[1.0, 0.0], [0.0, 1.0]],
        })

        self.assertEqual(response.status_code, 400)
        self.assertIn('distance_metric is required', response.json()['error'])

    def test_distance_metric_must_be_supported(self):
        response = self.post_json({
            'query_text': 'red car',
            'distance_metric': 'manhattan',
            'metric_matrix': [[1.0, 0.0], [0.0, 1.0]],
        })

        self.assertEqual(response.status_code, 400)
        self.assertIn("distance_metric must be 'euclidean'", response.json()['error'])


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

    def test_faiss_cosine_candidates_use_the_ip_index(self):
        service = ImageSimilarityService()
        service.dataset_loaded = True
        service.faiss_enabled = True
        service.faiss_l2_index = None
        service.faiss_ip_index = object()
        service.image_names = ['one.jpg', 'two.jpg']
        service.image_names_ip = service.image_names
        service.vector_dim = 2

        with (
            patch.object(
                service,
                'nearest_indices_cosine',
                return_value=(np.asarray([0, 1]), np.asarray([0.0, 1.0])),
            ) as cosine_mock,
            patch.object(
                service,
                '_get_vectors_by_indices',
                return_value=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            ) as vectors_mock,
        ):
            service.nearest_indices_mahalanobis(
                anchor_vector=np.asarray([0.0, 0.0]),
                metric_matrix=np.eye(2),
                num_indices=2,
                base_metric='cosine',
            )

        cosine_mock.assert_called_once()
        self.assertEqual(vectors_mock.call_args.kwargs['distance_metric'], 'cosine')
