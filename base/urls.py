from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('load-dataset/', views.load_dataset_view, name='load_dataset'),
    path('load-clip/', views.load_clip_model, name='load_clip_model'),
    path('api/search/', views.search_api, name='search_api'),
    path('api/text-search/', views.text_search_api, name='text_search_api'),
    path('api/mahalanobis-search/', views.mahalanobis_search_api, name='mahalanobis_search_api'),
    path('search-by-image/', views.image_upload_search, name='image_upload_search'),
    path('images/<path:image_path>', views.serve_image, name='serve_image'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),

    # Feedback and Metric Learning endpoints
    path('api/feedback/save/', views.save_feedback, name='save_feedback'),
    path('api/feedback/session/', views.get_session_feedback, name='get_session_feedback'),
    path('api/feedback/apply/', views.apply_feedback_learning, name='apply_feedback_learning'),
    path('api/feedback/apply/status/<uuid:job_id>/', views.feedback_apply_job_status, name='feedback_apply_job_status'),
    path('api/matrix/reset/', views.reset_metric_matrix, name='reset_metric_matrix'),
    path('api/matrix/download/', views.download_current_matrix, name='download_current_matrix'),
    path('api/settings/update/', views.update_learning_settings, name='update_learning_settings'),
    path('api/settings/get/', views.get_learning_settings, name='get_learning_settings'),
    path('api/settings/available/', views.get_available_settings, name='get_available_settings'),
    path('api/startup-status/', views.startup_status_api, name='startup_status_api'),
    path('api/query/rerun/', views.rerun_query, name='rerun_query'),
]
