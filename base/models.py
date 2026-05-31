from django.db import models
from django.contrib.auth.hashers import make_password, check_password
import pickle
import uuid
from .matrix_cache import get_cached_matrix_for_user_matrix, store_cached_matrix_for_user_matrix

# Create your models here.

class User(models.Model):
    username = models.CharField(max_length=150, unique=True)
    password = models.CharField(max_length=255)
    email = models.EmailField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.username

    def set_password(self, raw_password):
        self.password = make_password(raw_password)

    def check_password(self, raw_password):
        return check_password(raw_password, self.password)


class UserMetricMatrix(models.Model):
    """Store the metric matrix for each user."""
    DISTANCE_MODE_CHOICES = [
        ('euclidean', 'Euclidean'),
        ('dot_product', 'Dot Product'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='metric_matrix')
    matrix_data = models.BinaryField()  # Pickled numpy array
    matrix_dimension = models.IntegerField(default=768)
    scaling_factor = models.FloatField(default=1.0)
    updated_at = models.DateTimeField(auto_now=True)

    # Settings for metric learning
    metric_learning_model = models.CharField(max_length=50, default='OMDML')
    model_hyperparameters = models.JSONField(default=dict)  # Store hyperparameters as JSON
    feedback_type = models.IntegerField(default=2)  # 1, 2, or 3
    feedback_hyperparameters = models.JSONField(default=dict)

    # Filter-and-refine settings
    distance_mode = models.CharField(
        max_length=20,
        choices=DISTANCE_MODE_CHOICES,
        default='euclidean',
        help_text='Distance mode for filter-and-refine search'
    )

    def set_matrix(self, matrix):
        """Store a numpy array as pickled binary data."""
        self.matrix_data = pickle.dumps(matrix)
        self.matrix_dimension = matrix.shape[0]

    def get_matrix(self):
        """Retrieve the numpy array from pickled binary data."""
        if self.pk:
            cached = get_cached_matrix_for_user_matrix(self)
            if cached is not None:
                return cached
        if self.matrix_data:
            matrix = pickle.loads(self.matrix_data)
            if self.pk:
                store_cached_matrix_for_user_matrix(self, matrix)
            return matrix
        # Return identity matrix if no matrix is stored
        import numpy as np
        return np.eye(self.matrix_dimension)

    def reset_to_identity(self):
        """Reset the matrix to identity matrix."""
        import numpy as np
        identity = np.eye(self.matrix_dimension)
        self.set_matrix(identity)
        self.scaling_factor = 1.0
        self.save()
        store_cached_matrix_for_user_matrix(self, identity)

    def __str__(self):
        return f"Metric matrix for {self.user.username}"


class FeedbackItem(models.Model):
    """Store user feedback on image relevance for a query."""
    FEEDBACK_CHOICES = [
        ('positive', 'Relevant'),
        ('negative', 'Irrelevant'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='feedbacks')
    query_type = models.CharField(max_length=20)  # 'text', 'image', 'uploaded'
    query_text = models.TextField(blank=True, null=True)
    query_image_index = models.IntegerField(blank=True, null=True)
    query_vector = models.BinaryField(blank=True, null=True)  # Pickled numpy array

    result_image_index = models.IntegerField()
    result_image_name = models.CharField(max_length=500)
    feedback_type = models.CharField(max_length=10, choices=FEEDBACK_CHOICES)

    session_id = models.CharField(max_length=100)  # To group feedback from same search session
    created_at = models.DateTimeField(auto_now_add=True)

    def set_query_vector(self, vector):
        """Store query vector as pickled binary data."""
        self.query_vector = pickle.dumps(vector)

    def get_query_vector(self):
        """Retrieve the query vector."""
        if self.query_vector:
            return pickle.loads(self.query_vector)
        return None

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'session_id']),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.feedback_type} feedback on {self.result_image_name}"


class FeedbackApplyJob(models.Model):
    """Track asynchronous feedback-application jobs across Gunicorn workers."""

    STATUS_PENDING = 'pending'
    STATUS_RUNNING = 'running'
    STATUS_COMPLETED = 'completed'
    STATUS_FAILED = 'failed'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_RUNNING, 'Running'),
        (STATUS_COMPLETED, 'Completed'),
        (STATUS_FAILED, 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='feedback_apply_jobs')
    session_id = models.CharField(max_length=100, blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    result = models.JSONField(default=dict, blank=True)
    error = models.TextField(blank=True)
    error_type = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(blank=True, null=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['user', 'session_id', 'status']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"Feedback apply job {self.id} ({self.status})"
