from django.apps import AppConfig

import importlib
import threading
import sys
import os

try:
    from . import startup_status  # type: ignore
except Exception:
    class _StartupStatusFallback:
        @staticmethod
        def start(*args, **kwargs):
            return None

        @staticmethod
        def log(*args, **kwargs):
            return None

        @staticmethod
        def mark_ready(*args, **kwargs):
            return None

        @staticmethod
        def mark_error(*args, **kwargs):
            return None

        @staticmethod
        def mark_disabled(*args, **kwargs):
            return None

    startup_status = _StartupStatusFallback()

class BaseConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'base'
    _warmup_lock = threading.Lock()
    _warmup_started = False

    @staticmethod
    def _warmup_services(image_similarity_service, clip_service, config):
        try:
            startup_status.start("Auto-load sequence started.")
            print("=" * 60)
            print("Auto-loading image similarity dataset...")
            startup_status.log("Auto-loading image similarity dataset...")
            image_names, dataset = image_similarity_service.load_dataset_from_files(
                config.DATASET_DIRECTORY,
                config.DATASET_FILES
            )
            print("Dataset loaded successfully!")
            print(f"Images: {len(image_names)}")
            print(f"Dimensions: {image_similarity_service.vector_dim}")
            print("=" * 60)
            startup_status.log(f"Dataset loaded successfully. Images={len(image_names)}, dimensions={image_similarity_service.vector_dim}")

            if config.AUTO_LOAD_CLIP_MODEL:
                print("=" * 60)
                print("Auto-loading CLIP model (ViT-L/14)...")
                startup_status.log("Auto-loading CLIP model (ViT-L/14)...")
                clip_service.load_model()
                print("CLIP model loaded successfully!")
                print("=" * 60)
                startup_status.log("CLIP model loaded successfully.")
            else:
                print("CLIP model auto-load disabled. It will be loaded on first use.")
                print("=" * 60)
                startup_status.log("CLIP model auto-load disabled.")
            startup_status.mark_ready("Startup finished successfully.")
        except Exception as e:
            print("=" * 60)
            print(f"Warning: Could not auto-load dataset/model: {e}")
            print("You can manually load it from the web interface.")
            print("=" * 60)
            startup_status.mark_error(f"Could not auto-load dataset/model: {e}")
            import traceback
            traceback.print_exc()

    def ready(self):
        """Load dataset and CLIP model automatically when Django starts."""
        try:
            importlib.import_module('base.db_optimizations')
        except Exception as exc:
            print(f"[WARN] Could not load DB optimizations module: {exc}", flush=True)

        argv0 = os.path.basename(sys.argv[0]) if sys.argv else ''
        command = sys.argv[1].strip().lower() if len(sys.argv) > 1 else ''

        if os.environ.get('SIMSEARCH_SKIP_APP_AUTOLOAD', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
            return

        # Do not trigger heavy startup warmup for management commands such as
        # migrate, shell, createsuperuser, etc. Those paths should stay fast
        # and must not depend on runtime log file permissions.
        if argv0 == 'manage.py' and command and command != 'runserver':
            return

        # Avoid duplicate load in Django runserver's reloader parent process.
        # Under Gunicorn RUN_MAIN is typically unset, so warmup should run.
        is_runserver = any(arg.endswith('runserver') for arg in sys.argv)
        if is_runserver and os.environ.get('RUN_MAIN') != 'true':
            return

        # Import services in main thread first to avoid a race with URL import.
        from .services import image_similarity_service
        from .clip_service import clip_service
        from . import config

        if not config.AUTO_LOAD_DATASET:
            print("Auto-load disabled. Load dataset manually from the web interface.")
            startup_status.mark_disabled("Auto-load disabled. Waiting for manual load.")
            return

        if getattr(config, 'AUTO_LOAD_BACKGROUND', True):
            with self._warmup_lock:
                if self._warmup_started:
                    return
                self._warmup_started = True

            warmup_thread = threading.Thread(
                target=self._warmup_services,
                args=(image_similarity_service, clip_service, config),
                name='simsearch-startup-warmup',
                daemon=True
            )
            warmup_thread.start()
            print("Auto-load started in background thread.")
            startup_status.log("Auto-load started in background thread.")
            return

        self._warmup_services(image_similarity_service, clip_service, config)
