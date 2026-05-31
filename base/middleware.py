import os
import time

from . import config
from .keepalive import ensure_keepalive_started


_PROCESS_PID = os.getpid()
_PROCESS_BOOT_TIME = time.time()


class SimSearchDebugHeadersMiddleware:
    """
    Optional lightweight diagnostics to confirm whether requests are hitting
    a cold/new worker or a process with unloaded services.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            ensure_keepalive_started()
        except Exception:
            pass

        response = self.get_response(request)
        if not getattr(config, 'DEBUG_RESPONSE_HEADERS', False):
            return response

        try:
            from .services import image_similarity_service
            from .clip_service import clip_service

            response['X-SimSearch-Pid'] = str(_PROCESS_PID)
            response['X-SimSearch-Process-Age-Sec'] = f"{time.time() - _PROCESS_BOOT_TIME:.1f}"
            response['X-SimSearch-Dataset-Loaded'] = '1' if image_similarity_service.dataset_loaded else '0'
            response['X-SimSearch-Clip-Loaded'] = '1' if clip_service.model_loaded else '0'
        except Exception:
            pass
        return response
