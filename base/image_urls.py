"""Build public image URLs from image names stored in the search indexes."""

from concurrent.futures import ThreadPoolExecutor
import socket
from urllib.parse import quote
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import config


ALTERNATIVE_RESULT_MULTIPLIER = 10
IMAGE_PROBE_TIMEOUT_SECONDS = 4
IMAGE_PROBE_WORKERS = 20


def _join_url(base_url, relative_path):
    base_url = str(base_url or '').strip().rstrip('/')
    if not base_url:
        return f'/{relative_path.lstrip("/")}'
    return f'{base_url}/{relative_path.lstrip("/")}'


def _parse_indexed_image_name(image_name):
    normalized = str(image_name or '').strip().replace('\\', '/').lstrip('/')
    parts = [part for part in normalized.split('/') if part and part != '.']
    if parts and parts[0] == 'images':
        parts = parts[1:]

    if len(parts) < 2 or '..' in parts:
        raise ValueError(f'Invalid indexed image name: {image_name!r}')

    prefix_folder = parts[0]
    filename = parts[-1]
    filename_stem, separator, extension = filename.rpartition('.')
    if not separator or not filename_stem or not extension:
        raise ValueError(f'Indexed image filename has no extension: {image_name!r}')

    return prefix_folder, filename, filename_stem


def build_image_url(image_name):
    """Return the configured public URL for an indexed image name."""
    prefix_folder, filename, filename_stem = _parse_indexed_image_name(image_name)

    if config.USE_DISA_PROFIMEDIA:
        relative_path = 'images/{}/{}'.format(
            quote(prefix_folder, safe=''),
            quote(filename, safe=''),
        )
        return _join_url(config.DISA_BASE_URL, relative_path)

    if not config.ALTERNATIVE_IMAGES_BASE_URL:
        raise ValueError('ALTERNATIVE_IMAGES_BASE_URL must be set when DISA images are disabled')

    relative_path = '{}/profimedia-{}'.format(
        quote(filename_stem, safe=''),
        quote(filename, safe=''),
    )
    return _join_url(config.ALTERNATIVE_IMAGES_BASE_URL, relative_path)


def build_image_url_template():
    """Return a browser-side URL template using encoded replacement tokens."""
    if config.USE_DISA_PROFIMEDIA:
        return _join_url(
            config.DISA_BASE_URL,
            'images/__FOLDER__/__FILENAME__',
        )

    if not config.ALTERNATIVE_IMAGES_BASE_URL:
        raise ValueError('ALTERNATIVE_IMAGES_BASE_URL must be set when DISA images are disabled')

    return _join_url(
        config.ALTERNATIVE_IMAGES_BASE_URL,
        '__STEM__/profimedia-__FILENAME__',
    )


def expanded_result_count(requested_count, available_count=None):
    """Return the retrieval count needed to replace unavailable alternatives."""
    requested_count = max(0, int(requested_count))
    if config.USE_DISA_PROFIMEDIA:
        return requested_count

    expanded_count = requested_count * ALTERNATIVE_RESULT_MULTIPLIER
    if available_count is not None:
        expanded_count = min(expanded_count, max(0, int(available_count)))
    return expanded_count


def _image_url_is_available(image_url):
    """Probe an image URL using a ranged GET so its real response path is used."""
    request = Request(
        image_url,
        headers={
            'Accept': 'image/*,*/*;q=0.8',
            'Range': 'bytes=0-0',
            'User-Agent': 'Interactive-Personalised-Image-Search/1.0',
        },
        method='GET',
    )
    try:
        with urlopen(request, timeout=IMAGE_PROBE_TIMEOUT_SECONDS) as response:
            status = int(getattr(response, 'status', response.getcode()))
            return 200 <= status < 400
    except (HTTPError, URLError, TimeoutError, socket.timeout, OSError, ValueError):
        return False


def filter_available_image_items(items, requested_count):
    """Preserve order while returning only successful alternative image URLs."""
    requested_count = max(0, int(requested_count))
    candidates = list(items or [])
    if requested_count == 0:
        return []
    if config.USE_DISA_PROFIMEDIA:
        return candidates[:requested_count]

    candidates = candidates[:requested_count * ALTERNATIVE_RESULT_MULTIPLIER]
    available_items = []
    batch_size = max(1, min(IMAGE_PROBE_WORKERS, len(candidates)))

    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start:batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            availability = list(executor.map(
                lambda item: _image_url_is_available(build_image_url(item.get('image_name'))),
                batch,
            ))

        for item, is_available in zip(batch, availability):
            if is_available:
                available_items.append(item)
                if len(available_items) >= requested_count:
                    return available_items

    return available_items
