"""Build public image URLs from image names stored in the search indexes."""

from urllib.parse import quote

from . import config


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
