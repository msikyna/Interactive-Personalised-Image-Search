from django import template

from ..image_urls import build_image_url, build_image_url_template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Get an item from a dictionary by key."""
    if dictionary is None:
        return None
    return dictionary.get(key)


@register.filter
def image_source_url(image_name):
    """Build the configured public URL for an indexed image."""
    return build_image_url(image_name)


@register.simple_tag
def image_source_url_template():
    """Build the configured URL template used by dynamic result cards."""
    return build_image_url_template()
