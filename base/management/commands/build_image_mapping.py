import os

from django.core.management.base import BaseCommand, CommandError

from base import config
from base.services import ImageSimilarityService


class Command(BaseCommand):
    help = (
        "Build and persist image-name mapping for FAISS index mode. "
        "This avoids expensive image-directory scans on each startup."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--output',
            default='',
            help='Output mapping file path (default: SIMSEARCH_IMAGE_NAMES_CACHE).',
        )
        parser.add_argument(
            '--overwrite',
            action='store_true',
            help='Overwrite existing mapping file if it already exists.',
        )
        parser.add_argument(
            '--verify',
            action='store_true',
            help='Reload the saved mapping and verify the entry count.',
        )

    def handle(self, *args, **options):
        output = (options.get('output') or '').strip()
        if not output:
            output = (getattr(config, 'IMAGE_NAMES_CACHE_PATH', '') or '').strip()

        if not output:
            raise CommandError(
                "No output path provided. Use --output or set SIMSEARCH_IMAGE_NAMES_CACHE."
            )

        output = os.path.abspath(output)
        overwrite = bool(options.get('overwrite'))
        verify = bool(options.get('verify'))

        if os.path.exists(output) and not overwrite:
            self.stdout.write(
                self.style.WARNING(
                    f"Mapping file already exists: {output}. "
                    "Use --overwrite to rebuild."
                )
            )
            return

        service = ImageSimilarityService()

        self.stdout.write(
            f"Building image-name mapping by scanning image directory: "
            f"{service._resolve_image_base_path()}"
        )
        image_names = service._scan_image_names_from_directory()
        if not image_names:
            raise CommandError("No images found while building mapping.")

        service._save_image_names_cache(image_names, cache_path=output)
        self.stdout.write(
            self.style.SUCCESS(
                f"Saved mapping with {len(image_names)} entries to: {output}"
            )
        )

        if verify:
            loaded = service._load_image_names_from_file(output)
            if len(loaded) != len(image_names):
                raise CommandError(
                    f"Verification failed: saved={len(image_names)} loaded={len(loaded)}"
                )
            self.stdout.write(
                self.style.SUCCESS(
                    f"Verification OK: loaded {len(loaded)} entries from saved mapping."
                )
            )
