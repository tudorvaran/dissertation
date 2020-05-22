import os

import django

os.environ['DJANGO_SETTINGS_MODULE'] = 'dissertation.settings'
django.setup()

from photos_ml.tasks import build_vectors, sync_picture_directory  # NOQA

sync_picture_directory()
build_vectors(ignore_existing=True)
