import os

import django

os.environ['DJANGO_SETTINGS_MODULE'] = 'dissertation.settings'
django.setup()

from photos_ml.tasks import build_tree # NOQA

build_tree()
