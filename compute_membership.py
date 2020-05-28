import init # NOQA, initialize django APP

from photos_ml.object_detection_tasks import build_vectors, sync_picture_directory  # NOQA

sync_picture_directory()
build_vectors()
