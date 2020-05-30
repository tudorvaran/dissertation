import init # NOQA, initialize django APP

from photos_ml.object_detection_tasks import object_detection, sync_picture_directory  # NOQA

sync_picture_directory()
object_detection()
