import os
from datetime import datetime

import numpy as np
import structlog
from django.conf import settings
from django.db import transaction
from imageai.Detection import ObjectDetection

from photos_ml.models import Photo, EnvironmentFuzzyMembership, PhotoEnvironment, ObjectCategory, ObjectIdentification

logger = structlog.get_logger()


def sync_picture_directory():
    total_created = 0
    for photo_name in os.listdir(settings.IMAGE_PATH):
        if settings.IMAGE_LIMIT and Photo.objects.count() >= settings.IMAGE_LIMIT:
            break
        if photo_name.split('.')[1] in ['mp4', 'gif', 'DS_Store']:
            continue
        photo, was_created = Photo.objects.get_or_create(name=photo_name)
        if was_created:
            total_created += 1

    logger.info("Finished sync process", count=total_created, photo_count=Photo.objects.count())


def object_detection():
    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(settings.OBJECT_RECOGNITION_MODEL)
    detector.loadModel()

    done = 0
    total_photos = Photo.objects.count()
    t = datetime.now()

    for photo in Photo.objects.all().order_by('pk'):
        if not photo.exists(check_input=True):
            logger.warning("Path does not exist!", id=photo.id, name=photo.name)
            total_photos -= 1
            continue

        if photo.get_objects():
            total_photos -= 1
            continue

        ext = photo.name.split('.')[1]
        tmp_file = os.path.join(execution_path, f"photo.{ext}")
        new_tmp_file = os.path.join(execution_path, f"photo_out.{ext}")
        if os.path.islink(tmp_file):
            os.unlink(tmp_file)
        os.symlink(photo.get_input_path(), tmp_file,)

        detections = detector.detectObjectsFromImage(
            input_image=tmp_file,
            output_image_path=new_tmp_file
        )

        with transaction.atomic():
            for detection in detections:
                object_category, _ = ObjectCategory.objects.get_or_create(name=detection["name"])
                ObjectIdentification.objects.create(
                    photo=photo,
                    category=object_category,
                    confidence=detection["percentage_probability"]
                )

        os.unlink(tmp_file)
        if os.path.isfile(photo.get_output_path()):
            os.remove(photo.get_output_path())
        os.rename(new_tmp_file, photo.get_output_path())

        done += 1

        elapsed_time = (datetime.now() - t).seconds
        estimated_time_left = elapsed_time / done * (total_photos - done)
        stats = dict(
            name=photo.name,
            id=photo.pk,
            done=done,
            left=total_photos-done,
            proc=f'{"%.2f" % (done / total_photos * 100)}',
            est_left=f'{"%.2f" % estimated_time_left}s'
        )
        if not len(detections):
            logger.warning(
                "No objects found!",
                **stats
            )
        else:
            logger.info(
                "Detecting",
                **stats
            )

    logger.info("Finished image recognition")


def create_fuzzy_vectors():
    import gensim
    EnvironmentFuzzyMembership.objects.all().delete()

    logger.info("Loading word2vec model")
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(fname=settings.WORD2VEC_MODEL, binary=True)
    logger.info("Loaded word2vec model")

    for photo in Photo.objects.all():
        if not photo.exists(check_input=True):
            continue

        items = photo.get_objects()

        if not items:
            logger.warning("No objects detected in photo", id=photo.id, name=photo.name)
            continue

        detections = [item[0] for item in items]
        weights = [item[1] for item in items]

        for env in PhotoEnvironment.objects.all():
            wv_results = [1 - np.average(word2vec_model.wv.distances(env.name, detection.split(' '))) for
                          detection in
                          detections]
            kwargs = dict(
                photo=photo,
                environment=env
            )
            if len(weights):
                EnvironmentFuzzyMembership.objects.create(
                    value=np.average(
                        wv_results,
                        weights=weights
                    ),
                    **kwargs
                )
            else:
                EnvironmentFuzzyMembership.objects.create(
                    value=0.0,
                    **kwargs
                )
