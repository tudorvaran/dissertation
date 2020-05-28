import os

import gensim
import numpy as np
import structlog
from django.conf import settings
from imageai.Detection import ObjectDetection

from photos_ml.models import Photo, EnvironmentFuzzyMembership, PhotoEnvironment

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


def build_vectors(ignore_existing=False):
    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(settings.OBJECT_RECOGNITION_MODEL)
    detector.loadModel()

    all_detections = dict()

    for photo in Photo.objects.all():
        if not photo.exists(check_input=True):
            logger.warning("Path does not exist!", id=photo.id, name=photo.name)
            continue

        if ignore_existing and photo.get_vector():
            continue

        ext = photo.name.split('.')[1]
        tmp_file = os.path.join(execution_path, f"photo.{ext}")
        new_tmp_file = os.path.join(execution_path, f"photo_out.{ext}")
        if os.path.islink(tmp_file):
            os.unlink(tmp_file)
        os.symlink(photo.get_input_path(), tmp_file,)

        logger.info("Detecting", name=photo.name, id=photo.pk)
        detections = detector.detectObjectsFromImage(
            input_image=tmp_file,
            output_image_path=new_tmp_file
        )

        os.unlink(tmp_file)
        if os.path.isfile(photo.get_output_path()):
            os.remove(photo.get_output_path())
        os.rename(new_tmp_file, photo.get_output_path())

        all_detections[photo.id] = detections
        if not len(detections):
            logger.warning("No objects found!", photo_id=photo.id)

    logger.info("Finished image recognition. Loading word2vec model")
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(fname=settings.WORD2VEC_MODEL, binary=True)
    logger.info("Loaded word2vec model")

    if not ignore_existing:
        EnvironmentFuzzyMembership.objects.all().delete()

    for photo in Photo.objects.all():
        if not photo.exists(check_input=True):
            continue

        if ignore_existing and photo.get_vector():
            continue

        detections = all_detections.get(photo.id, [])
        weights = [detection["percentage_probability"] for detection in detections]
        if not len(weights):
            logger.warning("No objects detected in photo", id=photo.id, name=photo.name)
            continue

        for env in PhotoEnvironment.objects.all():
            wv_results = [1 - np.average(word2vec_model.wv.distances(env.name, detection["name"].split(' '))) for
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
