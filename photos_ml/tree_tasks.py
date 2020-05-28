import json
import pickle

import numpy as np
import structlog
from django.conf import settings
from django.core.cache import cache
from sklearn.neighbors import BallTree

from photos_ml.models import Photo
from photos_ml.precision import precision_at_k, mean_average_precision
from photos_ml.recommender import Recommender

logger = structlog.get_logger()


def build_tree():
    logger.info("Building tree...")
    limit = settings.RECOMMENDER_ITEMS_LIMIT

    photo_list = [photo for photo in Photo.objects.filter(active=True).order_by('pk') if photo.exists() and photo.get_vector()]
    if limit:
        logger.info("Not selecting photo", id=photo_list[limit].pk)
        photo_list = photo_list[:limit]

    id_mapping = [photo.id for photo in photo_list]
    vectors = np.array([
        np.array(photo.get_vector()) for photo in photo_list
    ])
    tree = BallTree(vectors)

    cache.set("tree", pickle.dumps(tree))
    cache.set("index", pickle.dumps(id_mapping))
    logger.info("Done!")


def evaluate_recommender(recommender):
    with open(settings.VALIDATION_FILE, 'r') as f:
        validation_data = json.load(f)

    validation_images = validation_data["images"]

    img_mapping = dict()
    reverse_img_mapping = dict()

    for img in validation_images:
        photo = Photo.objects.filter(name=img['file_name']).first()

        if not photo:
            continue

        img_mapping[photo.id] = img["id"]
        reverse_img_mapping[img["id"]] = photo.id

    dimension_labels = {category["id"]: category["name"] for category in validation_data["categories"]}

    max_category_id = max(dimension_labels.keys())

    vectors = []
    ids = []
    id_to_vector_mapping = dict()

    for annotation in validation_data['annotations']:
        img_id = annotation['image_id']
        if img_id not in reverse_img_mapping:
            continue

        ids.append(img_id)

        v = [0 for _ in range(max_category_id)]

        for segment_info in annotation['segments_info']:
            v[segment_info["category_id"] - 1] = 1

        vectors.append(np.array(v))
        id_to_vector_mapping[img_id] = np.array(v)

    vectors = np.array(vectors)

    logger.info("Building valid recommendations tree...")
    validation_tree = BallTree(vectors)

    validation_recommender = Recommender(validation_tree, ids)

    K = 15
    p_at_k_values = [1, 3, 5, 10]

    prec_values = {
        'map': [],
        **{f'P@{k}': [] for k in p_at_k_values}
    }

    logger.info("Starting to test recommendations...")
    for photo_id in img_mapping:
        photo = Photo.objects.filter(pk=photo_id).first()
        if not photo.get_vector():
            logger.info("Could not verify photo!", id=photo_id)
            continue

        recommendations = recommender.recommend([Photo.objects.filter(pk=photo_id).first().get_vector()], k=K, exclude_index=photo_id)

        img_id = img_mapping[photo_id]
        validated_recommendations = validation_recommender.recommend([id_to_vector_mapping[img_id]], k=K, exclude_index=img_id)
        mapped_recommendations = [item["i"] for item in recommendations]

        valid_mapped_recommendations = [reverse_img_mapping[imgs_id["i"]] for imgs_id in validated_recommendations]

        for k in p_at_k_values:
            prec_values[f'P@{k}'].append(precision_at_k(mapped_recommendations, valid_mapped_recommendations, k))
        prec_values['map'].append(mean_average_precision(mapped_recommendations, valid_mapped_recommendations))

    prec_values['mean'] = {
        'map': np.average(np.array(prec_values['map'])),
        **{f'P@{k}': np.average(np.array(prec_values[f'P@{k}'])) for k in p_at_k_values}
    }
    logger.info("Saving precisions...")

    with open(settings.EVALUATION_RESULTS_FILE, 'w') as f:
        json.dump(prec_values, f)

    logger.info("Done!")