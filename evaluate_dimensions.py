import json
import random

import numpy as np
import structlog
from django.conf import settings

import init # NOQA
from photos_ml.models import PhotoEnvironment, Photo
from photos_ml.recommender import Recommender
from photos_ml.tree_tasks import build_tree, evaluate_recommender

logger = structlog.get_logger()

environments = list(PhotoEnvironment.objects.all().order_by('pk'))
initial_state = [environment.active for environment in environments]

PhotoEnvironment.objects.all().update(active=True)

env_no = len(environments)
active_env_no_of_checks = [0, 0, 8, 8, 8, 8, 8, 8, 1]
#active_env_no_of_checks = [0, 0, 1, 1, 0, 0, 0, 0, 0]
total_checks = sum(active_env_no_of_checks)
combinations_checked = 0

bit_masks = [x for x in range(2 ** env_no)]
random.shuffle(bit_masks)

precision_values = dict()

logger.info("Pre-fetching database items")
photo_items = [(photo, photo.get_vector(), photo.exists()) for photo in Photo.objects.filter(active=True).order_by('pk')]
photo_list = [item[0] for item in photo_items if item[1] and item[2]]
vectors = np.array([np.array(item[1]) for item in photo_items if item[1] and item[2]])
logger.info("Pre-fetching done!")

pos = 0
while combinations_checked < total_checks and pos < len(bit_masks):
    crt_combination = bit_masks[pos]
    pos += 1

    active_positions = []
    for i in range(env_no):
        p = 2 ** i
        if p & crt_combination:
            active_positions.append(i)
    active_bits = len(active_positions)

    if active_env_no_of_checks[active_bits] == 0:
        continue

    active_env_no_of_checks[active_bits] -= 1
    dimensions_labels = []

    for i in range(env_no):
        p = 2 ** i
        environments[i].active = ((p & crt_combination) != 0)
        environments[i].save(update_fields=['active'])

        if p & crt_combination:
            dimensions_labels.append(environments[i].name)

    logger.info(
        "Running evaluation...",
        set=dimensions_labels,
        active=active_bits,
        checked=combinations_checked,
        total=total_checks
    )
    vectors_subset = vectors[:, active_positions]
    tree, scaler, id_mapping = build_tree(photo_list=photo_list, vectors=vectors_subset)
    recommender = Recommender(tree, scaler, id_mapping)

    precision = evaluate_recommender(recommender)

    if active_bits not in precision_values:
        precision_values[active_bits] = []

    precision_values[active_bits].append(dict(
        set=dimensions_labels,
        mean=precision['mean']
    ))
    combinations_checked += 1

for i, environment in enumerate(environments):
    environment.active = initial_state[i]
    environment.save(update_fields=["active"])

precision_values["means"] = dict()
for bits_no in precision_values:
    if bits_no == "means":
        continue
    if len(precision_values[bits_no]):
        precision_values["means"][bits_no] = {
            metric: np.average(
                [precision_values[bits_no][i]["mean"][metric] for i in range(len(precision_values[bits_no]))]
            ) for metric in precision_values[bits_no][0]["mean"]
        }

with open(f"results_dimensions_{settings.RECOMMENDER_SCALER}.json", "w") as f:
    json.dump(precision_values, f)
