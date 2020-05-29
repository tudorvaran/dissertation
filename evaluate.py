import init  # NOQA

import json

from django.conf import settings

from photos_ml.models import Photo
from photos_ml.tree_tasks import evaluate_recommender
from photos_ml.utils import get_recommender

prec_values = evaluate_recommender(get_recommender())

with open(f"results_{Photo.objects.count() if not settings.IMAGE_LIMIT else settings.IMAGE_LIMIT}_{settings.RECOMMENDER_SCALER}.json", 'w') as f:
    json.dump(prec_values, f)
