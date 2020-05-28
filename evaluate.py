import init  # NOQA
from photos_ml.tree_tasks import evaluate_recommender
from photos_ml.utils import get_recommender

evaluate_recommender(get_recommender())
