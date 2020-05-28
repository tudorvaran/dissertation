import pickle

from django.core.cache import cache

from photos_ml.recommender import Recommender
from photos_ml.tree_tasks import build_tree


def get_recommender():
    tree_data = cache.get('tree')
    index_data = cache.get('index')
    if not tree_data or not index_data:
        build_tree()
        index_data = cache.get('index')
        tree_data = cache.get('tree')

    tree = pickle.loads(tree_data)
    index_list = pickle.loads(index_data)

    return Recommender(tree, index_list)

