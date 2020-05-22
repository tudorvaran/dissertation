import os
import pickle
import random

from django.conf import settings
from django.core.cache import cache
from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404

from photos_ml.models import Photo
from photos_ml.tasks import build_tree


def homepage(request):
    photo_ids = [photo.pk for photo in Photo.objects.all() if photo.get_vector()]

    random.shuffle(photo_ids)
    context = dict(
        ids=photo_ids[:30]
    )
    return render(request, 'homepage.html', context)


def get_image(request):
    name = request.GET.get('name')
    photo_id = request.GET.get('id')
    photo = None

    if photo_id:
        photo = Photo.objects.filter(id=photo_id).first()
    if not photo and name:
        photo = Photo.objects.filter(name=name).first()

    if not photo:
        return HttpResponse(status=404)

    if not photo.exists():
        return HttpResponse(
            content="Photo was not found",
            status=404
        )

    with open(os.path.join(settings.IMAGE_PATH, photo.name), 'rb') as f:
        return HttpResponse(content=f.read(), content_type=f'image/{photo.name.split(".")[1]}')


def query_view(request):
    photo_id = request.GET.get('id')
    k = request.GET.get('k')

    try:
        k = int(k)
        photo_id = int(photo_id)
    except ValueError:
        return HttpResponse(content="Not an integer", status=500)

    tree_data = cache.get('tree')
    index_data = cache.get('index')
    if not tree_data or not index_data:
        build_tree()
        index_data = cache.get('index')
        tree_data = cache.get('tree')

    photo = get_object_or_404(Photo, pk=photo_id)
    if not photo.get_vector():
        return HttpResponse(status=204)

    tree = pickle.loads(tree_data)
    index_list = pickle.loads(index_data)

    distances, leafs = tree.query([photo.get_vector()], k=k+1, return_distance=True)

    photo_ids = [index_list[leaf] for leaf in leafs[0] if index_list[leaf] != photo_id]

    items = [dict(d=distances[0][i], i=photo_ids[i]) for i in range(k)]

    context = dict(
        main_photo_id=photo_id,
        photos=items,
        k=k
    )

    return render(request, 'knn_view.html', context)
