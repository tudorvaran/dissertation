import os
import random

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404

from photos_ml.models import Photo
from photos_ml.utils import get_recommender


def homepage(request):
    photo_ids = []
    min_pk = Photo.objects.all().order_by("pk").first().pk
    max_pk = min_pk + settings.RECOMMENDER_ITEMS_LIMIT

    while len(photo_ids) < settings.HOMEPAGE_PICTURES:
        random_id = random.randint(min_pk, max_pk)

        photo = Photo.objects.filter(pk=random_id).first()

        if photo:
            photo_ids.append(random_id)

    context = dict(
        ids=photo_ids
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

    photo = get_object_or_404(Photo, pk=photo_id)
    if not photo.get_vector():
        return HttpResponse(status=204)

    recommender = get_recommender()

    items = recommender.recommend([photo.get_vector()], k=k+1, exclude_index=photo_id)

    context = dict(
        main_photo_id=photo_id,
        photos=items,
        k=k
    )

    return render(request, 'knn_view.html', context)
