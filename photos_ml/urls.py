from django.urls import path

from photos_ml import views

urlpatterns = [
    path('media/', views.get_image, name='get-image'),
    path('query/', views.query_view, name='query'),
    path('', views.homepage, name='home')
]
