import os

from django.conf import settings
from django.db import models

# Create your models here.


class PhotoEnvironment(models.Model):
    name = models.CharField(max_length=40)
    display_name = models.CharField(max_length=40)
    active = models.BooleanField(default=True)

    def __str__(self):
        return str(self.display_name) + (" (inactive)" if not self.active else "")


class EnvironmentFuzzyMembership(models.Model):
    environment = models.ForeignKey('PhotoEnvironment', on_delete=models.CASCADE)
    photo = models.ForeignKey('Photo', on_delete=models.CASCADE)
    value = models.FloatField()


class Photo(models.Model):
    name = models.CharField(max_length=255, default='', unique=True)
    vector = models.ManyToManyField('PhotoEnvironment', through='EnvironmentFuzzyMembership')
    objects = models.ManyToManyField('ObjectCategory', through='ObjectIdentification')
    active = models.BooleanField(default=True)

    def __str__(self):
        return self.name

    def exists(self, check_input=False):
        if check_input:
            return os.path.exists(self.get_input_path())
        return os.path.exists(self.get_path())

    def get_input_path(self):
        return os.path.join(settings.PHOTOS_INPUT_PATH, self.name)

    def get_output_path(self):
        return os.path.join(settings.PHOTOS_OUTPUT_PATH, self.name)

    def get_path(self):
        return os.path.join(settings.IMAGE_PATH, self.name)

    def get_vector(self):
        return [f.value for f in EnvironmentFuzzyMembership.objects.filter(photo=self).order_by('environment_id') if f.environment.active]

    def get_dict_vector(self):
        return [(f.environment.display_name, f.value) for f in EnvironmentFuzzyMembership.objects.filter(photo=self).order_by('environment_id') if f.environment.active]

    def get_objects(self):
        return [(identification.category.name, identification.confidence) for identification in ObjectIdentification.objects.filter(photo=self)]

    get_dict_vector.__name__ = 'Vector'
    get_objects.__name__ = 'Objects'


class ObjectCategory(models.Model):
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name


class ObjectIdentification(models.Model):
    category = models.ForeignKey('ObjectCategory', on_delete=models.CASCADE)
    photo = models.ForeignKey('Photo', on_delete=models.CASCADE)
    confidence = models.FloatField()

    def __str__(self):
        return f"{self.photo} -> {self.category} ({'%.2f' % self.confidence})"

