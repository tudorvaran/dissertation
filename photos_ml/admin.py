from django.contrib import admin
from django.utils.safestring import mark_safe
from .models import Photo, PhotoEnvironment, EnvironmentFuzzyMembership, ObjectCategory, ObjectIdentification


# Register your models here.
admin.site.register(ObjectCategory)


@admin.register(ObjectIdentification)
class ObjectIdentificationAdmin(admin.ModelAdmin):
    list_display = ['image', 'category', 'confidence']

    def image(self, obj):
        return mark_safe(f"<img src=/photos/media/?id={obj.photo.id} width='50' height='50' />")

    image.allow_tags = True
    image.__name__ = 'Image'


@admin.register(PhotoEnvironment)
class PhotoEnvironmentAdmin(admin.ModelAdmin):
    list_display = ['display_name', 'name', 'active']


@admin.register(Photo)
class PhotoAdmin(admin.ModelAdmin):
    list_display = ['name', 'id', 'image', 'get_dict_vector', 'get_objects']
    readonly_fields = ['id', 'name', 'larger_image']

    def image(self, obj, w=50, h=50):
        return mark_safe(f"<img src=/photos/media/?id={obj.id} width='{w}' height='{h}' />")

    def larger_image(self, obj):
        return self.image(obj, 400, 400)

    image.allow_tags = True
    larger_image.allow_tags = True
    larger_image.__name__ = 'Image'
    image.__name__ = 'Image'


@admin.register(EnvironmentFuzzyMembership)
class EnvironmentFuzzyMembershipAdmin(admin.ModelAdmin):
    list_display = ['environment', 'image', 'value']

    def image(self, obj):
        return mark_safe(f"<img src=/photos/media/?id={obj.photo.id} width='50' height='50' />")

    image.allow_tags = True
    image.__name__ = 'Image'


