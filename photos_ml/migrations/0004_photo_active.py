# Generated by Django 3.0.5 on 2020-05-28 17:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('photos_ml', '0003_auto_20200426_1812'),
    ]

    operations = [
        migrations.AddField(
            model_name='photo',
            name='active',
            field=models.BooleanField(default=True),
        ),
    ]