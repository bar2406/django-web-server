# -*- coding: utf-8 -*-
# Generated by Django 1.10.2 on 2016-10-22 17:02
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('MNISTDist', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='minibatch',
            name='isTrain',
        ),
    ]
