# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from django.utils.encoding import python_2_unicode_compatible

class Synlinks(models.Model):
	ids = models.IntegerField()
	pre_x = models.IntegerField()
	pre_y = models.IntegerField()
	pre_z = models.IntegerField()
	post_x = models.IntegerField()
	post_y = models.IntegerField()
	post_z = models.IntegerField()
	scores = models.FloatField()
	segmentid_x = models.IntegerField(db_index=True)
	segmentid_y = models.IntegerField(db_index=True)
	max = models.IntegerField()
	index = models.IntegerField()
