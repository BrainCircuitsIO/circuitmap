# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from django.utils.encoding import python_2_unicode_compatible

class Synlinks(models.Model):

	pre_x = models.FloatField()
	pre_y = models.FloatField()
	pre_z = models.FloatField()
	post_x = models.FloatField()
	post_y = models.FloatField()
	post_z = models.FloatField()
	scores = models.FloatField()
	cleft_scores = models.IntegerField()
	dist = models.FloatField()
	segmentid_x = models.IntegerField(db_index=True)
	segmentid_y = models.IntegerField(db_index=True)
	offset = models.IntegerField(db_index=True)
	prob_min = models.IntegerField()
	prob_max = models.IntegerField()
	prob_sum = models.IntegerField()
	prob_mean = models.IntegerField()
	prob_count = models.IntegerField()
	cleft_id = models.BigIntegerField(default=0)
	clust_con_offset = models.IntegerField()
