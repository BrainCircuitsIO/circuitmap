# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.conf.urls import url

import circuitmap.control

app_name = 'circuitmap'

urlpatterns = [
    url(r'^is-installed$', circuitmap.control.is_installed),
    url(r'^index$', circuitmap.control.index),
    url(r'^(?P<project_id>\d+)/synapses/fetch$', circuitmap.control.fetch_synapses),
]
