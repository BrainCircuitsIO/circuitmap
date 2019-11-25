# -*- coding: utf-8 -*-
from catmaid.tests.apis.common import CatmaidApiTestCase


class CircuitmapTestCase(CatmaidApiTestCase):
    fixtures = CatmaidApiTestCase.fixtures + ['circuitmap_testdata.json']

    @classmethod
    def setUpTestData(cls):
        super(CircuitmapTestCase, cls).setUpTestData()
