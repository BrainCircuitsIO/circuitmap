# -*- coding: utf-8 -*-
import json
from circuitmap.tests.common import CircuitmapTestCase


URL_PREFIX = '/ext/circuitmap'


class InstallationTest(CircuitmapTestCase):
    def test_is_installed(self):
        response = self.client.get(URL_PREFIX + '/is-installed')
        self.assertEqual(response.status_code, 200)
        parsed_response = json.loads(response.content.decode('utf-8'))
        assert parsed_response['is_installed']
