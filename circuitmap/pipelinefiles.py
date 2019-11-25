# -*- coding: utf-8 -*-
from __future__ import unicode_literals

"""Specifies static assets (CSS, JS) required by the CATMAID front-end.

This module specifies all the static files that are required by the
extension's front-end.

Static files specified here will be added to CATMAID's index page after
all of CATMAID's own files and dependencies, but before standalone static
files. CSS will be compressed; Javascript will just be concatenated in the
order that the source files are specified.

Globs can be used to pick up multiple source files.
"""

from collections import OrderedDict

STYLESHEETS = OrderedDict()

STYLESHEETS['circuitmap'] = {
    'source_filenames': (
        'circuitmap/css/*.css',
    ),
    'output_filename': 'css/circuitmap.css'
}

JAVASCRIPT = OrderedDict()

JAVASCRIPT['circuitmap'] = {
    'source_filenames': (
        'circuitmap/js/widgets/*.js',
    ),
    'output_filename': 'js/circuitmap.js'
}
