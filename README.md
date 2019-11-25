[![Build Status](https://travis-ci.org/unidesigner/circuitmap.svg?branch=master)](https://travis-ci.org/unidesigner/circuitmap)
[![Coverage Status](https://coveralls.io/repos/github/unidesigner/circuitmap/badge.svg?branch=master)](https://coveralls.io/github/unidesigner/circuitmap?branch=master)

# circuitmap

Circuit Map is a django application which acts as a drop-in
extension for [CATMAID](http://www.catmaid.org). It contains API
endpoints and static files.

## Quick start

1. Install circuitmap in whichever python environment is running
CATMAID with `pip install -e path/to/this/directory`

2. Run `python manage.py migrate` to create the circuitmap models.

3. Run `python manage.py collectstatic -l` to pick up
circuitmap's static files.
