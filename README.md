# Circuit Map

Circuit Map is a Django application which acts as a drop-in
extension for [CATMAID](https://catmaid.readthedocs.io/en/latest/extensions.html)
or can be run as an independent Django app.

Circuit Map exposes a widget in CATMAID with associated endpoints and data models
for integrating automatically generated synaptic link data and
segmentations into CATMAID circuit mapping workflows.

The synaptic link data for the [FAFB dataset](http://www.temca2data.org/)
is based on the publication by [Buhmann et al. 2019](ADDREF). For interactive
use of the data in a Jupyter notebook, please refer to 
[this example in fafbseg-py](https://github.com/flyconnectome/fafbseg-py/examples).

## Quick start for integration into CATMAID

1. Install the dependencies with `pip install -r requirements.txt`.

2. Install circuitmap in whichever python environment is running
CATMAID with `pip install -e .`

3. Run `python manage.py migrate` to create the circuitmap models.

4. Run `python manage.py collectstatic -l` to pick up
circuitmap's static files.

5. The extension needs to be enabled in the main CATMAID instance by
updating the `KNOWN_EXTENSIONS` variable in `CATMAID/django/projects/pipelinefiles.py`.

6. The available synaptic link data needs to be ingested into the generated
Postgres table `circuitmap_synlinks`.

7. A few additional settings have to be configured to interoperate with segmentation
data and CATMAID import procedure (see `circuitmap/control/settings.py.example`).
