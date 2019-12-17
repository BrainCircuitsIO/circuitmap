# Circuit Map

Circuit Map is a Django application which can act as a drop-in
extension for [CATMAID](https://catmaid.readthedocs.io/en/latest/extensions.html).

Circuit Map provides tooling to integrate remote skeletons derived from automated segmentations
and automatically generated synaptic link data into CATMAID circuit mapping workflows.

An example use of the tool is based on automatically generated synaptic link data by [Buhmann et al. 2019](https://www.biorxiv.org/content/10.1101/2019.12.12.874172v1) for the [FAFB dataset](http://www.temca2data.org/). Synaptic link data can be added to manually reconstructed skeletons or automatically generated skeletons derived from a [whole brain segmentation](https://fafb-ffn1.storage.googleapis.com/landing.html). Up- and downstream skeleton partners derived from this synaptic link data can be imported automatically into the current CATMAID project.


## Integration into CATMAID

1. Install the dependencies with `pip install -r requirements.txt`.

2. Install circuitmap in whichever python environment is running
CATMAID with `pip install -e .`

3. Run `python manage.py migrate` (in the CATMAID folder) to create the circuitmap models.

4. Run `python manage.py collectstatic -l`  (in the CATMAID folder) to pick up
circuitmap's static files.

5. The synaptic link data needs to be ingested into the generated
Postgres table `circuitmap_synlinks` from the [SQLite database dump](https://github.com/funkelab/synful_fafb).

6. A few additional settings have to be configured to interoperate with segmentation
data and CATMAID import procedure (see `circuitmap/control/settings.py.example`).
