from six.moves import input

from django.core.management.base import BaseCommand
from django.apps import apps
from django.db import connection

from psycopg2.extensions import AsIs


class Command(BaseCommand):
    help = 'Drops all database tables backing the circuitmap models.'

    def add_arguments(self, parser):
        parser.add_argument('-y', action='store_true', dest='yes', default=False)

    def handle(self, *args, **options):
        selection = 'y' if options['yes'] else 'not an option'
        while selection.lower() not in ['y', 'n', '']:
            selection = input('This will drop all circuitmap-related tables. Are you sure ([y]/n)? ')

        if selection == 'n':
            self.stdout.write(self.style.FAILURE('Aborting'))
            return

        cursor = connection.cursor()

        for ss_model in apps.get_app_config('circuitmap').get_models():
            table = ss_model._meta.db_table
            self.stdout.write(
                '{}: Dropping {}...'.format(ss_model.__name__, table)
            )

            cursor.execute('DROP TABLE IF EXISTS %s CASCADE;', (AsIs(table),))

        self.stdout.write(self.style.SUCCESS(
            'Successfully dropped circuitmap tables. '
            '`pip uninstall circuitmap` and remove from your INSTALLED_APPS to finish uninstall.'
        ))
