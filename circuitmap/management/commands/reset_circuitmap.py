from six.moves import input

from django.core.management.base import BaseCommand
from django.apps import apps


class Command(BaseCommand):
    help = 'Deletes all rows from database tables backing the circuitmap models.'

    def add_arguments(self, parser):
        parser.add_argument('-y', action='store_true', dest='yes', default=False)

    def handle(self, *args, **options):
        selection = 'y' if options['yes'] else 'not an option'
        while selection.lower() not in ['y', 'n', '']:
            selection = input('This will empty all circuitmap-related database tables. Are you sure ([y]/n)? ')

        if selection == 'n':
            self.stdout.write(self.style.FAILURE('Aborting'))
            return

        self.stdout.write('Note: row counts may change due to foreign key deletion cascading')
        for model in apps.get_app_config('circuitmap').get_models():
            all_rows = model.objects.all()
            self.stdout.write(
                '{}: Deleting {} rows from {}...'.format(model.__name__, all_rows.count(), model._meta.db_table)
            )
            all_rows.delete()

        self.stdout.write(self.style.SUCCESS('Successfully cleared circuitmap tables'))
