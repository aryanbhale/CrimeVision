from django.core.management.base import BaseCommand
from police.models import CrimeRecords
from police.train import start_train


class Command(BaseCommand):
    help = 'Audits DB records and retrains the classifier from valid ones.'

    def handle(self, *args, **kwargs):
        total = CrimeRecords.objects.count()
        valid = CrimeRecords.objects.exclude(key_points__isnull=True).exclude(key_points='').filter(key_points__contains='@').count()
        invalid = total - valid

        self.stdout.write(f"Total records : {total}")
        self.stdout.write(f"Valid (has key_points): {valid}")
        self.stdout.write(f"Invalid (missing key_points): {invalid}")

        if invalid:
            self.stdout.write(self.style.WARNING(
                f"\n{invalid} record(s) have no key_points and will be SKIPPED during training."
            ))
            for r in CrimeRecords.objects.all():
                if not r.key_points or '@' not in r.key_points:
                    self.stdout.write(self.style.ERROR(f"  BAD: key={r.key}  name={r.name}"))

        if valid == 0:
            self.stdout.write(self.style.ERROR("No valid records to train on. Add records via /police/match/ first."))
            return

        self.stdout.write("\nRetraining classifier...")
        start_train()
        self.stdout.write(self.style.SUCCESS("classifier.pkl saved. Match should now work."))
