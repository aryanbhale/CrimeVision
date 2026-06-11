from django.core.management.base import BaseCommand
from police.models import CrimeRecords
from police.face_encoding import get_key_points
import base64
import io
from PIL import Image


class Command(BaseCommand):
    help = 'Add a criminal record directly from command line. Generates key_points automatically.'

    def add_arguments(self, parser):
        parser.add_argument('--key',     required=True,  help='Unique criminal ID e.g. CR_001')
        parser.add_argument('--name',    required=True,  help='Full name e.g. "John Doe"')
        parser.add_argument('--image',   required=True,  help='Full path to the image file')
        parser.add_argument('--against', default='0',    help='Number of cases (default: 0)')
        parser.add_argument('--gender',  default='Male', help='Male or Female (default: Male)')

    def handle(self, *args, **options):
        key     = options['key']
        name    = options['name']
        image_path = options['image']
        against = options['against']
        gender  = options['gender']

        # Check if key already exists
        if CrimeRecords.objects.filter(key=key).exists():
            self.stdout.write(self.style.ERROR(f"Key '{key}' already exists. Use a different key."))
            return

        # Open and convert image to base64
        self.stdout.write(f"Opening image: {image_path}")
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Could not open image: {e}"))
            return

        buff = io.BytesIO()
        img.save(buff, format='JPEG')
        img_bytes = buff.getvalue()
        img_b64 = base64.b64encode(img_bytes)

        # Extract face encoding
        self.stdout.write("Extracting face encoding (this may take a few seconds)...")
        try:
            key_points = get_key_points(img_b64)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Face encoding failed: {e}"))
            return

        if not key_points:
            self.stdout.write(self.style.ERROR(
                "No face detected in the image. Use a clear, front-facing photo."
            ))
            return

        # Save to DB
        record = CrimeRecords(
            key=key,
            name=name,
            against=against,
            gender=gender,
            img=str(img_b64),
            key_points=key_points,
        )
        record.save()

        self.stdout.write(self.style.SUCCESS(
            f"\n✓ Record added successfully!"
            f"\n  Key     : {key}"
            f"\n  Name    : {name}"
            f"\n  Against : {against} cases"
            f"\n  Gender  : {gender}"
            f"\n\nNext step: run  .\\env\\Scripts\\python.exe manage.py retrain"
        ))
