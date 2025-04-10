# Generated by Django 5.1.5 on 2025-02-18 18:20

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='modelpredict',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.TextField(default='', max_length=100000000000000000)),
                ('predict_class', models.TextField(default='', max_length=100000000000000000)),
                ('predict_accuracy', models.IntegerField(default=0, max_length=10000000)),
                ('predicted', models.BooleanField(default=False)),
            ],
        ),
    ]
