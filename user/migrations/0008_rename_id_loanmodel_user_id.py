# Generated by Django 4.2.11 on 2024-12-12 08:48

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0007_remove_loanmodel_user_id_loanmodel_algorithm_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='loanmodel',
            old_name='id',
            new_name='user_id',
        ),
    ]
