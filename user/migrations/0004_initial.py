# Generated by Django 4.2.8 on 2024-09-24 14:43

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('user', '0003_delete_loanmodel'),
    ]

    operations = [
        migrations.CreateModel(
            name='LoanModel',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('gender', models.CharField(choices=[('male', 'Male'), ('female', 'Female')], max_length=6)),
                ('married', models.CharField(choices=[('yes', 'Yes'), ('no', 'No')], max_length=3)),
                ('dependent', models.PositiveIntegerField()),
                ('education', models.CharField(choices=[('graduate', 'Graduate'), ('not_graduate', 'Not Graduate')], max_length=20)),
                ('income', models.PositiveIntegerField()),
                ('co_income', models.PositiveIntegerField()),
                ('loan', models.PositiveIntegerField()),
                ('loan_term', models.PositiveIntegerField()),
                ('credit', models.PositiveIntegerField()),
                ('property', models.CharField(choices=[('urban', 'Urban'), ('semiurban', 'Semiurban'), ('rural', 'Rural')], max_length=20)),
                ('user_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'loan',
            },
        ),
    ]
