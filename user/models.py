from django.db import models
from django.contrib.auth.models import User

class LoanModel(models.Model):
    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
    ]
    MARRIED_CHOICES = [
        ('yes', 'Yes'),
        ('no', 'No'),
    ]
    EDUCATION_CHOICES = [
        ('graduate', 'Graduate'),
        ('not_graduate', 'Not Graduate'),
    ]
    SELF_EMPLOYED = [
        ('yes', 'YES'),
        ('no', 'NO'),
    ]
    PROPERTY_AREA_CHOICES = [
        ('urban', 'Urban'),
        ('semiurban', 'Semiurban'),
        ('rural', 'Rural'),
    ]

    id = models.AutoField(primary_key=True)
    gender = models.CharField(max_length=6, choices=GENDER_CHOICES)
    married = models.CharField(max_length=3, choices=MARRIED_CHOICES)
    dependent = models.PositiveIntegerField()
    education = models.CharField(max_length=20, choices=EDUCATION_CHOICES)
    self_employed= models.CharField(max_length=20, choices=SELF_EMPLOYED)
    income = models.PositiveIntegerField()
    co_income = models.PositiveIntegerField()
    loan = models.PositiveIntegerField()
    loan_term = models.PositiveIntegerField()
    credit = models.PositiveIntegerField()
    property_area= models.CharField(max_length=20, choices=PROPERTY_AREA_CHOICES)
      # Make sure this field is defined correctly

    user_id = models.ForeignKey(User, on_delete=models.CASCADE)

    class Meta:
        db_table = 'loan'
















