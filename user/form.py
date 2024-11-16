from pyexpat import model
from django import forms
from user.models import LoanModel

class LoanForm(forms.ModelForm):
    class Meta:
        model=LoanModel
        fields ="__all__"