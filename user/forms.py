# loan_predictor/forms.py
from django import forms

class LoanPredictionForm(forms.Form):
    Gender = forms.ChoiceField(choices=[('Male', 'Male'), ('Female', 'Female')])
    Married = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    Dependents = forms.IntegerField(min_value=0, max_value=3)
    Education = forms.ChoiceField(choices=[('Graduate', 'Graduate'), ('Not Graduate', 'Not Graduate')])
    Self_Employed = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    ApplicantIncome = forms.FloatField()
    CoapplicantIncome = forms.FloatField()
    LoanAmount = forms.FloatField()
    Loan_Amount_Term = forms.FloatField()
    Credit_History = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    Property_Area = forms.ChoiceField(choices=[('Urban', 'Urban'), ('Rural', 'Rural'), ('Semiurban', 'Semiurban')])

    def clean(self):
        cleaned_data = super().clean()

        # Encode categorical features (mapping logic)
        gender_mapping = {'Male': 1, 'Female': 0}
        married_mapping = {'No': 0, 'Yes': 1}
        dependents_mapping = {0: 0, 1: 1, 2: 2, '3+': 3}
        education_mapping = {'Graduate': 0, 'Not Graduate': 1}
        self_employed_mapping = {'No': 0, 'Yes': 1}
        credit_history_mapping = {'No': 0, 'Yes': 1}
        property_area_mapping = {'Urban': 2, 'Rural': 0, 'Semiurban': 1}

        # Apply mappings
        cleaned_data['Gender'] = gender_mapping.get(cleaned_data['Gender'])
        cleaned_data['Married'] = married_mapping.get(cleaned_data['Married'])
        cleaned_data['Dependents'] = dependents_mapping.get(cleaned_data['Dependents'])
        cleaned_data['Education'] = education_mapping.get(cleaned_data['Education'])
        cleaned_data['Self_Employed'] = self_employed_mapping.get(cleaned_data['Self_Employed'])
        cleaned_data['Credit_History'] = credit_history_mapping.get(cleaned_data['Credit_History'])
        cleaned_data['Property_Area'] = property_area_mapping.get(cleaned_data['Property_Area'])
        
        return cleaned_data

    
#Loan_ID,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,Loan_Status
