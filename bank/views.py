from django.shortcuts import render
from .forms import LoanEligibilityForm  
import joblib
import numpy as np
import pandas as pd

def loan_eligibility(request):
    prediction = None  # Default prediction as None
    form = LoanEligibilityForm(request.POST or None)
    
    if request.method == 'POST':
        if form.is_valid():
            encoders = joblib.load("C:\\Users\\Public.NAWARAJ\\Desktop\\Code\\Django\\LoanPrediction\\App\\encoders.pkl")
            model = joblib.load("C:\\Users\\Public.NAWARAJ\\Desktop\\Code\\Django\\LoanPrediction\\App\\naive_bayes_model.pkl")  
            #accuracy = 0.7886178861788617 
            
           # Encode categorical data from the form using the saved encoders
           
            try:
                data = {

                    'Gender': encoders['Gender'].transform([form.cleaned_data['gender']])[0],
                    'Married': encoders['Married'].transform([form.cleaned_data['married']])[0],
                    'Dependents': encoders['Dependents'].transform([form.cleaned_data['dependents']])[0],
                    'Education': encoders['Education'].transform([form.cleaned_data['education']])[0],
                    'Self_Employed': encoders['Self_Employed'].transform([form.cleaned_data['self_employed']])[0],
                    'ApplicantIncome': float(form.cleaned_data['applicant_income']),
                    'CoapplicantIncome': float(form.cleaned_data['coapplicant_income']),
                    'LoanAmount': float(form.cleaned_data['loan_amount']),
                    'Loan_Amount_Term': float(form.cleaned_data['loan_amount_term']),
                    'Credit_History': float(form.cleaned_data['credit_history']),
                    'Property_Area': encoders['Property_Area'].transform([form.cleaned_data['property_area']])[0],
                }

            except ValueError as e:
                print("Error in data conversion:", e)
                return render(request, 'form.html', {'form': form, 'error': "Invalid input data format."})

            # Convert the data to a list and then to numpy array for prediction
            data_array = np.array(list(data.values())).reshape(1, -1)
            
    
            
            feature_names = [
               'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
               'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
               'Credit_History', 'Property_Area'
]

            # Wrap `data_array` as a DataFrame with the specified feature names
            data_df = pd.DataFrame(data_array, columns=feature_names)

            # Predict using the model with named features
            prediction = model.predict(data_df)[0]
            prediction_result = 'Eligible' if prediction == 1 else 'Not Eligible'
            
            # Predict using the loaded model
            # prediction = model.predict(data_array)[0]
            # prediction_result = 'Eligible' if prediction == 1 else 'Not Eligible'
            print("Prediction result:", prediction_result)  # Debug line
            
        else:
            print("Form is not valid:", form.errors)
    else:
        form = LoanEligibilityForm()    

    return render(request, 'form.html', {'form': form})

