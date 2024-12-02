from urllib import request
from django.shortcuts import redirect, render
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login as log
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from user.form import LoanForm

#Imports for Loan prediction
from .forms import LoanPredictionForm  
import joblib
import numpy as np
import pandas as pd
from .forms import LoanPredictionForm
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def index(request):
    return render(request,'home.html')
#test function:
def test(request):
    return render(request,'contact.html')

def home(request):
    return render(request,'home.html')


def login(request):
    return render(request,'user/login.html')

def login_user(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']    
        
        user = authenticate(request,username=username, password=password)

        if user is not None:
            if user.is_superuser == 0:

                log(request,user)
                return redirect('/home')

            else:
                messages.error(request,"Username and Password Don't Match, Please Try Again !")

                return redirect("/user/login")


        else:
            messages.error(request,"Username and Password Don't Match, Please Try Again !")

            return redirect("/user/login")

    else:
            messages.error(request,"Something is worng with your form validation, Please Try Again !")

            return redirect("/user/login")


def register(request):
    return render(request,'user/register.html')

# To register User

def register_user(request):
    if request.method == "POST":
        User.objects.create_user(
            first_name = request.POST['fullname'],
            username = request.POST['username'],
            password = request.POST['password'],
            email = request.POST['phonenumber'],

        )
        return redirect('/user/login')
    
    else:
        return render(request, '404.html', status=404)

def log_out(request):
    logout(request)
    return redirect('/home')

def contact(request):

    return render(request,'contact.html')


@login_required(login_url='/user/login')

#Loan prediction
def loan_prediction(request):
    form = LoanPredictionForm(request.POST or None)

    if request.method == 'POST' and form.is_valid():
        algorithm = request.POST.get('algorithm')
        try:
            cleaned_data = form.cleaned_data

            # Converting the form data into a DataFrame (or array)
            data_array = np.array(list(cleaned_data.values())).reshape(1, -1)
            data_df = pd.DataFrame(data_array, columns=cleaned_data.keys())

            if algorithm == 'naive_bayes':
                model = joblib.load('C:/Users/Public.NAWARAJ/Desktop/Code/Django/Loan-Prediction-System-main/user/loan_model.pkl')
                scaler = joblib.load('C:/Users/Public.NAWARAJ/Desktop/Code/Django/Loan-Prediction-System-main/user/scaler.pkl')
                metrics = joblib.load('C:/Users/Public.NAWARAJ/Desktop/Code/Django/Loan-Prediction-System-main/user/metrics.pkl')
               
                # Scaling the features
                data_scaled = scaler.transform(data_df)
                # Prediction
                prediction = model.predict(data_scaled)
                prediction_result = 'Eligible' if prediction == 1 else 'Not Eligible'

            elif algorithm == 'random_forest':
                model = joblib.load('C:/Users/Public.NAWARAJ/Desktop/Code/Django/Loan-Prediction-System-main/user/loan_model2.pkl')
                #scaler = joblib.load('C:/Users/Public.NAWARAJ/Desktop/Code/Django/Loan-Prediction-System-main/user/scaler.pkl')
                metrics = joblib.load('C:/Users/Public.NAWARAJ/Desktop/Code/Django/Loan-Prediction-System-main/user/metrics2.pkl')
                
                prediction = model.predict(data_df)
                prediction_result = 'Eligible' if prediction == 1 else 'Not Eligible'
            
            elif algorithm == 'XG_Boost':
                model = joblib.load('C:/Users/Public.NAWARAJ/Desktop/Code/Django/Loan-Prediction-System-main/user/loan_xgb_model.pkl')
                #encoder = joblib.load('C:/Users/Public.NAWARAJ/Desktop/Code/Django/Loan-Prediction-System-main/user/label_encoders_xgb.pkl')
                metrics = joblib.load('C:/Users/Public.NAWARAJ/Desktop/Code/Django/Loan-Prediction-System-main/user/metrics_xgb.pkl')
                
                prediction = model.predict(data_df)
                prediction_result = 'Eligible' if prediction == 1 else 'Not Eligible'

            else:
                raise ValueError("Invalid algorithm selected")

            # Returning values
            accuracy = metrics["accuracy"]
            classification = metrics["classification_report"]

            return render(request, "alert.html", {
                "result": prediction_result,
                "accuracy": f"{accuracy:.2f}",
                "classification": classification
            })

        except Exception as e:
            error_message = f"An error occurred during prediction: {str(e)}"
            return render(request, 'loan.html', {'form': form, 'error': error_message})

    # Render the form if it's a GET request or if the form is invalid
    return render(request, 'loan.html', {'form': form})



@login_required(login_url='/user/login')


def addloan(request,uid):


    if request.method == 'POST':
        # Create a new LoanModel instance with the submitted data
        loan_instance = LoanForm(
            request.POST
        )

        if loan_instance.is_valid():
            loan_instance.save()
        # # else:
        # #     print(loan_instance.errors)


    return redirect('/index')





