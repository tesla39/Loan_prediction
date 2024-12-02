# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load the dataset
file_path = "C:/Users/Public.NAWARAJ/Desktop/Code/Django/Loan-Prediction-System-main/user/loan.csv"
data = pd.read_csv(file_path)

# Drop irrelevant columns
data = data.drop(columns=["Loan_ID"])

# Handle missing values
categorical_columns = ["Gender", "Married", "Self_Employed", "Dependents", "Loan_Amount_Term", "Credit_History"]
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

numerical_columns = ["LoanAmount"]
for col in numerical_columns:
    data[col] = data[col].fillna(data[col].median())

# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features (X) and target (y)
X = data.drop(columns=["Loan_Status"])
y = data["Loan_Status"]

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Train XGBoost Classifier
xgb_model = XGBClassifier(random_state=42, scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                           eval_metric='logloss', use_label_encoder=False)
xgb_model.fit(X_train, y_train)

# Make predictions and calculate probabilities for threshold adjustment
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate ROC AUC and find the optimal threshold
fpr, tpr, thresholds = roc_curve(y_test, y_prob_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_prob_xgb)

# Find the optimal threshold where the difference between TPR and FPR is maximized
optimal_idx = (tpr - fpr).argmax()
optimal_threshold = thresholds[optimal_idx]

# Make predictions based on the optimal threshold
y_pred_threshold_xgb = (y_prob_xgb >= optimal_threshold).astype(int)

# Evaluate the model
accuracy_xgb = accuracy_score(y_test, y_pred_threshold_xgb)
classification_xgb = classification_report(y_test, y_pred_threshold_xgb)

# Define save path
save_path = "C:/Users/Public.NAWARAJ/Desktop/Code/Django/Loan-Prediction-System-main/user/"

# Save metrics
metrics_xgb = {
    "accuracy": accuracy_xgb,
    "classification_report": classification_xgb,
}
metrics_path_xgb = save_path + "metrics_xgb.pkl"
joblib.dump(metrics_xgb, metrics_path_xgb)

# Save the XGBoost model, label encoders, and any additional objects
xgb_model_path = save_path + "loan_xgb_model.pkl"
encoders_path_xgb = save_path + "label_encoders_xgb.pkl"

joblib.dump(xgb_model, xgb_model_path)              
joblib.dump(label_encoders, encoders_path_xgb)    
  
# Print results
print(f"XGBoost Model Accuracy: {accuracy_xgb:.2f}")
print(f"XGBoost ROC AUC Score: {roc_auc_xgb:.2f}")
print("Classification Report:")
print(classification_xgb)
