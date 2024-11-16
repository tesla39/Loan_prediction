# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = "C:/Users/Public.NAWARAJ/Desktop/Code/Django/Loan-Prediction-System-main/train.csv"
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

# Use Random Forest Classifier to handle non-linearity and improve performance
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Make predictions and calculate probabilities for threshold adjustment
y_prob = model.predict_proba(X_test)[:, 1]  # Probability for Class 1

# Evaluate ROC AUC and find the optimal threshold
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Find the optimal threshold where the difference between TPR and FPR is maximized
optimal_idx = (tpr - fpr).argmax()
optimal_threshold = thresholds[optimal_idx]

# Make predictions based on the optimal threshold
y_pred_threshold = (y_prob >= optimal_threshold).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_threshold)
classification = classification_report(y_test, y_pred_threshold)
#confusion = confusion_matrix(y_test, y_pred_threshold)

metrics = {
    "accuracy": accuracy,
    "classification_report": classification,
    #"confusion_matrix": confusion.tolist()  # Convert numpy array to list for serialization
}

# Define save path
save_path = "C:/Users/Public.NAWARAJ/Desktop/Code/Django/Loan-Prediction-System-main/user/"

# Save metrics
metrics_path = save_path + "metrics2.pkl"
joblib.dump(metrics, metrics_path)

# Save the model, label encoders, and any scalers or additional objects
model_path = save_path + "loan_model2.pkl"
encoders_path = save_path + "label_encoders2.pkl"

joblib.dump(model, model_path)                     # Save the trained model
joblib.dump(label_encoders, encoders_path)         # Save label encoders

# Print results
print(f"Model Accuracy: {accuracy:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")
print("Classification Report:")
print(classification)

