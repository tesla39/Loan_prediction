import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib 

# Load the dataset
file_path = "C:/Users/Public.NAWARAJ/Desktop/Code/Django/Loan-Prediction-System-main/user/loan.csv"
data = pd.read_csv(file_path)

# Drop irrelevant columns
data = data.drop(columns=["Loan_ID"])

# Handling of missing values

categorical_columns = ["Gender", "Married", "Self_Employed", "Dependents", "Loan_Amount_Term", "Credit_History"]
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Filling numerical features with median
numerical_columns = ["LoanAmount"]
for col in numerical_columns:
    data[col] = data[col].fillna(data[col].median())

# Encoding
label_encoders = {}
for col in data.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    original_values = data[col].unique()  # Get original string values
    encoded_values = le.fit_transform(data[col])  # Encode categorical values
    
    data[col] = encoded_values  # Assign the encoded values to the column
    label_encoders[col] = le

    # Debug: Print encoding mapping
    # print(f"Column: {col}")
    # for original, encoded in zip(original_values, le.transform(original_values)):
    #     print(f"Original Value: {original} --> Encoded Value: {encoded}")
    # print('-' * 50)

# Define features (X) and target (y)
X = data.drop(columns=["Loan_Status"])
y = data["Loan_Status"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (important for Naive Bayes)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handling imbalance 
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train_balanced, y_train_balanced)

# Prediction probabilities
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability for class 1

#Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# optimal threshold where precision and recall are balanced (or maximize F1-score)
optimal_threshold = thresholds[np.argmax(precision + recall)] 

# Adjust predictions based on the new threshold
y_pred_adjusted = (y_prob >= optimal_threshold).astype(int)

# Evaluate the adjusted model
accuracy = accuracy_score(y_test, y_pred_adjusted)
classification = classification_report(y_test, y_pred_adjusted)
confusion = confusion_matrix(y_test, y_pred_adjusted)
metrics = {
    "accuracy": accuracy,
    "classification_report": classification,
    "confusion_matrix": confusion.tolist()  # Convert numpy array to list for serialization
}

metrics_path = "C:/Users/Public.NAWARAJ/Desktop/Code/Django/Loan-Prediction-System-main/user/metrics.pkl"
joblib.dump(metrics, metrics_path)
# Print results
print(f"Adjusted Model Accuracy: {accuracy:.2f}")
print("Adjusted Classification Report:")
print(classification)

# Save the model, scaler, and label encoders 
save_path = "C:/Users/Public.NAWARAJ/Desktop/Code/Django/Loan-Prediction-System-main/user/"

# Save the model, scaler, and label encoders to the specified directory
joblib.dump(model, save_path + 'loan_model.pkl')               # Save the trained model
joblib.dump(scaler, save_path + 'scaler.pkl')                   # Save the scaler
joblib.dump(label_encoders, save_path + 'label_encoders.pkl')   


