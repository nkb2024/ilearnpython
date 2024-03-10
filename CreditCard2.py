# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the credit card application data (replace 'credit_card_data.csv' with your actual file path)
file_path = "C:\\Users\\nobin\\Documents\\mscAI\\research\\clean_dataset1.csv"
data = pd.read_csv(file_path)
print(data.head())
print(data.columns)
# Preprocessing steps (replace these with your specific data cleaning and feature engineering steps)
# 1. Handle missing values (e.g., using imputation techniques)
# 2. Encode categorical features (e.g., using one-hot encoding)
# 3. Scale numerical features (e.g., using StandardScaler)

# Separate features (X) and target variable (y)
X = data.drop('Approved', axis=1)  # Assuming 'approval_status' is the target variable
y = data['Approved']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (replace with your chosen scaling method if needed)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# (Optional) Save the trained model for future use (using libraries like pickle or joblib)
