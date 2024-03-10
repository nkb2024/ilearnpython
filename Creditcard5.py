import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load the credit card application data (replace 'credit_card_data.csv' with your actual file path)
file_path = "C:\\Users\\nobin\\Documents\\mscAI\\research\\clean_dataset.csv"
data = pd.read_csv(file_path)
print(data.head())
print(data.columns)

data= data.dropna()
#print(data.head())
data= data.drop_duplicates()
#print(data.head())
LabelEncoder()
print(data.head())
StandardScaler()
# Separate features (X) and target variable (y)
X = data.drop('Age', axis=1)  # Assuming 'approval_status' is the target variable
y = data['Age']
from sklearn.preprocessing import OneHotEncoder

# ... (your data loading and preprocessing steps)

# One-Hot encode categorical features
encoder = OneHotEncoder(with_mean=False)  # Set sparse=False for easier handling
X_encoded = encoder.fit_transform(X)  # Transform categorical features

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
# # Standardize features (replace with your chosen scaling method if needed)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# ... (rest of your code for evaluation and visualization)
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
#
# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)
# # Standardize features (replace with your chosen scaling method if needed)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
