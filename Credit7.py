# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the credit card application data (replace 'credit_card_data.csv' with your actual file path)
file_path = "C:\\Users\\nobin\\Documents\\mscAI\\research\\clean_dataset.csv"
data = pd.read_csv(file_path)
print(data.head())
print(data.columns)

# Data preprocessing
# (Include your data preprocessing steps)
# Assuming 'Industry' is a categorical variable
# One-hot encode the 'Industry' column
data = pd.get_dummies(data, columns=['Industry'], drop_first=True)
data = pd.get_dummies(data, columns=['Ethnicity'], drop_first=True)
data = pd.get_dummies(data, columns=['Citizen'], drop_first=True)


# Split the data into features (X) and target variable (y)
X = data.drop('Approved', axis=1)
y = data['Approved']

# Label encoding for categorical variables
#le = LabelEncoder()
#X['Gender'] = le.fit_transform(X['Gender'])
# Continue label encoding for other categorical variables if needed

# Standard scaling for numerical variables
scaler = StandardScaler()
X[['Age', 'Debt', 'YearsEmployed', 'CreditScore', 'Income']] = scaler.fit_transform(X[['Age', 'Debt', 'YearsEmployed', 'CreditScore', 'Income']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:\n", conf_matrix)
