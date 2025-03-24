import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the credit card application data (replace 'credit_card_data.csv' with your actual file path)
file_path = "C:\\Users\\nobin\\Documents\\mscAI\\research\\Heart_health1.csv"
data = pd.read_csv(file_path)
print(data.head())
#print(data.columns)

# Data preprocessing
# (Include your data preprocessing steps)
# Assuming 'Industry' is a categorical variable
# One-hot encode the 'Industry' column
# data = pd.get_dummies(data, columns=['Name'], drop_first=True)
# data = pd.get_dummies(data, columns=['Gender'], drop_first=True)
# data = pd.get_dummies(data, columns=['Blood Pressure(mmHg)'], drop_first=True)
# data = pd.get_dummies(data, columns=['Smoker'], drop_first=True)
#print(data.head())
#print(data.columns)
# Split the data into features (X) and target variable (y)
X = data.drop('Heart Attack', axis=1)
Y = data['Heart Attack']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define and train various classification models
models = {}

# Logistic Regression
models['Logistic Regression'] = LogisticRegression(random_state=42)
models['Logistic Regression'].fit(X_train, y_train)

# Support Vector Machine (SVM)
models['SVM'] = SVC(random_state=42)  # Adjust hyperparameters as needed
models['SVM'].fit(X_train, y_train)

# Random Forest
models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)  # Adjust hyperparameters as needed
models['Random Forest'].fit(X_train, y_train)

# K-Nearest Neighbors (KNN)
models['KNN'] = KNeighborsClassifier(n_neighbors=5)  # Adjust hyperparameters as needed
models['KNN'].fit(X_train, y_train)

# Evaluate model performance
print("Model Performance Comparison:")
print("{:20s} {:8s} {:8s} {:8s} {:8s}".format('Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'))
print("-"*70)

for model_name, model in models.items():
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print("{:20s} {:8.4f} {:8.4f} {:8.4f} {:8.4f}".format(model_name, accuracy, precision, recall, f1))
