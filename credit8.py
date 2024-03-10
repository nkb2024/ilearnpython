import pandas as pd
from pyexpat import features

#from Demos.win32cred_demo import target
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from CreditCard import ensemble_model
from sklearn import base_estimator

# Load the dataset (replace 'path_to_dataset.csv' with the actual path)
file_path = "C:\\Users\\nobin\\Documents\\mscAI\\research\\clean_dataset.csv"
data = pd.read_csv(file_path)
print(data.head())
print(data.columns)

# Assume 'features' is a DataFrame containing the input features and 'target' is a Series containing the target variable
# Adjust this based on the actual column names in your dataset
data = pd.get_dummies(data, columns=['Industry'], drop_first=True)
data = pd.get_dummies(data, columns=['Ethnicity'], drop_first=True)
data = pd.get_dummies(data, columns=['Citizen'], drop_first=True)


# Split the data into features (X) and target variable (y)
X = data.drop('Approved', axis=1)
y = data['Approved']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lrc_model = LogisticRegression()
lrc_model.fit(X_train, y_train)
lrc_predictions = lrc_model.predict(X_test)
lrc_accuracy = metrics.accuracy_score(y_test, lrc_predictions)

# Random Forest Classifier
rfc_model = RandomForestClassifier()
rfc_model.fit(X_train, y_train)
rfc_predictions = rfc_model.predict(X_test)
rfc_accuracy = metrics.accuracy_score(y_test, rfc_predictions)

# Support Vector Classifier
svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_predictions = svc_model.predict(X_test)
svc_accuracy = metrics.accuracy_score(y_test, svc_predictions)

# Ensemble Bagging Classifier with Logistic Regression as base estimator
# ensemble_model = BaggingClassifier(base_estimator=lrc_model, n_estimators=10, random_state=42)
# ensemble_model.fit(X_train, y_train)
# ensemble_predictions = ensemble_model.predict(X_test)
# ensemble_accuracy = metrics.accuracy_score(y_test, ensemble_predictions)

# ensemble_predictions = ensemble_model.predict(X_test)
#
# ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
# ensemble_precision = precision_score(y_test, ensemble_predictions)
# ensemble_recall = recall_score(y_test, ensemble_predictions)
# ensemble_f1 = f1_score(y_test, ensemble_predictions)

# Display the results
print("Logistic Regression Accuracy:", lrc_accuracy)
print("Random Forest Classifier Accuracy:", rfc_accuracy)
print("Support Vector Classifier Accuracy:", svc_accuracy)
#print("Ensemble Bagging Classifier Accuracy:", ensemble_accuracy)
