# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    roc_curve
from sklearn.model_selection import cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load the dataset
url = "https://raw.githubusercontent.com/samuelcortinhas/credit-card-approval-clean-data/main/credit_card_approval_clean_data.csv"
file_path = "C:\\Users\\nobin\\Documents\\mscAI\\research\\clean_dataset.csv"
df = pd.read_csv(file_path)
print(df.columns)

# Data Preprocessing
# (Include the code for duplicate removal, handling missing data, label encoding, and standard scaling)

# Data Splitting
X = df.drop("Approved", axis=1)
y = df["Approved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base estimators
lrc = LogisticRegression()
rfc = RandomForestClassifier()
svc = SVC(probability=True)

# Ensemble Bagging Classifier
ensemble_model = VotingClassifier(estimators=[('lrc', lrc), ('rfc', rfc), ('svc', svc)], voting='soft')

# Create BaggingClassifier with the ensemble model as the base estimator
bagging_classifier = BaggingClassifier(base_estimator=ensemble_model, n_estimators=10, random_state=42)

# Train the bagging classifier
bagging_classifier.fit(X_train, y_train)


# Now you can use bagging_classifier for predictions and evaluation

# Model Validation
cv_scores = cross_val_score(bagging_classifier, X_train, y_train, cv=5, scoring='accuracy')
cv_predictions = cross_val_predict(bagging_classifier, X_train, y_train, cv=5, method='predict_proba')

# Testing the Models
y_pred = bagging_classifier.predict(X_test)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, cv_predictions[:, 1])

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizations (Include any relevant plots or visualizations)
plt.figure(figsize=(12, 6))

# Confusion Matrix Heatmap
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# ROC Curve
plt.subplot(1, 2, 2)
fpr, tpr, _ = roc_curve(y_test, cv_predictions[:, 1])
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.tight_layout()
plt.show()

# Print Results
print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
