import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

file_path = "C:\\Users\\nobin\\Documents\\mscAI\\research\\clean_dataset.csv"
data = pd.read_csv(file_path)
print(data.head())

# Separate features (X) and target variable (y)
X = data.drop('Approved', axis=1)  # Assuming 'approval_status' is the target variable
y = data['Approved']