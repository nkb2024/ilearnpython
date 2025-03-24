import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.preprocessing import StandardScaler


#from Creditcard5 import encoder

# Load and preprocess your credit card approval data (replace with your data loading steps)
# ... (your data loading and preprocessing code)
# Load the credit card application data (replace 'credit_card_data.csv' with your actual file path)
file_path = "C:\\Users\\nobin\\Documents\\mscAI\\research\\clean_dataset05.csv"
data = pd.read_csv(file_path)
print(data.head())
print(data.columns)
#Handiling missing values
data= data.dropna()
#Handiling duplicates
data= data.drop_duplicates()

print("data.head()")
print(data.head())

#Label encoding
le=LabelEncoder()
for col in data.columns:
    if data[col].dtypes == object:
        data[col]= le.fit_transform(data[col])
print(data.dtypes)

print("---------------------------------------")
#standard scaling
# scaler=StandardScaler()
# for col in data.columns:
#     if data[col].dtypes != object:
#         data[col]=scaler.fit_transform(data[[col]])


# Assuming you have a DataFrame named 'data' with columns 'Debt', 'ZipCode', 'Income', 'Age', and 'YearsEmployed'
# You need to replace 'data' with your actual DataFrame

# Before applying StandardScaler
print("Instances before applying standard scaler:")
print(data)

# Select numerical columns for scaling
numerical_cols = ['Debt', 'ZipCode', 'Income', 'Age', 'YearsEmployed']
numerical_data = data[numerical_cols]

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(numerical_data)

# Convert the scaled data back to a DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols)

# Concatenate the scaled numerical features with non-numerical features if any
# For example:
# scaled_data_with_categorical = pd.concat([scaled_df, data[['Categorical_Column1', 'Categorical_Column2']]], axis=1)

# After applying StandardScaler
print("Instances after applying standard scaler:")
print(scaled_df)
print("DataFrame after applying standard scaler")
le=LabelEncoder()
for col in data.columns:
    if data[col].dtypes == object:
        data[col]= le.fit_transform(data[col])
print(data.dtypes)

# You can then use 'scaled_data' or 'scaled_df' in your further analysis or modeling.

print("---------------------------------------")
print("---------------------------------------")

print("---------------------------------------")
#
data = pd.get_dummies(data, columns=['Industry'], drop_first=True)
data = pd.get_dummies(data, columns=['Ethnicity'], drop_first=True)
data = pd.get_dummies(data, columns=['Citizen'], drop_first=True)

# Separate features (X) and target variable (y)
X = data.drop('Approved', axis=1)  # Assuming 'approval_status' is the target variable
y = data['Approved']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train various classification models
models = {}

# Logistic Regression
models['Logistic Regression'] = LogisticRegression(random_state=42)
models['Logistic Regression'].fit(X_train, y_train)

# Support Vector Machine (SVM)
models['SVM'] = SVC(random_state=42)  # Adjust hyperparameters as needed
models['SVM'].fit(X_train, y_train)

# Random Forest
models['Random Forest'] = RandomForestClassifier(n_estimators=10, random_state=42)  # Adjust hyperparameters as needed
models['Random Forest'].fit(X_train, y_train)
#Bagging Classifier
models['Bagging Classifier'] = BaggingClassifier(n_estimators=10, random_state=42)
models['Bagging Classifier'].fit(X_train, y_train)

# # K-Nearest Neighbors (KNN)
# models['KNN'] = KNeighborsClassifier(n_neighbors=5)  # Adjust hyperparameters as needed
# models['KNN'].fit(X_train, y_train)

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

