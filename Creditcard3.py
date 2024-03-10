# Import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns  # Optional for confusion matrix heatmap



# Load data (replace with your actual file path)
file_path = "C:\\Users\\nobin\\Documents\\mscAI\\research\\clean_dataset1.csv"
data = pd.read_csv(file_path)
print(data.columns)

# Separate features and target variable
X = data.drop('Income', axis=1)  # Assuming 'approval_status' is the target variable
y = data['Income']

# Standardize features (replace with your chosen scaling method if needed)
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
X_test = scaler.transform(X)

# Create and train the Logistic Regression model (with increased max_iter)
model = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence warning persists
model.fit(X_train, y_train)

# Get coefficients (weights)
coefficients = model.coef_.flatten()
feature_names = X.columns.tolist()

# Sort features by absolute coefficient values (ensure sorted_idx contains integers)
sorted_idx = np.argsort(np.abs(coefficients)).astype(int)  # Ensure integer indices

# Prepare data for visualization (optional)
plt.figure(figsize=(10, 6))
plt.barh(feature_names[sorted_idx], coefficients[sorted_idx])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature Name')
plt.title('Feature Importance in Credit Card Approval Prediction')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()

# Display or save the plot
# plt.show()  # Uncomment to display the plot
plt.savefig('feature_importance.png')  # Uncomment to save the plot

# (Optional) Confusion Matrix
y_pred = model.predict(X_test)  # Assuming you have predicted labels
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Credit Card Approval Prediction')
plt.show()

