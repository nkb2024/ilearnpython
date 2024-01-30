import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
size = np.array([1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700])
price = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])

# Reshape the data
size = size.reshape((-1, 1))

# Create and fit the linear regression model
model = LinearRegression().fit(size, price)

# Make predictions
predicted_prices = model.predict(size)

# Plot the data and regression line
plt.scatter(size, price, color='blue', label='Actual Prices')
plt.plot(size, predicted_prices, color='red', label='Linear Regression')
plt.xlabel('House Size (sq. ft.)')
plt.ylabel('House Price ($)')
plt.title('Linear Regression: House Price Prediction')
plt.legend()
plt.show()
