import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Prepare the dataset
# Features (X): [Area in sq ft, Number of Rooms]
X = np.array([[500, 2], [700, 3], [1000, 4], [1200, 5]])
# Target (y): Price of the house in Lakhs
y = np.array([30, 45, 65, 80])

# Step 2: Initialize and train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 3: Make a prediction for a new house (e.g., 800 sq ft and 3 rooms)
new_house = np.array([[800, 3]])
prediction = model.predict(new_house)
print(f"Predicted Price for 800 sq ft & 3 rooms: {prediction[0]:.2f} Lakhs")

# Step 4: Visualization
# Plotting actual data points (Using only Area for 2D visualization)
plt.scatter(X[:, 0], y, color='blue', label='Actual Data')

# Plotting the model's regression line based on predictions
plt.plot(X[:, 0], model.predict(X), color='red', label='Regression Line')

# Highlighting our predicted point on the graph
plt.scatter(800, prediction[0], color='green', marker='x', s=100, label='Predicted Point')

# Adding graph details
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (Lakhs)')
plt.title('Multiple Linear Regression: Predicting House Prices')
plt.legend() # Shows the index box for colors
plt.show()