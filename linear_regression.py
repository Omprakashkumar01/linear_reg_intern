# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a simple dataset (Alternatively, you can load your own dataset)
data = {
    'Feature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Target': [3, 4, 2, 5, 6, 7, 8, 9, 10, 12]
}

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Step 2: Split the dataset into features and target
X = df[['Feature']]  # Features
y = df['Target']  # Target variable

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model performance metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Step 7: Plot the regression line with the data points
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Regression line')

# Adding labels and title
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression: Feature vs Target')
plt.legend()

# Show plot
plt.show()
