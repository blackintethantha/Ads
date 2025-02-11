import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load data
data = pd.read_csv('../../Desktop/data.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Check for duplicate rows
duplicate_count = data.duplicated().sum()
print(f"Duplicate Rows Found: {duplicate_count}")

# Visualize correlation between features
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Visualize linear relationships between features and target
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.scatterplot(x=data["TV"], y=data["Sales"])
plt.title("TV vs Sales")

plt.subplot(1, 3, 2)
sns.scatterplot(x=data["Radio"], y=data["Sales"])
plt.title("Radio vs Sales")

plt.subplot(1, 3, 3)
sns.scatterplot(x=data["Newspaper"], y=data["Sales"])
plt.title("Newspaper vs Sales")

plt.tight_layout()
plt.show()

# Create a new feature: Total_Budget
data["Total_Budget"] = data["TV"] + data["Radio"] + data["Newspaper"]
print(data.head())

# Split data into features (X) and target (y)
X = data.drop(columns=["Sales"])
y = data["Sales"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data successfully split into training and testing sets!")

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Save the model as a pickle file
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model saved as 'model.pkl'")
