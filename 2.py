import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

data = pd.read_csv('../../Desktop/data.csv')

print(data.head())

# missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

duplicate_count = data.duplicated().sum()
print(f"Duplicate Rows Found: {duplicate_count}")

#Check Correlation Between Features
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

#Checking Linear Relationships
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

plt.show()

#Creating Total_Budget to represents the total amount spent on advertising across all channels
data["Total_Budget"] = data["TV"] + data["Radio"] + data["Newspaper"]
data.head()

#Spliting Data into Training & Testing Sets
from sklearn.model_selection import train_test_split

X = data.drop(columns=['Sales'])
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data successfully split into training and testing sets!")

# Training Linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

#saving this file as pickle file
import pickle

with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model saved as 'model.pkl'")

