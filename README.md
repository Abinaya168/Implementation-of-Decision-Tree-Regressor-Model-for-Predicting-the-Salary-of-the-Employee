# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries. 2.Upload the dataset and check for any null values using .isnull() function. 3.Import LabelEncoder and encode the dataset. 4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset. 5.Predict the values of arrays. 6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset. 7.Predict the values of array. 8.Apply to new unknown values

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: P.ABINAYA
RegisterNumber: 212225230003 
*/
import pandas as pd
import os

# Show current files in folder
print("Files in current directory:")
print(os.listdir())

# Load dataset
# Make sure Salary.csv is present in this folder
data = pd.read_csv("Salary.csv")

# Display first 5 rows
print("\nDataset:")
print(data.head())

# Dataset information
print("\nDataset Info:")
print(data.info())

# Check null values
print("\nNull Values:")
print(data.isnull().sum())

# Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Convert Position column into numeric values
data["Position"] = le.fit_transform(data["Position"])

print("\nAfter Label Encoding:")
print(data.head())

# Selecting features and target
x = data[["Position", "Level"]]
y = data["Salary"]

print("\nFeatures:")
print(x.head())

print("\nTarget:")
print(y.head())

# Splitting dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=2
)

# Import and train model
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=2)

# Train model
dt.fit(x_train, y_train)

# Prediction on test data
y_pred = dt.predict(x_test)

print("\nPredicted Values:")
print(y_pred)

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

# Predict salary for new input
# Example:
# Position encoded value = 5
# Level = 6

prediction = dt.predict([[5, 6]])

print("\nPredicted Salary:", prediction[0])
```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
<img width="1022" height="842" alt="Screenshot 2026-05-20 101837" src="https://github.com/user-attachments/assets/caadfc1d-9725-457b-9e6d-efa4ee8be59f" />
<img width="822" height="728" alt="Screenshot 2026-05-20 101851" src="https://github.com/user-attachments/assets/4f4c7530-bb9d-45cc-b0eb-b7be85f04a9d" />
<img width="1527" height="582" alt="Screenshot 2026-05-20 101909" src="https://github.com/user-attachments/assets/721baf45-0538-432a-bdfd-09a692296af5" />




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
