# EXPERIMENT NO: 9
# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee
## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program & Output:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: viswanadham venkata sai sruthi
RegisterNumber:  212223100061
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("/content/Salary (2).csv")
data.head()
```
![image](https://github.com/user-attachments/assets/79dfd9fe-5c9f-4864-be9b-28e3ee1bf644)
```
data.info()
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/0bdd4d56-190d-4240-8b4a-d8890d223298)
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
```
![image](https://github.com/user-attachments/assets/ef817efc-6ed0-499a-bd97-260eaf13b020)
```
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor,plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
```
![image](https://github.com/user-attachments/assets/0266cdd5-5da1-4e05-bbbf-d749343e7510)
```
r2=metrics.r2_score(y_test,y_pred)
r2
```
![image](https://github.com/user-attachments/assets/91aea4fe-1252-49af-b185-9489eb8dd121)
```
dt.predict([[5,6]])
```
![image](https://github.com/user-attachments/assets/4edb6291-8ee4-4b79-adf2-e1bf0dd6e303)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
