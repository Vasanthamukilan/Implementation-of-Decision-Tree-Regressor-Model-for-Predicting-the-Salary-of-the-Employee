# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import dataset and get data info
2. check for null values
3. Map values for position column
4. Split the dataset into train and test set
5. Import decision tree regressor and fit it for data 
6. Calculate MSE,R2 and y predict.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:Vasanthamukilan M
RegisterNumber:212222230167
*/
```
```python
import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
l0=LabelEncoder()

data["Position"]=l0.fit_transform(data['Position'])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## Output:
### data.head()
![Screenshot 2024-04-03 142119](https://github.com/Vasanthamukilan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119559694/127bd0bc-3d79-4fb0-89c4-de4fd657a447)

### data.info()
![Screenshot 2024-04-03 142128](https://github.com/Vasanthamukilan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119559694/d4f9ee5a-cf51-4969-a150-ab02ee72afa3)

### isnull() and sum()
![Screenshot 2024-04-03 142135](https://github.com/Vasanthamukilan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119559694/1c1c24e9-dc3d-4a9e-b101-2eb7221a749b)

### data.head() for salary
![Screenshot 2024-04-03 142144](https://github.com/Vasanthamukilan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119559694/7ff4c8a2-00eb-4a4a-ba4d-242d377a7036)

### MSE Value
![Screenshot 2024-04-03 142152](https://github.com/Vasanthamukilan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119559694/4a2754e6-f64e-42b0-be78-1b61a4b2a31a)

### r2 value 
![Screenshot 2024-04-03 142158](https://github.com/Vasanthamukilan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119559694/85af504b-8a0c-4514-bc77-61b527d27107)

### data prediction
![Screenshot 2024-04-03 142208](https://github.com/Vasanthamukilan/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119559694/450e0ae4-6ec0-4035-b6c8-9cc5502cdecd)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
