BLENDED_LEARNING

Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

Equipments Required: 

Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

Algorithm

1.Load the dataset, separate features and target, scale the features using MinMaxScaler, and encode the target labels using LabelEncoder.

2.Split the dataset into training and testing sets using train_test_split() with stratified sampling.

3.Train a Logistic Regression model with L2 regularization (multinomial) on the training data and make predictions on the test data. 

4.Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix, then visualize the confusion matrix using a heatmap.

Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: SHRIHARI M
RegisterNumber: 212225230265
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#Load the dataset
df=pd.read_csv('food_items (1).csv')
#inspect the dataset
print('Name: SHRIHARI M ')
print('Reg. No:212225230265  ')
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
X_raw = df.iloc[:,:-1]
y_raw = df.iloc[:,-1:]
scaler= MinMaxScaler()
#Scaling the raw input features
X= scaler.fit_transform(X_raw)
#Create a LabelEncoder object
label_encoder = LabelEncoder()
#Encode the target variable
y= label_encoder.fit_transform(y_raw.values.ravel())
#Note the ravel() function flattens the vector


#First, let's split the the training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify =y, random_state=123)

#L2 penalty to shrink coefficients without removing any features from the model
penalty='l2'

#Our classification problem is multinomial
multi_class='multinomial'

#Use of lbfgs for L2 penalty and multinomial classes
solver= 'lbfgs'

#Max iteration=1000
max_iter=1000

#Define a logistic regression model with the  above arguments
l2_model = LogisticRegression(
    random_state=123,
    penalty=penalty,
    multi_class=multi_class, 
    solver=solver, 
    max_iter=max_iter
)
l2_model.fit(X_train, y_train)

y_pred= l2_model.predict(X_test)
print('Name:SHRIHARI')
print('Reg. No:25013276')
print("\nModel Evaluation:")
print("Accuracy:",accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Confusion Matrix
conf_matrix= confusion_matrix(y_test, y_pred)
print(conf_matrix)

print('Name:SHRIHARI M')
print('Reg. No:212225230265 ')
```
Output:
<img width="644" height="529" alt="Screenshot 2026-03-25 145433" src="https://github.com/user-attachments/assets/b14becbb-74d8-4ea6-b64d-06665b61c043" />


<img width="629" height="743" alt="Screenshot 2026-03-25 145441" src="https://github.com/user-attachments/assets/5e77c9b3-4f5c-405f-8fd8-289f75806bc0" />

<img width="553" height="103" alt="Screenshot 2026-03-25 145446" src="https://github.com/user-attachments/assets/ac38b86b-404f-46b7-8058-db0077d560ce" />

Result:

Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
