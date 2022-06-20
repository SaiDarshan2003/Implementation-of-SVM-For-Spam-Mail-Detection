# Implementation of SVM For Spam Mail Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required library packages.
2. Import the dataset to operate on.
3. Split the dataset into required segments.
4. Predict the required output.
5. Run the program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Sai Darshan G
RegisterNumber:  212221240047
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv("/content/spam.csv",encoding='latin-1')
dataset
dataset.head()
dataset.info()
dataset.isnull().sum()
X = dataset["v1"].values
Y = dataset["v2"].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
Y_pred
from sklearn import metrics
accuracy = metrics.accuracy_score(Y_test,Y_pred)
accuracy
```
## Output:
![SVM For Spam Mail Detection](1.png)
![SVM For Spam Mail Detection](2.png)
![SVM For Spam Mail Detection](3.png)
![SVM For Spam Mail Detection](4.png)
![SVM For Spam Mail Detection](5.png)
![SVM For Spam Mail Detection](6.png)
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
