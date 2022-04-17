# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 19:20:19 2022

@author: sjha
"""

#linear regression model prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
from sklearn import linear_model

#Load data
Data = pd.read_csv('data.csv')
Data
Data.shape
Data.keys()

X = Data['YearsExperience']
Y = Data['Salary']
#slit data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33)

#trasnform data into 2D array
X_train = np.array(X_train).reshape(len(X_train),1)
X_test = np.array(X_test).reshape(len(X_test),1)
Y_train = np.array(Y_train).reshape(len(Y_train),1)
Y_test = np.array(Y_test).reshape(len(Y_test),1)

#Train model
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

#predict trained data set
Y_train_pred = model.predict(X_train)

#Plot trained data set
pp.scatter(X_train,Y_train,color = 'red', label='True data')
pp.plot(X_train,Y_train_pred, color = 'blue', label = 'predict data')
pp.legend()
pp.xlabel("Years of Experience")
pp.ylabel("Salary")
pp.show()

#predict trained data set
Y_test_pred = model.predict(X_test)

#plot test data
pp.scatter(X_test,Y_test,color = 'red', label='True data')
pp.plot(X_test,Y_test_pred, color = 'blue', label = 'predict data')
pp.legend()
pp.xlabel("Years of Experience")
pp.ylabel("Salary")
pp.show()

#Model evaluation
print("Mean square error:", round(sm.mean_squared_error(Y_test, Y_test_pred),2))
print("Exlained varience scor: ", round(sm.explained_variance_score(Y_test, Y_test_pred),2))
print("R2 score:", round(sm.r2_score(Y_test, Y_test_pred),2))
