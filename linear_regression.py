# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:55:08 2023

@author: salome
"""
import pandas as pd 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm

from utiles import process_data


X, y = process_data('03-Oct-2023_patAnalysis_2.csv')
# DATASET SPLTING

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

reg = LinearRegression()
reg.fit(X_train, y_train)
print(f"The linear regression score on the training set is : {reg.score(X_train, y_train)} \nThe coefficients are : {reg.coef_}")

y_pred = reg.predict(X_test)
print(f"The linear regression score on the testing set is : {reg.score(X_test, y_test)}")

clf_lin = svm.SVR(kernel='linear', C=1)
clf_lin.fit(X_train ,y_train)
y_pred_lin = clf_lin.predict(X_test)
print(f'The accuracy of the linear model is : {clf_lin.score(X_test, y_test)}')

clf_rbf = svm.SVR(kernel='rbf', C=1)
clf_rbf.fit(X_train ,y_train)
y_pred_rbf = clf_rbf.predict(X_test)
print(f'The accuracy of the gaussian model is : {clf_rbf.score(X_test, y_test)}')