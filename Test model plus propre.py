# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:32:38 2023

@author: tanne
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn import linear_model 
## Read excel in data frame 
df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv')

## Getting the data right with the correct index
time = df['wrist@(9mm,809nm)_delay_s']
data_squashed = df.dropna(subset=['wrist@(9mm,809nm)_filtered_pat_bottomTI'])
data_squashed = data_squashed.dropna(subset=['blood pressure_systolic'])
PAT= data_squashed['wrist@(9mm,809nm)_filtered_pat_bottomTI'].to_numpy()
BP = data_squashed['blood pressure_systolic'].to_numpy()


Y = BP
X_raw =PAT

#Initializing the model :
    
X_invert = (1/X_raw).reshape(-1, 1)
X_raw = X_raw.reshape(-1,1)

tscv = TimeSeriesSplit(n_splits=6)
model = MLPRegressor(activation = "tanh", solver="lbfgs", max_iter=100000)
X_train, X_test, Y_train, Y_test = train_test_split(X_invert,Y, test_size=0.3)
# Scaler = StandardScaler()
# Scaler.fit_transform(X_train)
# Scaler.transform(X_test)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print("Accuracy for Testing data : ", model.score(X_test, Y_test), "RMSE:", mean_squared_error(Y_test, Y_pred,squared=False))
# Training + testing and Accuracy 
for train_index, test_index in tscv.split(X_invert,Y):
    X_train, X_test = X_invert[train_index], X_invert[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    lasso = linear_model.Lasso(alpha=1)
    lasso.fit(X_train,y_train)
    y_pred = lasso.predict(X_test)
    print("Accuracy for Testing data : ", lasso.score(X_test, y_test), "RMSE:", mean_squared_error(y_test, y_pred,squared=False))
