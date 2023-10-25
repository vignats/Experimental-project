# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:25:47 2023

@author: tanne
"""

import pandas as pd 
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

def interpol(df, name_col):
   time = df['wrist@(9mm,809nm)_delay_s']
   data_squashed = df.dropna(subset=['wrist@(9mm,809nm)_filtered_pat_bottomTI'])
   interp_func = interp1d(data_squashed['wrist@(9mm,809nm)_delay_s'].array, data_squashed['wrist@(9mm,809nm)_filtered_pat_bottomTI'].array, kind='linear', fill_value="extrapolate")
   continuous_values = interp_func(time)
   return continuous_values 

if __name__ == '__main__' :
 df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv')
 name_col ='wrist@(9mm,809nm)_filtered_pat_bottomTI'
 mat = interpol(df,name_col)
 X_raw = mat[23:2325]
 X_invert = (1/X_raw).reshape(-1, 1)
 X_raw = X_raw.reshape(-1,1)
 y = df['blood pressure_systolic'].dropna().to_numpy()
 
 #Modelling phase :
Scaler = StandardScaler()
Scaler.fit_transform(X_raw)
tscv = TimeSeriesSplit(n_splits=6)
for train_index, test_index in tscv.split(X_raw,y):
    X_train, X_test = X_invert[train_index], X_invert[test_index]
    y_train, y_test = y[train_index], y[test_index]
    SV = SVR(kernel='rbf')
    SV.fit(X_train, y_train)
    y_pred = SV.predict(X_test)
    print("Accuracy for Testing data : ", SV.score(X_test, y_test), "RMSE:", mean_squared_error(y_test, y_pred,squared=False))
