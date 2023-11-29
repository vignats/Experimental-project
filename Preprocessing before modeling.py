# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:14:08 2023

@author: tanne
"""
import pandas as pd 
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

def interpol(df, name_col):
   time = df['wrist@(9mm,809nm)_delay_s']
   data_squashed = df.dropna(subset=['wrist@(9mm,809nm)_filtered_pat_bottomTI'])
   interp_func = interp1d(data_squashed['wrist@(9mm,809nm)_delay_s'].array, data_squashed['wrist@(9mm,809nm)_filtered_pat_bottomTI'].array, kind='linear', fill_value="extrapolate")
   continuous_values = interp_func(time)
   return continuous_values 
    
def Detrend(mat):
    Diff = np.zeros(mat.shape)
    Diff[0] = mat[0]
    Diff[len(mat)-1]=Diff[len(mat)-1]
    for i in range (1,len(mat)-1,1):
        Diff[i]= np.abs(mat[i]-mat[i-1])
    return Diff

if __name__ == '__main__' :
    
 df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv')
 name_col ='wrist@(9mm,809nm)_filtered_pat_bottomTI'
 mat = interpol(df,name_col)
 Diff = Detrend(mat)
 X_Stationnary = Diff[23:2325]
 X_raw = mat[23:2325]
 y = df['blood pressure_systolic'].dropna().to_numpy()
 
 #Modelling phase :
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_raw, y, train_size=0.7)
Xtrain = Xtrain.reshape(-1, 1)
Xtest = Xtest.reshape(-1, 1)
Scaler = StandardScaler()
Xtrain_scale = Scaler.fit_transform(Xtrain)
Xtest_scale = Scaler.transform(Xtest)
train_regr = MLPRegressor(activation = "tanh", solver="lbfgs", max_iter=100000)
SV = SVR(kernel='linear')
SV2 = SVR(kernel='rbf')
SV.fit(Xtrain,Ytrain)
train_regr.fit(Xtrain_scale, Ytrain)
SV2.fit(Xtrain, Ytrain)

Ytest_pred = train_regr.predict(Xtest_scale)
Ytest_pred_scale = SV2.predict(Xtest)

print("Normal data", mean_squared_error(Ytest, Ytest_pred,squared=False), "R2 score : ", r2_score(Ytest, Ytest_pred))
print("Scaled data", mean_squared_error(Ytest, Ytest_pred_scale,squared=False), "R2 score : ", r2_score(Ytest, Ytest_pred_scale))