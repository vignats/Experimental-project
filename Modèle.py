# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:01:05 2023

@author: lilac
"""

import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error

## Read excel in data frame 
df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv') 

## Data processing
time = df['wrist@(9mm,809nm)_delay_s']
bp_systolic = df['blood pressure_systolic']
pat_filtred = df['wrist@(9mm,809nm)_filtered_pat_bottomTI']
correlation = bp_systolic.corr(pat_filtred)
spearmanr_corr = bp_systolic.corr(pat_filtred, method='spearman')
time = df['wrist@(9mm,809nm)_delay_s']

## Delete na values
data_squashed = df.dropna(subset=['wrist@(9mm,809nm)_filtered_pat_bottomTI'])
interp_func = interp1d(data_squashed['wrist@(9mm,809nm)_delay_s'], data_squashed['wrist@(9mm,809nm)_filtered_pat_bottomTI'], kind='linear', fill_value="extrapolate")
PAT_inversed = interp_func(time)

data_squashed_bp = df.dropna(subset=['blood pressure_systolic'])
interp_func_bp = interp1d(data_squashed_bp['wrist@(9mm,809nm)_delay_s'], data_squashed_bp['blood pressure_systolic'], kind='linear', fill_value="extrapolate")
BP = interp_func_bp(time)

## Inverse PAT
PAT = 1/PAT_inversed


## Train a model
X = pd.Series(PAT).values.reshape(-1, 1)
Y = pd.Series(BP).values.reshape(-1, 1)
X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)
model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print(model.score(X_test, Y_test))
print (mean_squared_error(Y_test, y_pred, squared=False))


## Use Kfold
from sklearn.model_selection import KFold
scores=[]
abss =0
curve = []

'''
for N in range (2,20,2): 

    kFold=KFold(n_splits=N)
    
    for train_index,test_index in kFold.split(X):
        X_train, X_test, Y_train, Y_test = X[train_index],X[test_index],Y[train_index],Y[test_index]
        model = LinearRegression()
        model.fit(X_train, Y_train)
        scores.append(model.score(X_test, Y_test))
        abss += 1
        curve.append(abss)
plt.scatter(curve, scores)

for index, accuracy in enumerate (scores):
    if accuracy == max(scores) :
       print(index/5) 
    
'''
abss =0
curve = []
kFold=KFold(n_splits=5)
for train_index,test_index in kFold.split(X):
    X_train, X_test, Y_train, Y_test = X[train_index],X[test_index],Y[train_index],Y[test_index]
    model = LinearRegression()
    model.fit(X_train, Y_train)
    scores.append(model.score(X_test, Y_test))
    abss+=1
    curve.append(abss)
plt.scatter(curve, scores)
