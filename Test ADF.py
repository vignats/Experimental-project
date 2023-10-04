# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:42:32 2023

@author: tanne
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
import csv
# Generate a sample time series
file = open('03-Oct-2023_patAnalysis_2.csv',"r")
data = csv.reader(file, delimiter = ",")
data = np.array(list(data))
for i in range(2343):    #Allows conversion of data into float by replacing the empty spots by 0
    for j in range(8):
        if data[i][j]== '' :
            data[i][j]=0
X = data[1:, 1:4].astype(float)
y = data[1: , 5].astype(float)
PAT =[]
Y_new =[]
for i in range(2342):    #Allows to plot the real data
    for j in range(3):
        if X[i][j]== 0 :
            X[i][j]= None
        elif X[i][j]!= 0 and j==2:
            PAT.append(X[i][2])
    if y[i]==0:
        y[i]=None  
    else :
        Y_new.append(y[i])
Y_new=np.array(Y_new)
PAT_filtered = np.array(PAT)      
# ADF Test
result = adfuller(Y_new, autolag='AIC')
print(f'ADF Statistic for BP: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
    
    
result2 = adfuller(PAT_filtered, autolag='AIC')
print(f'ADF Statistic for PAT: {result2[0]}')
print(f'n_lags: {result2[1]}')
print(f'p-value: {result2[1]}')
for key, value in result2[4].items():
    print('Critial Values:')
    print(f'   {key}, {value} \n')
    

resultsKPSS = kpss(Y_new)
print(f'KPSS Statistic for BP: {resultsKPSS[0]}')
print(f'n_lags: {resultsKPSS[1]}')
print(f'p-value: {resultsKPSS[1]}')
for key, value in resultsKPSS[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}\n')
    
resultsKPSS2 = kpss(PAT_filtered)
print(f'KPSS Statistic for PAT: {resultsKPSS2[0]}')
print(f'n_lags: {resultsKPSS2[1]}')
print(f'p-value: {resultsKPSS2[1]}')
for key, value in resultsKPSS2[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}\n')