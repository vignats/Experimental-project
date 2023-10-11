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
time=[]
Y_new =[]
for i in range(2342):    #Allows to plot the real data
    for j in range(3):
        if X[i][j]== 0 :
            X[i][j]= None
        elif X[i][j]!= 0 and j==2:
            PAT.append(X[i][2])
            time.append(X[i][0])
    if y[i]==0:
        y[i]=None  
    else :
        Y_new.append(y[i])
Y_new=np.array(Y_new)
PAT_filtered = np.array(PAT)
      
# ADF Test
result = adfuller(Y_new, autolag='AIC')
print(f'ADF Statistic for BP: {result[0]}')
print(f'n_lags: {result[2]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}\n')
    
    
result2 = adfuller(PAT_filtered, regression='ct', autolag='AIC')
print(f'ADF Statistic for PAT: {result2[0]}')
print(f'n_lags: {result2[2]}')
print(f'p-value: {result2[1]}')
for key, value in result2[4].items():
    print('Critial Values:')
    print(f'   {key}, {value} \n')
    

resultsKPSS = kpss(Y_new)
print(f'KPSS Statistic for BP: {resultsKPSS[0]}')
print(f'n_lags: {resultsKPSS[2]}')
print(f'p-value: {resultsKPSS[1]}')
for key, value in resultsKPSS[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}\n')
    
resultsKPSS2 = kpss(PAT_filtered, regression='ct')
print(f'KPSS Statistic for PAT: {resultsKPSS2[0]}')
print(f'n_lags: {resultsKPSS2[2]}')
print(f'p-value: {resultsKPSS2[1]}')
for key, value in resultsKPSS2[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}\n')
    
#Detrending the PAT dataset
PAT_diff = np.zeros(1490)
PAT_diff[0]= PAT_filtered[0]
PAT_diff[1489] = PAT_filtered[1489]
for i in range (1,len(PAT_filtered)-1,1):
    PAT_diff[i]= PAT_filtered[i]-PAT_filtered[i-1]
    
resultsKPSS2 = kpss(np.abs(PAT_diff))
print(f'KPSS Statistic for  Differenced PAT: {resultsKPSS2[0]}')
print(f'n_lags: {resultsKPSS2[2]}')
print(f'p-value: {resultsKPSS2[1]}')
for key, value in resultsKPSS2[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}\n')
    
result2 = adfuller(np.abs(PAT_diff), autolag='AIC')
print(f'ADF Statistic for Differenced PAT: {result2[0]}')
print(f'n_lags: {result2[2]}')
print(f'p-value: {result2[1]}')
for key, value in result2[4].items():
    print('Critial Values:')
    print(f'   {key}, {value} \n')
    
index = 150
roll_avg =[]
std=[]
for i in range (len(PAT_diff)-1-index):
    roll_avg.append(np.abs(PAT_diff[i:i+index]).mean())
    std.append(np.abs(PAT_diff[i:i+index]).std())
fig1 = plt.figure()
plt.subplot(1,1,1), plt.plot(time,np.abs(PAT_diff), label='PAT'),plt.ylabel("differed PAT"), plt.xlabel('Time (s)')  
fig2 = plt.figure()
plt.subplot(1,1,1), plt.plot(roll_avg, label='PAT'),plt.ylabel("mean differed PAT")
fig3 = plt.figure()
plt.subplot(1,1,1), plt.plot(std, label='PAT'),plt.ylabel("std differed PAT")    
