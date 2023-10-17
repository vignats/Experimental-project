#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:24:20 2023

@author: mariafoyen
"""

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import csv


# read excel in data frame 
file = open("/Users/mariafoyen/Documents/Experimental-project/03-Oct-2023_patAnalysis_2.csv", "r")
data = csv.reader(file, delimiter = ",")
data = np.array(list(data)) #Making data as an arrray


for i in range(2343):
    for j in range(8):
        if data[i][j]== '' :
            data[i][j]=0


time = data[1:, 1].astype(float) 
pat_raw = data[1:,2].astype(float)
pat_filt = data[1:,3].astype(float)

bp_systolic  = data[1: , 5].astype(float)
bp_mean = data[1:, 6].astype(float)
bp_diastolic = data[1:, 7].astype(float)


        

" Plotting data for first data-analysis"

fig1 = plt.figure()
## Filtered pat
plt.subplot(3,1,1), plt.plot(time,pat_filt, label='Raw PAT'),plt.ylabel("PAT"), plt.xlabel("Time [s]")
## Systoolic bloodpressure
plt.subplot(3,1,2), plt.plot(time,bp_systolic, 'g', label='Systolic BP'),plt.ylabel("Blood Pressure"), plt.xlabel("Time[s]")
## Systoic mean
plt.subplot(3,1,3) , plt.plot(time,bp_mean, 'r', label='Systolic BP mean'),plt.ylabel("Blood Pressure"), plt.xlabel("Time[s]")

# ACF plot
from statsmodels.graphics.tsaplots import plot_acf

lag_array = [20,50,100] 

for i  in range(len(lag_array)):
    
    plot_acf(bp_systolic, lags=lag_array[i])
    
    
    plt.title("Autocorrelation Function (ACF) for SBP total")
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.show()
    
# Cutting of the data according to protocol

bp_syst_rest1 = bp_systolic[:718]
bp_syst_static = bp_systolic[719:1004]
bp_syst_rest2 = bp_systolic[1004:1147]
bp_syst_dynamic = bp_systolic[1148:1415]
bp_syst_rest3 = bp_systolic[1146:1471]
bp_syst_mental = bp_systolic[1601:1890]
bp_syst_rest4 = bp_systolic[1890:2343]

plot_acf(bp_syst_rest1, lags= 100)
plt.title('Autocorrelation Function (ACF) for BP rest')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()

plot_acf(bp_syst_static, lags= 100)
plt.title('Autocorrelation Function (ACF) for BP static excercises')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()

plot_acf(bp_syst_dynamic, lags= 100)
plt.title('Autocorrelation Function (ACF) for BP dynamic excerrcises')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()

plot_acf(bp_syst_mental, lags= 100)
plt.title('Autocorrelation Function (ACF) for BP mental excerrcises')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()

# Moving mean

"""
use the np.convolve function to calculate the rolling average by convolving the 
data array with the weights array. 
The mode='valid' parameter ensures that the output array 
has the same length as the input data.
"""

def moving_average_numpy(data, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(data, weights, mode='valid')

mov_mean_BP =  moving_average_numpy(bp_systolic, 600)
mov_mean_PAT = moving_average_numpy(pat_filt, 600)

plt.figure()
plt.plot(mov_mean_BP)
plt.title("Moving mean BP systolic")
plt.show()

plt.figure()
plt.plot(mov_mean_PAT)
plt.title("Moving mean PAT")
plt.show()


"""
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

"""

