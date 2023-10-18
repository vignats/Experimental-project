# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:07:49 2023

@author: tanne
"""

import pandas as pd 
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
import csv



def interpol(df, name_col):
   time = df['wrist@(9mm,809nm)_delay_s']
   data_squashed = df.dropna(subset=[name_col])
   interp_func = interp1d(data_squashed['wrist@(9mm,809nm)_delay_s'].array, data_squashed[name_col].array, kind='linear', fill_value="extrapolate")
   continuous_values = interp_func(time) 
   return continuous_values 
    
    
def ADF(mat):
    result = adfuller(mat, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'n_lags: {result[2]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}\n')
    if result[1]<= 0.05:
        print("Hypothesis can be rejected --> stationnary")
    else :
        print("Hypothesis cannot be rejected --> not stationnary")
        
def KPSS(mat):
    results = kpss(mat)
    print(f'KPSS Statistic: {results[0]}')
    print(f'n_lags: {results[2]}')
    print(f'p-value: {results[1]}')
    for key, value in results[3].items():
        print('Critial Values:')
        print(f'   {key}, {value}\n')
    if results[1]>= 0.05:
        print("Hypothesis cannot be rejected --> stationnary")
    else :
        print("Hypothesis can be rejected --> not stationnary")
        
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
 ADF(Diff)
 KPSS(Diff)
 
