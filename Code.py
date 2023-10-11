# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:53:32 2023

@author: lilac
"""

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
## Read excel in data frame 
df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv').iloc[24:] 

## Data analysis
time = df['wrist@(9mm,809nm)_delay_s']
bp_systolic = df['blood pressure_systolic']
bp_systolic_mean = df['blood pressure_mean']
pat_raw = df['wrist@(9mm,809nm)_raw_pat']
pat_filtred = df['wrist@(9mm,809nm)_filtered_pat_bottomTI']

##Periods separation
def trace() :
    plt.axvline(x=716, color='r')
    plt.axvline(x=1002, color='r')
    plt.axvline(x=1145, color='r')
    plt.axvline(x=1469, color='r')
    plt.axvline(x=1599, color='r')
    plt.axvline(x=1890, color='r')

# Plot 
plt.figure(1)
plt.plot(bp_systolic)
trace()
'''
plt.plot(time, bp_systolic_mean)
plt.title('BP')
'''
pat_new = pat_filtred.fillna(method='ffill')
plt.figure(2)
plt.plot(time, pat_new)
#plt.xlim(0,2000)
plt.ylim(0, 2.2)
plt.title('PAT filtred')

plt.figure(3)
plt.plot(pat_filtred, bp_systolic)
plt.title('PAT raw')
plt.show()
'''
## Mean and standard deviation
# Global mean and standard deviation
mean = df.mean()
std = df.std()
 
# Moving mean and standard deviation
window = 150
window_derivative = 150

plt.figure(4)
rolling_mean = bp_systolic.rolling(window).mean()
plt.plot(rolling_mean)
plt.grid(True)
trace()
plt.title('Moving mean')

#Derivative
a = rolling_mean.size-1
df['Derivative']=pd.Series() 
for i in range (0,a-window_derivative):
    df['Derivative'].iloc[i] = (rolling_mean.iloc[i+window_derivative]-rolling_mean.iloc[i])

Derivee = df['Derivative']

plt.figure(5)
plt.plot(Derivee)
trace()
'''


