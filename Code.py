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

# Plot 
plt.figure(1)
plt.plot(bp_systolic)
'''
plt.plot(time, bp_systolic_mean)
plt.title('BP')
plt.figure(2)
plt.plot(time, pat_filtred)
plt.title('PAT filtred')
plt.figure(3)
plt.plot(time, pat_raw)
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
plt.axvline(x=716, color='r')
plt.axvline(x=1002, color='r')
plt.axvline(x=1145, color='r')
plt.axvline(x=1469, color='r')
plt.axvline(x=1599, color='r')
plt.axvline(x=1890, color='r')
plt.figure(4)
rolling_mean = bp_systolic.rolling(window).mean()
#rolling_mean_norm= rolling_mean[623:2325]
plt.plot(rolling_mean)
#plt.xlim([0, (2325-623)])
plt.grid(True)
plt.axvline(x=716, color='r')
plt.axvline(x=1002, color='r')
plt.axvline(x=1002, color='r')
plt.axvline(x=1145, color='r')
plt.axvline(x=1469, color='r')
plt.axvline(x=1599, color='r')
plt.axvline(x=1890, color='r')
#plt.xticks([0,3.142,6.283,9.425], [0,'π','2π','3π'], color='r')
plt.title('Moving mean')

a = rolling_mean.size-1
df['Derivative']=pd.Series() 
for i in range (0,a-window_derivative):
    df['Derivative'].iloc[i] = (rolling_mean.iloc[i+window_derivative]-rolling_mean.iloc[i])

Derivee = df['Derivative']

plt.figure(5)
plt.plot(Derivee)
plt.axvline(x=1002, color='r')
plt.axvline(x=1145, color='r')
plt.axvline(x=1469, color='r')
plt.axvline(x=1599, color='r')
plt.axvline(x=716, color='r')
plt.axvline(x=1890, color='r')

plt.figure(6)
plt.plot(pat_filtred)
plt.axvline(x=716, color='r')
plt.axvline(x=1002, color='r')
plt.axvline(x=1002, color='r')
plt.axvline(x=1145, color='r')
plt.axvline(x=1469, color='r')
plt.axvline(x=1599, color='r')
plt.axvline(x=1890, color='r')

'''
#Series separation 

time_at_rest = time[50:323]

time_shift = []

for i in range (0,323-50) :
    time_shift.append(time_at_rest.iloc[i] + 10)

    
bp_systolic_at_rest = bp_systolic_mean[50:323]
pat_at_rest = pat_filtred[50:323]

pat_at_rest_shift = []
for i in range(0,323-60) :
    pat_at_rest_shift.append(pat_at_rest.iloc[i+10])


time_med_effort = time[324:717]
time_big_effort = time[718]

window = 150


plt.figure(4)
plt.plot(time_at_rest,bp_systolic_at_rest)

plt.figure(5)
plt.plot(time_at_rest[10:273],pat_at_rest_shift)

plt.figure(6)
plt.plot(pat_at_rest,bp_systolic_at_rest)

plt.figure(7)
plt.plot(pat_at_rest_shift,bp_systolic_at_rest[10:273])

plt.figure(5)
pat_at_rest.rolling(window).mean().plot(style='k')
plt.title('Moving mean')

bp_s = bp_systolic_at_rest.fillna(0)
auto_correl = np.correlate(bp_s.array,bp_s.array, mode = 'same')
plt.figure(8)
plt.plot(auto_correl)
'''