# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:53:32 2023

@author: lilac
"""

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
## Read excel in data frame 
df = pd.read_csv('Experimental-project/03-Oct-2023_patAnalysis_2.csv') 

## Data analysis
time = df['wrist@(9mm,809nm)_delay_s']
bp_systolic = df['blood pressure_systolic']
bp_systolic_mean = df['blood pressure_mean']
pat_raw = df['wrist@(9mm,809nm)_raw_pat']
pat_filtred = df['wrist@(9mm,809nm)_filtered_pat_bottomTI']

# Plot 
plt.figure(1)
plt.plot(time, bp_systolic)
plt.plot(time, bp_systolic_mean)
plt.title('BP')
plt.figure(2)
plt.plot(time, pat_filtred)
plt.title('PAT filtred')
plt.figure(3)
plt.plot(time, pat_raw)
plt.title('PAT raw')
plt.show()

## Mean and standard deviation
# Global mean and standard deviation
mean = df.mean()
std = df.std()

# Moving mean and standard deviation
window = 200
'''
plt.figure(4)
bp_systolic.rolling(window).mean().plot(style='k')
plt.title('Moving mean')

plt.figure(5)
bp_systolic.rolling(window).std().plot(style='k')
plt.title('Moving std')

plt.figure(5)
pat_filtred.rolling(window).mean().plot(style='k')
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

'''
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
'''
bp_s = bp_systolic_at_rest.fillna(0)
auto_correl = np.correlate(bp_s.array,bp_s.array, mode = 'same')
plt.figure(8)
plt.plot(auto_correl)