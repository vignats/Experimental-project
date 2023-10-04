# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:48:59 2023

@author: salome
"""
import matplotlib.pyplot as plt

import pandas as pd 
import numpy as np 
## Read excel in data frame 
df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv') 

## Data analysis
time = df['wrist@(9mm,809nm)_delay_s']
bp_systolic = df['blood pressure_systolic']
pat_raw = df['wrist@(9mm,809nm)_raw_pat']
pat_filtred = df['wrist@(9mm,809nm)_filtered_pat_bottomTI']

# Plot 
plt.figure(1)
plt.plot(time, bp_systolic)
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
window = 60
plt.figure(4)
bp_systolic.rolling(window).mean().plot(style='k')
plt.title('Moving mean')

plt.figure(5)
bp_systolic.rolling(window).std().plot(style='k')
plt.title('Moving std')

## Auto-correlation
bp_s = bp_systolic.fillna(0)
auto_correlation = np.correlate(bp_s.array, bp_s.array, mode='same')

plt.figure(6)
plt.plot(auto_correlation)
plt.title('Autocorrelation')



