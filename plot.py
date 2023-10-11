# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:48:59 2023

@author: salome
"""
import matplotlib.pyplot as plt
import pandas as pd 

## Read excel in data frame 
df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv')

## Data analysis
time = df['wrist@(9mm,809nm)_delay_s']
bp_systolic = df['blood pressure_systolic']

# Plot 
plt.figure(1)
plt.subplot(3,1,1)
plt.axvline(time[716], color = 'r')
plt.axvline(time[1002], color = 'r')
plt.axvline(time[1145], color = 'r')
plt.axvline(time[1469], color = 'r')
plt.axvline(time[1599], color = 'r')
plt.axvline(time[1890], color = 'r')
plt.xlabel('Time (in s)')
plt.ylabel('Blood Pressure')
plt.title('BP')
plt.plot(time, bp_systolic)

plt.subplot(3,1,2)
plt.plot(time, df['wrist@(9mm,809nm)_filtered_pat_bottomTI'])
plt.title('PAT filtred')

plt.subplot(3,1,3)
plt.plot(time, df['wrist@(9mm,809nm)_raw_pat'])
plt.title('PAT raw')
plt.show()

def plot_periode(function, window, name):
    function.plot(style='k')
    plt.axvline(716, color = 'r')
    plt.axvline(1002, color = 'r')
    plt.axvline(1145, color = 'r')
    plt.axvline(1469, color = 'r')
    plt.axvline(1599, color = 'r')
    plt.axvline(1890, color = 'r')
    plt.xlabel('First index of the moving window')
    plt.ylabel(f'{name} of the BP')
    plt.title(f'{name} of the BP with window = {window}')

## Mean and standard deviation
# Global mean and standard deviation
mean = df.mean()
std = df.std()

# Moving mean and standard deviation
window = 150
window_derivation = 150
moving_mean = bp_systolic.rolling(window).mean()
derivated = pd.Series([moving_mean[i+1] - moving_mean[i] for i in range(len(moving_mean) - window_derivation)])

plt.figure(2)
plt.subplot(3,1,1)
plot_periode(moving_mean, window, 'Moving mean')
plt.subplot(3,1,2)
plot_periode(bp_systolic.rolling(window).median(), window, 'Moving median')
plt.subplot(3,1,3)
plot_periode(bp_systolic.rolling(window).std(), window, 'Moving standard deviation')

plt.figure(3)
plot_periode(derivated, window, 'Derivation of the moving mean')




