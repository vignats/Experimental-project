# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
Created on Wed Oct  4 10:58:04 2023

@author: maria
"""

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from statsmodels.tsa.stattools import adfuller, kpss
import csv

#specify columns that should be read as floats
float_columns = ['wrist@(9mm,809nm)_delay_s', 'blood pressure_systolic','blood pressure_mean', 'wrist@(9mm,809nm)_raw_pat', 'wrist@(9mm,809nm)_filtered_pat_bottomTI']
# Specify the NaN values and columns that should be read as floats
na_values = ['NA', 'N/A', 'NaN', '']
# read in csv file
df = pd.read_csv("/Users/mariafoyen/Documents/Experimental-project/03-Oct-2023_patAnalysis_2.csv")
# Replace NaN values with zeros in the entire DataFrame (if needed)
#df.fillna(0, inplace=True)

"""Data for analysis"""

time = df['wrist@(9mm,809nm)_delay_s']
bp_systolic = df['blood pressure_systolic']
bp_systolic_mean = df['blood pressure_mean']
pat_raw = df['wrist@(9mm,809nm)_raw_pat']
pat_filtered = df['wrist@(9mm,809nm)_filtered_pat_bottomTI']

print(time.head())



mean = df.mean()
std = df.std()

window_size = 600

plt.figure(4)
bp_systolic.rolling(window_size).mean().plot(style='k')
plt.title('Moving mean')

plt.figure(5)
bp_systolic.rolling(window_size).std().plot(style='k')
plt.title('Moving std')

plt.figure(5)
pat_filtered.rolling(window_size).mean().plot(style='k')

# Plot  ACF

from statsmodels.graphics.tsaplots import plot_acf

# Check for NaN or missing values in the data
if bp_systolic.isnull().any():
    print("Warning: Data contains NaN or missing values.")
else:
    # Calculate ACF
    max_lag = 50  # You can adjust the number of lags as needed
    plot_acf(bp_systolic, lags=max_lag)

    # Customize the plot (if necessary)
    plt.title('Autocorrelation Function (ACF) for Systolic Blood Pressure')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.show()