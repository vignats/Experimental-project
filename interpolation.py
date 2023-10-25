# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:22:25 2023

@author: salome
"""
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.interpolate import interp1d
import scipy as sp
import scipy.fftpack
import numpy as np

## Read excel in data frame 
df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv')
bp_systolic = df['blood pressure_systolic']
pat_filtred = df['wrist@(9mm,809nm)_filtered_pat_bottomTI']

## Interpolation
time = df['wrist@(9mm,809nm)_delay_s']
data_squashed = df.dropna(subset=['wrist@(9mm,809nm)_filtered_pat_bottomTI'])

interp_func = interp1d(data_squashed['wrist@(9mm,809nm)_delay_s'].array, data_squashed['wrist@(9mm,809nm)_filtered_pat_bottomTI'].array, kind='linear', fill_value="extrapolate")

continuous_values = interp_func(time)

plt.figure(1)
plt.plot(bp_systolic, continuous_values)

plt.figure(2)
plt.plot(continuous_values, bp_systolic)
TF_blood_pressure = sp.fftpack.fft(np.array(bp_systolic))
TF_pat = sp.fftpack.fft(np.array(continuous_values))

plt.figure(3)
plt.plot(TF_pat , TF_blood_pressure)