# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:22:25 2023

@author: salome
"""
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.interpolate import interp1d

## Read excel in data frame 
df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv')

## Interpolation
time = df['wrist@(9mm,809nm)_delay_s']
data_squashed = df.dropna(subset=['wrist@(9mm,809nm)_filtered_pat_bottomTI'])
interp_func = interp1d(data_squashed['wrist@(9mm,809nm)_delay_s'].array, data_squashed['wrist@(9mm,809nm)_filtered_pat_bottomTI'].array, kind='linear', fill_value="extrapolate")
continuous_values = interp_func(time)


