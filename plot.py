# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:48:59 2023

@author: salome
"""
import matplotlib.pyplot as plt

import pandas as pd 
import numpy as np 
# read excel in data frame 
df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv') 
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
