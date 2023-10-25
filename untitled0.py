# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:01:05 2023

@author: lilac
"""

import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split

## Read excel in data frame 
df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv') 

## Data analysis
time = df['wrist@(9mm,809nm)_delay_s']
bp_systolic = df['blood pressure_systolic']
pat_filtred = df['wrist@(9mm,809nm)_filtered_pat_bottomTI']
correlation = bp_systolic.corr(pat_filtred)

X =pat_filtred
Y = bp_systolic
X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3)
