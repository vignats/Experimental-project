# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:23:29 2023

@author: tanne
"""

 # Necessary libraries
import pandas as pd
import numpy  as np
import csv 
from scipy import stats
# Reading in the dataset
df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv')

# Remove NaN values of PAT
df_squashed = df.dropna(subset=['wrist@(9mm,809nm)_filtered_pat_bottomTI'])
#Remove rows of NaN values of blood pressure
df_squashed = df_squashed.dropna(subset = ["blood pressure_systolic"])
df_squashed

# Identify outliers of column by computing z-score
z = np.abs(stats.zscore(df_squashed['wrist@(9mm,809nm)_filtered_pat_bottomTI']))

# Identify outliers as pat_filtered with a z-score greater than 3
threshold = 3
outliers = df_squashed['wrist@(9mm,809nm)_filtered_pat_bottomTI'][z > threshold]

# Print the outliers
print(outliers)

# Remove outliers
df_squashed2 = df.drop(outliers.index)
df_squashed2