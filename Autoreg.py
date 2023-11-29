# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:23:56 2023

@author: tanne
"""

#Autoregression 

import matplotlib.pyplot as plt
import pandas as pd
#import pandas_datareader as pdr
import numpy as np
import seaborn as sns
from statsmodels.tsa.api import acf, graphics, pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from sklearn.model_selection import train_test_split
sns.set_style("darkgrid")
pd.plotting.register_matplotlib_converters()
# Default figure size
sns.mpl.rc("figure", figsize=(16, 6))
sns.mpl.rc("font", size=14)

## Read excel in data frame 
df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv')

## Getting the data right with the correct index
time = df['wrist@(9mm,809nm)_delay_s']
data_squashed = df.dropna(subset=['wrist@(9mm,809nm)_filtered_pat_bottomTI'])
data_squashed = data_squashed.dropna(subset=['blood pressure_systolic'])
PAT= data_squashed['wrist@(9mm,809nm)_filtered_pat_bottomTI'].to_numpy()
BP = data_squashed['blood pressure_systolic'].to_numpy()

#Data splitting
Y = BP
X_raw =PAT
X_train, X_test, Y_train, Y_test = train_test_split(X_raw, Y, test_size=0.3)


#Training
modct = AutoReg(X_train, 13, trend='c', exog=Y_train)
res = modct.fit()


#Accuracy 
print(res.summary())
print(res.params)
print("Accuracy = ", modct.score(res.params))
fig = plt.figure(figsize=(16, 9))
fig = res.plot_diagnostics(fig=fig, lags=30)

