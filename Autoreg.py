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


Y = BP
X_raw =PAT
data = np.zeros((1,len(X_raw),2))
data[:,:,0] = X_raw
data[:,:,1]= Y

mod = AutoReg(X_raw, 3, old_names=False)
res = mod.fit()
print(res.summary())