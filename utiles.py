# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:56:04 2023

@author: salome
"""
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

def interpol(df):
   time = df['wrist@(9mm,809nm)_delay_s']
   
   interp_func = interp1d(df['wrist@(9mm,809nm)_delay_s'].array, df['wrist@(9mm,809nm)_filtered_pat_bottomTI'].array, kind='linear', fill_value="extrapolate")
   continuous_values = interp_func(time)
   
   df = df.insert(loc = 3, column = 'pat_filtred_continuous', value = continuous_values)

def process_data(path, interpolation, invert, normalize):
    """"
    Process the data with multiple possibilities
    interpolation = True or False
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=['wrist@(9mm,809nm)_filtered_pat_bottomTI'])
    df = df.dropna(subset=['blood pressure_systolic'])

    if interpolation == True : 
        interpol(df)
    
    X, y = df.iloc[:, 3].to_numpy().reshape(-1, 1), df['blood pressure_systolic'].to_numpy().reshape(-1, 1)

    if invert == True:  
        X = 1/X
        
    if normalize == True : 
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
          
    return X, y

