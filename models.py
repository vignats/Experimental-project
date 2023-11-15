# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:21:13 2023

@author: salome
"""

import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error

class Preprocess():
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.df = self.df.dropna(subset=['wrist@(9mm,809nm)_filtered_pat_bottomTI'])
        self.df = self.df.dropna(subset=['blood pressure_systolic'])
                
    def interpol(self):
       time = self.df['wrist@(9mm,809nm)_delay_s']
       interp_func = interp1d(self.df['wrist@(9mm,809nm)_delay_s'].array, self.df['wrist@(9mm,809nm)_filtered_pat_bottomTI'].array, kind='linear', fill_value="extrapolate")
       continuous_values = interp_func(time)

       self.df.insert(loc=3, column='pat_filtred_continuous', value=continuous_values)       
              
    def process_data(self, interpolation, invert, normalize):
        """"
        Process the data with multiple possibilities
        interpolation = True or False
        """
        if interpolation == True : 
            self.interpol()
        
        X, y = self.df.iloc[:, 3].to_numpy().reshape(-1, 1), self.df['blood pressure_systolic'].to_numpy().reshape(-1, 1)

        if invert == True:  
            X = 1/X
            
        if normalize == True : 
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
              
        return X, y

class Model():
    def __init__(self, path, interpolation = True, invert = True, normalize = True):
        """

        Parameters
        ----------
        path : access to the data, need to be in the same file.

        """
        self.interpolation = interpolation 
        self.invert = invert
        self.normalize = normalize

        self.X, self.y = Preprocess(path).process_data(interpolation, invert, normalize)
        
        #self.model = model
     
    def data_splitting(self, test_size = 0.3, random_state = 42):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = test_size, random_state = random_state)
        return X_train, X_test, y_train, y_test
        
    def linear_regression(self, split_type = 'classical', n_splits=6):
        if split_type == 'classical':
            X_train, X_test, y_train, y_test = self.data_splitting()
            
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            
            y_pred = reg.predict(X_test)
            return mean_squared_error(y_test, y_pred, squared = False)
        
        if split_type == 'tscv':
            tscv = TimeSeriesSplit(n_splits)
            rmse = []
            for train_index, test_index in tscv.split(self.X, self.y):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                
                reg = LinearRegression()
                reg.fit(X_train, y_train)
                
                y_pred = reg.predict(X_test)
                rmse.append(mean_squared_error(y_test, y_pred, squared = False))
                return max(rmse)

model = Model('03-Oct-2023_patAnalysis_2.csv')     
rmse = model.linear_regression(split_type = 'tscv')  
        
        
                
        
        
        