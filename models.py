# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:21:13 2023

@author: salome
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score


class Preprocess():
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.df = self.df.dropna(subset=['blood pressure_systolic'])
                
    def interpol(self):
        time_continuous = self.df['wrist@(9mm,809nm)_delay_s']
        time_not_continuous = self.df.loc[self.df['wrist@(9mm,809nm)_filtered_pat_bottomTI'].notna(), 'wrist@(9mm,809nm)_delay_s']
       
        interp_func = interp1d(time_not_continuous.array, self.df.loc[self.df['wrist@(9mm,809nm)_filtered_pat_bottomTI'].notna(), 'wrist@(9mm,809nm)_filtered_pat_bottomTI'].array, kind='linear', fill_value="extrapolate")
        continuous_values = interp_func(time_continuous)
       
        self.df.insert(loc=3, column='pat_filtred_continuous', value=continuous_values)       
    
    def outliers(self):
        df_outlier = self.df.copy()
        df_outlier['wrist@(9mm,809nm)_filtered_pat_bottomTI'] = df_outlier['wrist@(9mm,809nm)_filtered_pat_bottomTI'].fillna(df_outlier['wrist@(9mm,809nm)_filtered_pat_bottomTI'].mean())
        
        # Commpute the z-statistic
        z = np.abs(stats.zscore(df_outlier['wrist@(9mm,809nm)_filtered_pat_bottomTI']))
        # Identify outliers as the pat_filtered with a z-score greater than 3
        threshold = 3
        outliers = df_outlier['wrist@(9mm,809nm)_filtered_pat_bottomTI'][z > threshold]    
        # Remove outliers
        self.df = self.df.drop(outliers.index)
          
    def process_data(self, interpolation, invert, outlier):
        """"
        Process the data with multiple possibilities
        interpolation : interpolate the missing PAT data using scipy function
        invert : the correlation between PAT and BP is negative so we invert the PAT
        outlier : remove the outliered values of the PAT data 
        """
        
        if outlier == True :
            self.outliers()
            
        if interpolation == True : 
            self.interpol()
        else :
            self.df.dropna(subset=['wrist@(9mm,809nm)_filtered_pat_bottomTI'])
        
        X, y = self.df.iloc[:, 3].to_numpy().reshape(-1, 1), self.df['blood pressure_systolic'].to_numpy().reshape(-1, 1)
        
        if invert == True:  
            X = 1/X
    
        return X, y
    

class Model():
    def __init__(self, path, mod, interpolation = True, invert = True, outlier = True, normalize = True):
        """

        Parameters
        ----------
        path : access to the data, need to be in the same file.
        mod : Used model od deep learning.
        interpolation : interpolate the missing PAT data using scipy function
            default is True
        invert : the correlation between PAT and BP is negative so we invert the PAT
            default is True
        normalize : normalization of the PAT data 
            default is True
            
        """
        self.mod = mod
        self.normalize = normalize

        self.X, self.y = Preprocess(path).process_data(interpolation, invert, outlier)
    
    def normalization(self, X_train, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
              
        return X_train, X_test
        
    def data_splitting(self, test_size = 0.3, random_state =42):
        if isinstance(self.mod, PolynomialFeatures):
            self.X = self.mod.fit_transform(self.X)
            self.mod = LinearRegression() #After adding polynomial feature, the prediction is done using a linear regressor.
            
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, 
                            test_size = test_size, random_state = random_state)
        
        if self.normalize == True :
            X_train, X_test = self.normalization(X_train, X_test)
        
        return X_train, X_test, y_train, y_test
        
    def n_split(self):
        rmse_n = []
        for n in range (2,10):
            rmse_n.append(self.accuracy(split_type = 'tscv', n_splits=n)[0])
        plt.plot(range(2,10), rmse_n)
        
        return rmse_n.index(min(rmse_n))       

    def accuracy(self, split_type = 'classical', n_splits=6):
        """
        Parameters
        ----------
        split_type : Type of cross validation used, optional
            -- The default is 'classical', corresponding to one training set 
            containg 70% of the data and one test set containing 30%. 
            -- 'tscv' : time series cross-validation.
            
        n_splits : int, optional
            Number of folds, training set, when using tscv. The default is 6.

        Returns
        -------
        rmse
            Root mean square error between the predicted value and the real value. 
            In the case of tscv it is the max rmse over all the folds.
        r2
            R^2 

        """
        if split_type == 'classical':
            X_train, X_test, y_train, y_test = self.data_splitting()
            
            self.mod.fit(X_train, y_train)
            
            y_pred = self.mod.predict(X_test)
            self.y_pred = y_pred
            self.y_test = y_test 
            return mean_squared_error(y_test, y_pred, squared = False), r2_score(y_test, y_pred)
        
        if split_type == 'tscv':
            tscv = TimeSeriesSplit(n_splits)
            rmse = 1000 
            r2 = 0
            for train_index, test_index in tscv.split(self.X, self.y):
                if isinstance(self.mod, PolynomialFeatures):
                    self.X = self.mod.fit_transform(self.X)
                    self.mod = LinearRegression() #After adding polynomial feature, the prediction is done using a linear regressor.
                    
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                
                if self.normalize == True :
                    X_train, X_test = self.normalization(X_train, X_test)
                
                self.mod.fit(X_train, y_train)               
                
                y_pred = self.mod.predict(X_test)
                
                if mean_squared_error(y_test, y_pred, squared = False) < rmse:
                    rmse = mean_squared_error(y_test, y_pred, squared = False)
                    r2 = r2_score(y_test, y_pred)
                    self.y_pred = y_pred
                    self.y_test = y_test
                    
            return rmse, r2
