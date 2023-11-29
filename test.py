# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:30:48 2023

@author: salome
"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from models import Preprocess
import matplotlib.pyplot as plt
import pandas as pd


# Création de données factices
df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv')
X, y = Preprocess('03-Oct-2023_patAnalysis_2.csv').process_data(True, False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)  


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

"""
# Transformation polynomiale
poly = PolynomialFeatures(degree=3)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)
"""
# Modèle de régression linéaire
regression = LinearRegression()

regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared = False)
