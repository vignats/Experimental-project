# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:46:30 2023

@author: tanne
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Assuming 'data' is your DataFrame with columns 'PAT' and 'BloodPressure'
df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv')

## Getting the data right with the correct index

# Feature Engineering (if needed)
# E.g., handling missing values, visualizing the data
time = df['wrist@(9mm,809nm)_delay_s']
data_squashed = df.dropna(subset=['wrist@(9mm,809nm)_filtered_pat_bottomTI'])
data_squashed = data_squashed.dropna(subset=['blood pressure_systolic'])
# Train-Test Split
train_size = int(len(data_squashed) * 0.8)
train, test = data_squashed[0:train_size], data_squashed[train_size:]

# Linear ARX Model (assuming p=5, q=0 for simplicity)
linear_model = ARIMA(train['blood pressure_systolic'], order=(0, 1, 1))
linear_model_fit = linear_model.fit()

# Extract linear model predictions
linear_predictions = linear_model_fit.forecast(steps=len(test))

# Residuals (difference between actual and linear predictions)
residuals = test['blood pressure_systolic'] - linear_predictions

# Define a custom nonlinearity (replace with your actual nonlinearity function)
def custom_nonlinearity(x):
    return np.sin(x)  # Example nonlinearity: sine function

# Apply custom nonlinearity to residuals
nonlinear_predictions = linear_predictions + custom_nonlinearity(residuals)

# Model Evaluation
rmse = sqrt(mean_squared_error(test['blood pressure_systolic'], nonlinear_predictions))
print(f'Root Mean Squared Error (Nonlinear ARX): {rmse}')

# Prediction
future_steps = 10  # Adjust as needed
future_residuals = test['blood pressure_systolic'] - linear_predictions[-1]
future_nonlinear_predictions = linear_model_fit.forecast(steps=future_steps) + custom_nonlinearity(future_residuals)

# Plotting
import matplotlib.pyplot as plt

plt.plot(train['blood pressure_systolic'], label='Train')
plt.plot(test['blood pressure_systolic'], label='Test')
plt.plot(test.index, nonlinear_predictions, label='Nonlinear Predictions', color='red')
plt.plot(pd.date_range(test.index[-1], periods=future_steps + 1, freq='D')[1:], future_nonlinear_predictions,
         label='Future Nonlinear Predictions', color='green')
plt.legend()
plt.show()
