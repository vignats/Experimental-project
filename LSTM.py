# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:48:09 2023

@author: salome
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from utiles import process_data

# Data extraction
X, y = process_data('03-Oct-2023_patAnalysis_2.csv')


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the data (you can also use StandardScaler if needed)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set the dimensions of the input data
input_dim = X_train.shape[1]
timesteps = 1  # In this example, we consider a single time step

# Create the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(timesteps, input_dim), return_sequences=False))
model.add(Dense(1))  # Output layer with a single output for BP

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train.reshape(X_train.shape[0], timesteps, input_dim), y_train, epochs=50, batch_size=32)

# Make predictions on the test set
y_pred = model.predict(X_test.reshape(X_test.shape[0], timesteps, input_dim))

# Calculate the Mean Squared Error (MSE) on the predictions
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
