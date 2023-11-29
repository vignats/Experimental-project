# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:53:36 2023

@author: tanne
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from models import Preprocess

def summarize_diagnostics(history):
	# plot loss
    plt.subplot(211)
    plt.title('Mean squared Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
    plt.subplot(212)
    plt.title('Root Mean Square Error')
    plt.plot(history.history['root_mean_squared_error'], color='blue', label='train')
    plt.plot(history.history['val_root_mean_squared_error'], color='orange', label='test') 
    plt.tight_layout()
    
# Importing the data

df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv')
df['wrist@(9mm,809nm)_delay_s'] = pd.to_datetime(df['wrist@(9mm,809nm)_delay_s'])
df.set_index('wrist@(9mm,809nm)_delay_s', inplace=True)
## Getting the data right with the correct index

# Feature Engineering
df_squashed = df.dropna(subset=['wrist@(9mm,809nm)_filtered_pat_bottomTI'])
df_squashed = df_squashed.dropna(subset = ["blood pressure_systolic"])
#Outliers removal 
##TO BE COMPLETED##

#Interpolation 
p = Preprocess('03-Oct-2023_patAnalysis_2.csv')
p.interpol()
df_squashed_interpolled= p.df

#Dropping Nan Values
#data_squashed = df.dropna(subset=['blood pressure_systolic'])

# Train-Test Split
train_size = int(len(df_squashed_interpolled) * 0.7)
train, test = df_squashed_interpolled[0:train_size], df_squashed_interpolled[train_size:]


# Normalize data using Standard scaling
scaler = StandardScaler()
train_normalized = scaler.fit_transform(train[['pat_filtred_continuous', 'blood pressure_systolic']])
test_normalized = scaler.transform(test[['pat_filtred_continuous', 'blood pressure_systolic']])

# Create sequences for time series data
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length, :]
        target = data[i+seq_length, 1]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

seq_length = 60  # Adjust as needed
X_train, y_train = create_sequences(train_normalized, seq_length)
X_test, y_test = create_sequences(test_normalized, seq_length)

# Build a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50,activation="relu", input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['RootMeanSquaredError'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Model Evaluation
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_rmse = sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = sqrt(mean_squared_error(y_test, test_predictions))
print("Training RMSE :", train_rmse, 'Testing RMSE : ', test_rmse)

# Prediction
future_steps = 5  # Adjust as needed
future_data = test_normalized[-seq_length:].reshape((1, seq_length, 2))  # Assuming 2 features (PAT and BloodPressure)
future_predictions = []

for _ in range(future_steps):
    prediction = model.predict(future_data)[0, 0]

    # Update future_data for the next iteration
    new_data_point = np.array([[future_data[0, -1, 0], prediction]])  # Assuming 'PAT' is 1st column, 'BloodPressure' is 2nd
    future_data = np.concatenate([future_data, new_data_point.reshape(1, 1, 2)], axis=1)

    future_predictions.append(prediction)

      
# Concatenate 'PAT' values with future predictions for inverse transform
future_predictions_with_pat = np.column_stack((future_data[0, 0:future_steps, 0], np.array(future_predictions)))
test_predictions_stacked = np.column_stack((test_normalized[seq_length:, 0],test_predictions))

# De-normalize predictions
future_predictions_denormalized = scaler.inverse_transform(future_predictions_with_pat)
test_predictions_denormalized = scaler.inverse_transform(test_predictions_stacked)
print('Predicted values for the next', future_steps ,'th step :', future_predictions_denormalized[:,1])

# Plotting data
plt.plot(train.index, train['blood pressure_systolic'], label='Train')
plt.plot(test.index, test['blood pressure_systolic'], label='Test')
plt.plot(test.index[seq_length:], test_predictions_denormalized[:,1], label='Test Predictions', color='red')
plt.plot(pd.date_range(test.index[-1], periods=future_steps+1, freq='N')[1:], future_predictions_denormalized,
         label='Future Predictions', color='green')
plt.legend()
plt.show()
#Plotting performance curves
summarize_diagnostics(history)