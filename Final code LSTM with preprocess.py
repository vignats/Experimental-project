# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:18:02 2023

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

import time
'''
##INFORMATIONS

This program is a standalone program for training and testing on an LSTM model and use this trained model on other datasets
It mainly requires the packages : Pandas, Tensorflow, numpy, sckikit-learn, math and  matplotlib

INPUT:
The dataset files are supposed to be in the same folder as the program and be csv or xlsx files.
Update the file location of the dataset file before use

OUTPUT : 
Model metrics, RMSE, and comparison graphs between predicted and expected values

General description : 
The preprocessing is done in class Preprocess, refer to its documentation for further information

After the preprocessing, the data is cut into 2 sets and normalized and then cut into smaller sequences whose size can be defined by the user
These sequence act like moving windows on the training and testing set allowing for a model training that take into account the time dependency of the data

Afterwards, predictions are made from the testing set and the data from another patient. 
metrics and graphs are obtained from these predictions after some post processing

Possible issues :
    - Size difference of the output values of the model depending on the sequence length chosen
'''
start = time.time()

def summarize_diagnostics(history):
    #Allows to plot the RMSE evolution curve for the model as a function 
    #Change the variable "Validation_data" in model.compile for comparing other sets of data
    plt.subplot(111)
    plt.title('Root Mean Square Error')
    plt.plot(history.history['root_mean_squared_error'], color='blue', label='train')
    plt.plot(history.history['val_root_mean_squared_error'], color='orange', label='test') 
    plt.legend()
    plt.tight_layout()
    
# Importing the data
df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv')
df = df.dropna(subset = ["blood pressure_systolic"])
Pre = Preprocess('03-Oct-2023_patAnalysis_2.csv')
#PREPROCESSING
#Here there is outlier picking, interpolation and inversion of the Pat value
X, y = Pre.process_data(interpolation=True, invert=True, outlier=True)
#TRAINING THE MODEL
# Train-Test Split
train_size = int(len(X) * 0.7)
X_train, X_test = X[0:train_size], X[train_size:]
y_train, y_test = y[0:train_size], y[train_size:]
train = np.column_stack((X_train, y_train))
test=np.column_stack((X_test, y_test))
# Normalize data using Standard scaling
scaler = StandardScaler()
train_normalized = scaler.fit_transform(train)
test_normalized = scaler.transform(test)

# Create sequences for time series data (Similar to time series split), creates a moving window of the length set up by the user
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length, :]
        target = data[i+seq_length, 1]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

seq_length = 5 # Adjust as needed
X_train2, y_train2 = create_sequences(train_normalized, seq_length)
X_test2, y_test2 = create_sequences(test_normalized, seq_length)

# Build a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(25, activation="sigmoid", input_shape=(X_train2.shape[1], X_train2.shape[2])),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['RootMeanSquaredError'])

# Train the model
history = model.fit(X_train2, y_train2, epochs=10, batch_size=32, validation_data=(X_test2, y_test2), verbose=1)

# Model Evaluation
train_predictions = model.predict(X_train2)
test_predictions = model.predict(X_test2)
test_predictions_stacked = np.column_stack((test_normalized[seq_length:, 0],test_predictions))

# #POST-PROCESSING 
#De-normalize predictions
test_predictions_denormalized = scaler.inverse_transform(test_predictions_stacked)
train_rmse = sqrt(mean_squared_error(y_train2, train_predictions))
test_rmse = sqrt(mean_squared_error(y_test2, test_predictions))
print("Training RMSE :", train_rmse, 'Testing RMSE : ', test_rmse)




# Plotting data
plt.plot(range(len(y_train)), y_train, label='Train')
plt.plot(range(len(y_train)+1,len(y_test)+len(y_train)+1,1), y_test, label='Test')
plt.plot(range(len(y_train)+5,len(y_test)+len(y_train),1), test_predictions_denormalized[:,1], label='Test Predictions', color='red')
#plt.plot(range(len(y_test)+len(y_train)+1,len(y_test)+len(y_train)+(len(future_predictions_denormalized)+1),1), future_predictions_denormalized[:,1], label='Future Predictions', color='green')
plt.legend()
plt.ylabel('Blood Pressure')
plt.show()

plt.figure()
plt.plot(df['wrist@(9mm,809nm)_delay_s'][train_size+8:]/60, y_test, label='Test', color ='red' )
plt.plot(df['wrist@(9mm,809nm)_delay_s'][train_size+13:]/60, test_predictions_denormalized[:,1], label='Test Predictions', color ='green' )
plt.xlabel("Delay since the first measurement (min)")
plt.ylabel('Systolic Blood Pressure (mmHg)')
plt.legend()
plt.show()
#Plotting performance curves
summarize_diagnostics(history)
end = time.time()
print(end - start)