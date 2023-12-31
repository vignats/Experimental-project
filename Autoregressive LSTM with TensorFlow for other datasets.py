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
from scipy import stats
from scipy.interpolate import interp1d
import time

'''
##INFORMATIONS

This program is a standalone program for training and testing on an LSTM model and use this trained model on other datasets
It mainly requires the packages : Pandas, Tensorflow, numpy, sckikit-learn, math, matplotlib and scipy

INPUT:
The dataset files are supposed to be in the same folder as the program and be csv or xlsx files.
Update the file location of the dataset file before use

OUTPUT : 
Model metrics, RMSE, and comparison graphs between predicted and expected values

General description : 
The preprocessing is done exactly like in class Preprocess, refer to its documentation for further information

After the preprocessing, the data is cut into 2 sets and normalized and then cut into smaller sequences whose size can be defined by the user
These sequence act like moving windows on the training and testing set allowing for a model training that take into account the time dependency of the data

Afterwards, predictions are made from the testing set and the data from another patient. 
metrics and graphs are obtained from these predictions after some post processing

Possible issues :
    - Size difference of the output values of the model depending on the sequence length chosen
    - The future predictions feature is not accurate and output are often irrealistic.


'''
start = time.time()


def summarize_diagnostics(history):
    #Allows to plot the RMSE evolution curve for the model as a function 
	# plot loss

	# plot accuracy
    plt.subplot(111)
    plt.title('Root Mean Square Error')
    plt.plot(history.history['root_mean_squared_error'], color='blue', label='train')
    plt.plot(history.history['val_root_mean_squared_error'], color='orange', label='test') 
    plt.xlabel("epochs")
    plt.ylabel("RMSE (mmHg)")
    plt.tight_layout()
    
# Importing the data

df = pd.read_csv('03-Oct-2023_patAnalysis_2.csv')
df = df.dropna(subset = ["blood pressure_systolic"])
#Importing a new testing set 
df2 = pd.read_excel('05-Dec-2023_patAnalysis_5.xlsx') # Be careful of the extension of the file, here the data provided was stored in a xlsx file
df2 = df2.dropna(subset = ["blood pressure_systolic"]) # Be careful of the label corresponding to the systolic BP, it has to be exactly "blood pressure_systolic" for the scaler to work

### PREPROCESSING
# Feature Engineering
#This process is the same as the one in the class Preprocess but with this model we needed to keep the data as pandas dataframe
#Outliers removal 
# Make a copy of the dataframe to fill in  NaN of PAT with mean NaN
df_outlier = df.copy()
df_outlier = df_outlier.dropna(subset = ["blood pressure_systolic"])
df_outlier['wrist@(9mm,809nm)_filtered_pat_bottomTI'] = df_outlier['wrist@(9mm,809nm)_filtered_pat_bottomTI'].fillna(df_outlier['wrist@(9mm,809nm)_filtered_pat_bottomTI'].mean())
# Commpute the z-statistic
z = np.abs(stats.zscore(df_outlier['wrist@(9mm,809nm)_filtered_pat_bottomTI']))
# Identify outliers as the pat_filtered with a z-score greater than 3
threshold = 3
outliers = df_outlier['wrist@(9mm,809nm)_filtered_pat_bottomTI'][z > threshold]
df = df.drop(outliers.index)
#Interpolation 
time_continuous = df['wrist@(9mm,809nm)_delay_s']
time_not_continuous = df.loc[df['wrist@(9mm,809nm)_filtered_pat_bottomTI'].notna(), 'wrist@(9mm,809nm)_delay_s']
interp_func = interp1d(time_not_continuous.array, df.loc[df['wrist@(9mm,809nm)_filtered_pat_bottomTI'].notna(), 'wrist@(9mm,809nm)_filtered_pat_bottomTI'].array, kind='linear', fill_value="extrapolate")
continuous_values = interp_func(time_continuous)
df.insert(loc=3, column='pat_filtred_continuous', value=continuous_values)

#Doing the same process for the new data set
df2_outlier = df2.copy()
df2_outlier = df2_outlier.dropna(subset = ["blood pressure_systolic"])
df2_outlier['wrist@(9mm,812nm)_filtered_pat_bottomTI'] = df2_outlier['wrist@(9mm,812nm)_filtered_pat_bottomTI'].fillna(df2_outlier['wrist@(9mm,812nm)_filtered_pat_bottomTI'].mean())
# Commpute the z-statistic
z2 = np.abs(stats.zscore(df2_outlier['wrist@(9mm,812nm)_filtered_pat_bottomTI'])) #Also be careful to adapt the name of the filtered_pat column
# Identify outliers as the pat_filtered with a z-score greater than 3
outliers2 = df2_outlier['wrist@(9mm,812nm)_filtered_pat_bottomTI'][z2 > threshold]
df2 = df2.drop(outliers2.index)
#Interpolation
time_continuous = df2['wrist@(9mm,812nm)_delay_s']
time_not_continuous = df2.loc[df2['wrist@(9mm,812nm)_filtered_pat_bottomTI'].notna(), 'wrist@(9mm,812nm)_delay_s']
interp_func = interp1d(time_not_continuous.array, df2.loc[df2['wrist@(9mm,812nm)_filtered_pat_bottomTI'].notna(), 'wrist@(9mm,812nm)_filtered_pat_bottomTI'].array, kind='linear', fill_value="extrapolate")
continuous_values = interp_func(time_continuous)
df2.insert(loc=3, column='pat_filtred_continuous', value=continuous_values)


#TRAINING THE MODEL
# Train-Test Split
train_size = int(len(df['pat_filtred_continuous']) * 0.9) # Change the value for bigger training set, Questions should be asked about how to use multiple patient data for training.
train, test = df[0:train_size], df[train_size:]

Test2 = df2[['pat_filtred_continuous','blood pressure_systolic']]
# Normalize data using Standard scaling
scaler = StandardScaler()
train_normalized = scaler.fit_transform(train[['pat_filtred_continuous', 'blood pressure_systolic']])
test_normalized = scaler.transform(test[['pat_filtred_continuous', 'blood pressure_systolic']])
Test2_normalized = scaler.transform(Test2[['pat_filtred_continuous','blood pressure_systolic']])
# Create sequences for time series data
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length, :]
        target = data[i+seq_length, 1]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

seq_length = 1   # Adjust as needed
X_train, y_train = create_sequences(train_normalized, seq_length)
X_test, y_test = create_sequences(test_normalized, seq_length)

X_test2, y_test2 = create_sequences(Test2_normalized,seq_length)

# Build a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(25, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['RootMeanSquaredError'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test2, y_test2), verbose=1)

# Model Evaluation
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
Test2_preds = model.predict(X_test2)
train_rmse = sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = sqrt(mean_squared_error(y_test, test_predictions))
print("Training RMSE :", train_rmse, 'Testing RMSE : ', test_rmse)


# BP PREDICTION # Not really working (really random output)
future_steps = 5  # Adjust as needed
future_data = test_normalized[-seq_length:].reshape((1, 1, 2))  # Assuming 2 features (PAT and BloodPressure)
future_predictions = []

for _ in range(future_steps):
    prediction = model.predict(future_data)[0, 0]

    # Update future_data for the next iteration
    new_data_point = np.array([[future_data[0, -1, 0], prediction]])  # Assuming 'PAT' is 1st column, 'BloodPressure' is 2nd
    future_data = np.concatenate([future_data, new_data_point.reshape(1, 1, 2)], axis=1)

    future_predictions.append(prediction)

 ##POST-PROCESSING     
# # Concatenate 'PAT' values with future predictions for inverse transform
future_predictions_with_pat = np.column_stack((future_data[0, 0:future_steps, 0], np.array(future_predictions)))
test_predictions_stacked = np.column_stack((test_normalized[seq_length:, 0],test_predictions))
test_predictions2_stacked = np.column_stack((Test2_normalized[seq_length:, 0],Test2_preds))
# De-normalize predictions
future_predictions_denormalized = scaler.inverse_transform(future_predictions_with_pat)
test_predictions_denormalized = scaler.inverse_transform(test_predictions_stacked)
test_predictions2_denormalized = scaler.inverse_transform(test_predictions2_stacked)
print('Predicted values for the next', future_steps ,'th step :', future_predictions_denormalized[:,1])

# PLOTTING THE RESULTS
#Full Training BP + Testing BP + Predicted BP + Future Predictions vs Index of values (Time index)
plt.plot(range(0,len(train),1), train['blood pressure_systolic'], label='Train')
plt.plot(range(len(train)+1,len(test)+len(train)+1,1), test['blood pressure_systolic'], label='Test')
plt.plot(range(len(train)+1+seq_length,len(test)+len(train)+1,1), test_predictions_denormalized[:,1], label='Test Predictions', color='red')
plt.plot(range(len(test)+len(train)+1,len(test)+len(train)+(len(future_predictions_denormalized)+1),1), future_predictions_denormalized[:,1], label='Future Predictions', color='green')
plt.legend()
plt.ylabel('Blood Pressure')
plt.show()
#Comparison between predicted BP values with target values as a function of time
plt.figure()
plt.plot(df['wrist@(9mm,809nm)_delay_s'][train_size:], test['blood pressure_systolic'], label='testing set', color ='red' )
plt.plot(df_outlier['wrist@(9mm,809nm)_delay_s'][train_size+9:], test_predictions_denormalized[:,1], label='testing predictions', color ='green' )
plt.xlabel("Delay(s)")
plt.ylabel('Systolic Blood Pressure (mmHg)')
plt.legend()
plt.show()
#Comparison on new Data set
plt.figure()
plt.plot(df2['wrist@(9mm,812nm)_delay_s'], Test2['blood pressure_systolic'], label='testing set', color ='red' )
plt.plot(df2['wrist@(9mm,812nm)_delay_s'][1:], test_predictions2_denormalized[:,1], label='testing predictions', color ='green' )
plt.xlabel("Delay(s)")
plt.ylabel('Systolic Blood Pressure (mmHg)')
plt.legend()
plt.show()
#Plotting performance curves
summarize_diagnostics(history)

end = time.time()
print("Time elasped :", end - start, "seconds")