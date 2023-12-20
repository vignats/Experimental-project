# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:48:11 2023

@author: salome
"""
import matplotlib.pyplot as plt
from models import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR 
from sklearn.neural_network import MLPRegressor
import tensorflow as tf




path = '03-Oct-2023_patAnalysis_2.csv'
normalize = invert = [False, True]
split_type = ['classical', 'tscv']
acc = pd.DataFrame(columns = ['Cross validation', 'Normalize', 'Invert', 'RMSE', 'R2'])

#model = MLPRegressor(activation = "tanh", solver="lbfgs", max_iter=10000)
#model = LinearRegression()
#model = SVR(kernel = 'rbf')
#model = train_regr = MLPRegressor(activation = "tanh", solver="lbfgs", max_iter=100000)
model = model = tf.keras.Sequential([
    tf.keras.layers.LSTM(25, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['RootMeanSquaredError'])
select_model = Model(path, model, invert = False)
rmse, r2= select_model.accuracy(split_type = 'tscv')

fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1.scatter(range(len(select_model.y_pred)), select_model.y_pred, s=10, c='b', marker="s", label='prediction')
ax1.scatter(range(len(select_model.y_pred)), select_model.y_test, s=10, c='r', marker="o", label='real values')
plt.legend(loc='upper left')
plt.show()

