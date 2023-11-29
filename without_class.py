# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:34:22 2023

@author: salome
"""
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

def preprocess(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['blood pressure_systolic'])

    return df

def interpol(df):
    time_continuous = df['wrist@(9mm,809nm)_delay_s']
    time_not_continuous = df.loc[df['wrist@(9mm,809nm)_filtered_pat_bottomTI'].notna(), 'wrist@(9mm,809nm)_delay_s']

    interp_func = interp1d(
        time_not_continuous.array,
        df.loc[df['wrist@(9mm,809nm)_filtered_pat_bottomTI'].notna(), 'wrist@(9mm,809nm)_filtered_pat_bottomTI'].array,
        kind='linear',
        fill_value="extrapolate"
    )
    continuous_values = interp_func(time_continuous)

    df.insert(loc=3, column='pat_filtred_continuous', value=continuous_values)

def process_data(df, interpolation, invert):
    if interpolation:
        interpol(df)
    else:
        df.dropna(subset=['wrist@(9mm,809nm)_filtered_pat_bottomTI'])

    X = df.iloc[:, 3].to_numpy().reshape(-1, 1)
    y = df['blood pressure_systolic'].to_numpy().reshape(-1, 1)

    if invert:
        X = 1 / X

    return X, y

def normalization(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

def data_splitting(X, y, mod, test_size=0.3, random_state=42, normalize=True):
    if isinstance(mod, PolynomialFeatures):
        X = mod.fit_transform(X)
        mod = LinearRegression()  # After adding polynomial feature, the prediction is done using a linear regressor.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if normalize:
        X_train, X_test = normalization(X_train, X_test)

    return X_train, X_test, y_train, y_test

def n_split(model, X, y):
    rmse_n = []
    for n in range(2, 20):
        rmse_n.append(accuracy(model, X, y, split_type='tscv', n_splits=n)[0])
    return rmse_n.index(min(rmse_n))

def accuracy(model, X, y, split_type='classical', n_splits=4):
    if split_type == 'classical':
        X_train, X_test, y_train, y_test = data_splitting(X, y, model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred, squared=False), r2_score(y_test, y_pred)

    if split_type == 'tscv':
        tscv = TimeSeriesSplit(n_splits)
        rmse = []
        for train_index, test_index in tscv.split(X, y):
            if isinstance(model, PolynomialFeatures):
                X = model.fit_transform(X)
                model = LinearRegression()  # After adding polynomial feature, the prediction is done using a linear regressor.

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train, X_test = normalization(X_train, X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse.append(mean_squared_error(y_test, y_pred, squared=False))
        return min(rmse), r2_score(y_test, y_pred)