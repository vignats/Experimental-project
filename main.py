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

"""
## INFORMATIONS
=== MODEL === 
Before computing the accuracy you need to initialize the Model class
---- INPUTS ----
The inputs of the function are the path to the data, the model to train and test.
---- PARAMETERS ----
the parameters determine the type of pre-process. Initially the datas are normalized, 
interpolated, inverted and the outliers are removed. To prevent from doing one of this 
process you need to put the parameters to False. 

=== ACCURACY ===
---- PARAMETERS ----
the parameters determine the type of split type. 
'tscv' : Time Serie split 
'classical' : usual train test split 
# ---- OUTPUT ----
The accuracy function return a list rmse, r2  
"""

def plot_predicted_values(y_pred, y_test, model):
    """
    The function allows to plot the predicted values and actual value of the BP
    """
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    ax1.scatter(range(len(y_pred)), y_pred, s=10, c='b', marker="s", label='prediction')
    ax1.scatter(range(len(y_pred)), y_test, s=10, c='r', marker="o", label='real values')
    plt.legend(loc='upper left')
    plt.xlabel('Time index')
    plt.ylabel('Blood pressure')
    plt.title(f'Comparaison of prediction and actual blood pressure values with {str(model)}')
    plt.show()

def compute_accuracies(model):
    # In order to obtain an excel file with the RMSE and R2 regarding the different type of 
    # pre-processing.
    # /!\ You need to create an excel named accuracy.xlsx in the same file as the code. 
    normalize = invert = [False, True]
    split_type = ['classical', 'tscv']
    acc = pd.DataFrame(columns = ['Cross validation', 'Normalize', 'Invert', 'RMSE', 'R2'])
    
    for split in split_type :
        for norm in normalize :
            for inv in invert : 
                select_model = Model('03-Oct-2023_patAnalysis_2.csv', model, 
                                     invert = inv,
                                     normalize = norm)
                rmse, r2= select_model.accuracy(split_type = split, n_splits = 4)
    
                acc = acc.append({'Cross validation' : split,
                                        'Normalize': norm,
                                        'Invert': inv,
                                        'RMSE': rmse,
                                        'R2' : r2}, ignore_index=True)
    
    acc.to_excel("accuracy.xlsx")

def k_fold():
    rmses = []
    for n in range (2, 8):
        rmse, r2= select_model.accuracy(split_type = 'tscv', n_splits=n)
        rmses.append(rmse)
    
    fig = plt.figure(2)
    plt.scatter(range(2,8), rmses)
    plt.xlabel('Number of fold')
    plt.ylabel('RMSE')
    plt.title('Elbow method')
    plt.show()

path = '03-Oct-2023_patAnalysis_2.csv'
#model = SVR(kernel = 'rbf')
#model = MLPRegressor(activation = "tanh", solver="lbfgs", max_iter=10000)
model = LinearRegression()
#model = train_regr = MLPRegressor(activation = "tanh", solver="lbfgs", max_iter=100000)
#model = PolynomialFeatures(degree = 3)
select_model = Model(path, model)
compute_accuracies(model)