# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:58:04 2023

@author: tanne
"""

import matplotlib.pyplot as plt

import pandas
import numpy as np 
import csv
# read excel in data frame 
file = open('03-Oct-2023_patAnalysis_2.csv',"r")
data = csv.reader(file, delimiter = ",")
data = np.array(list(data))
for i in range(2343):    #Allows conversion of data into float by replacing the empty spots by 0
    for j in range(8):
        if data[i][j]== '' :
            data[i][j]=0
X = data[1:, 1:4].astype(float)
y = data[1: , 5].astype(float)
for i in range(2342):    #Allows to plot the real data
    for j in range(3):
        if X[i][j]== 0 :
            X[i][j]= None
    if y[i]==0:
        y[i]=None

fig1 = plt.figure()
plt.subplot(2,1,1), plt.plot(X[:,0],X[:,2], label='PAT'),plt.ylabel("PAT")
plt.subplot(2,1,2), plt.plot(X[:,0],y, 'g', label='BP'),plt.ylabel("Blood Pressure")

fig2 = plt.figure()
plt.plot(X[:,2],y)
plt.xlim([0.1,0.4])