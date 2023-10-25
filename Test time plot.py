# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:58:04 2023

@author: tanne
"""

import matplotlib.pyplot as plt


import numpy as np 
import csv
# read excel in data frame 
file = open('03-Oct-2023_patAnalysis_2.csv',"r")
data = csv.reader(file, delimiter = ",")

data = np.array(list(data))
for i in range(2343):
    for j in range(8):
        if data[i][j]== '' :
            data[i][j]=0
X = data[1:, 1:4].astype(float)
y = data[1: , 5].astype(float)
fig1 = plt.figure()
plt.plot(X[:,0],X[:,2], label='PAT'),plt.xlabel("time"),plt.ylabel("PAT")
fig2 = plt.figure()
plt.plot(X[:,0],y, 'g', label='BP'),plt.xlabel("time"),plt.ylabel("Blood Pressure")