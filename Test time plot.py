# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:58:04 2023

@author: tanne
"""

import matplotlib as plt

import pandas as pd 
import numpy as np 
# read excel in data frame 
df = pd.read_excel(r'2023-10-03_dataset/03-Oct-2023_patAnalysis_2.xlsx') 
 
# convert a data frame to a Numpy 2D array 
np.asarray(df) 
X, y = data[:, 0], data[:, 1]