#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:08:07 2023

@author: mariafoyen
"""

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from statsmodels.tsa.stattools import adfuller, kpss
import csv

# read in csv file
df = pd.read_csv("/Users/mariafoyen/Documents/Experimental-project/03-Oct-2023_patAnalysis_2.csv").iloc[24:]
list(df.columns)







