#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 22:29:48 2019

@author: younessubhi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.linalg import svd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

#%% define plot settings

# defualt pyplot colour scheme
print(plt.rcParams['axes.prop_cycle'].by_key(['color']]))

#%% initial data look-through

# load data to pd dataframe
df = pd.read_csv("../xyz_dataArray.csv", dtype=None, delimiter=',', encoding=None, usecols = range(145))

df.head()

# extract attributes
attributeNames = df.columns

# preliminary overview of data
df.describe()

# remove all N/A rows, if any
df = df.dropna()
# df = df.drop('-', axis=1) # if any cols have to dropped // when class label is defined

df.describe()

# pair plot to quickly find any "easy" tracks of proportionality
sns.pairplot(df[:, df.columns=])

df.corr()

plt.figure()
sns.heatmap(df.corr(), annot=True, linewidths=2)

#%% PCA begin
# class label

print(df.columns)
print(df.head())

discrete_mask = {"C1" : 1, "C2" : 2, "C3" : 3, "C4" : 4, 
                 "C5" : 5, "C6" : 6, "C7" : 7, "C8" : 8,
                 "C9" : 9, "C10" : 10, "C11" : 11, "C12" : 12,
                 "C13" : 13, "C14" : 14, "C15" : 15, "C16" : 17,
                 "C17" : 17, "C18" : 18, "C19" : 19
                 }

# return the unique scs values
classLabels = df['scs'].unique()
# class vector
y = np.asarray(df['scs'])
classVec = len(classLabels)

# remove labeled data
df = df.drop("scs", axis=1)

# check if it worked
print(df.head())

#%% extract raw data
attributeNames = npasarray(df.columns)
print(attributeNames)

X = df.values
N, M = X.shape
print(N,M)
