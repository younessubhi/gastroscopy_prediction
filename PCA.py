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

#%% one out of K-encoding for coils
print(attributeNames)
coils = np.array(X[:, 0], dtype=int).T
# K number of cols
K = coils.max() + 1

# empty 19 col N row array
coil_encoding = np.zeros((coil.size, K))
# map values
coil_encoding[np.arrange(coil.size), coils]

# concatante to end of prev data and remove the first col
X = np.concatenate( (X[:, 1:M], coil_encoding), axis=1)
 
# remap attributes
attributeNames = np.append(attributeNames[1:M], ["C1", "C2",
                                               "C3", "C4",
                                               "C5", "C6",
                                               "C7", "C8",
                                               "C9", "C10",
                                               "C11", "C12",
                                               "C13", "C14",
                                               "C15", "C16",
                                               "C17", "C17",
                                               "C19"])

print(X[0,:])

# sanity check
print(attributeNames)
 
# should have gained 1 more attribute due to K encoding
N, M = X.shape
print(N,M)

#%% one-out-of-K encoded done, now for normalization
 
# normalize with mean value
X_norm = X - np.ones((N,1))*X.mean(axis=0)
print(np.abs(X_norm[0]).max() / np.abs(X_norm[0]).min())
 
# subtracting standard deviation reducing factor
X_norm = (X - np.ones((N,1)) * X.mean(axis=0)) / X.std(axis=0)
print(np.abs(X_norm[0]).max() / np.abs(X_norm[0]).min())
 
#%% singular value decomposition (SVD)
 
U,S,V = svd(X_norm,full_matrices=False)
 
# variance explained:
rho = (S*S) / (S*S).sum() 
print(S)
print("Sigma matrix:", np.round(S,2))
 
print("V:", np.round(V,2))
 
print(rho)
 
threshold = 0.9
 
# plot 'variance explained'
plt.figure(figsize = (width, height))
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
#plt.xticks([1,2,3,4,5,6,7,8,9, 10])
plt.xticks(np.linspace(1,M,M))
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
#plt.savefig('Figures/Variance_explained.pgf')
plt.show()
 
#%% PCA component coefficients

print(V[:,0])
print(len(attributeNames))
pcs = [0,1,2]
#pcs = [7,8,9]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
f = plt.figure(figsize = (width_fullsize,height))
#f = plt.figure(dpi=300)
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
 
 
locs, labels = plt.xticks(r+bw, attributeNames)
#plt.setp(labels, rotation=90)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs, loc='upper right')
plt.grid()
#plt.savefig('Figures/PCA_Coefficients.pgf')
plt.show()
 
#%% plotting PC1 to PC3 (2d)
v1 = V[:,0] # PC1
v2 = V[:,1] # PC2
v3 = V[:,3] # PC3
print("PC1:", v1)
print("PC2:", v2)
print("PC3:", v3)
 
# check for orthogonality (0 = linear independence)
print(np.dot(v1,v2))
 # vector origin point
origin = [0], [0]
 
fig, ax = plt.subplots(dpi=150)
plt.quiver(*origin, v1[0:2], v2[0:2], color=['r','b','g'])
 
custom_lines = [Line2D([0], [0], color='r', lw=4),
                Line2D([0], [0], color='b', lw=4)]
ax.legend(custom_lines, ['PC1', 'PC2'])
 
plt.tight_layout()
plt.show()
 
 
#%% transform to PC domain & 2d     plot.
B = X@V[:,0:3]
 
#plt.figure(figsize = (width,height))
plt.figure(dpi=150)
color_array = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
i = 0
for c in classLabels:
    class_mask = y == c
    plt.scatter(B[class_mask,0], B[class_mask,1], label=c, color=color_array[i], alpha=.75)
    i += 1
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.legend()
#plt.legend(title='CDR', bbox_to_anchor=(0., 1.02, 1, .102), loc=3, ncol=4, mode="expand", borderaxespad=0)
#plt.savefig('Figures/2dprojection.pgf', bbox_inches="tight")
plt.show()
 
print(np.corrcoef(B[:,0], B[:,1]))
#%% plotting 3d, yaaaaay
 
 
#fig = plt.figure(dpi=150)
fig = plt.figure(figsize = (width_fullsize,height))
ax = fig.add_subplot(111, projection='3d')
i = 0
for c in classLabels:
    class_mask = y == c
    ax.scatter(B[class_mask,0], B[class_mask,1], B[class_mask,2], label=c, color=color_array[i], alpha=.75)
    i += 1
 
 
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend(loc='upper left')
ax.view_init(30, 120)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.tight_layout()
plt.savefig('Figures/3dprojection.pgf')
plt.show()
 
print(np.corrcoef(B[:,0], B[:,1]))
print(np.corrcoef(B[:,0], B[:,2]))
print(np.corrcoef(B[:,1], B[:,2]))
