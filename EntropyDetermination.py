#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:34:10 2019

@author: younessubhi
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os,sys,time
import datetime

from scipy import signal, misc


import HelperFunctions as HF

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

import math

# myp = os.path.dirname(sys.argv[0])
# datap = myp.split("lib")[0] + os.sep + "data"
# resp = myp.split("lib")[0] + "results" + os.sep

# print(myp,datap,resp)

myp = '/Users/younessubhi/Documents/GitHub/gastroscopy_prediction'
datap = myp.split("lib")[0] + os.sep + 'CoPS_data' + os.sep + '01-14 -Y 2019- - kl 09-56 (32 min)' + os.sep
datafp = [datap + name for name in os.listdir(datap) if (os.path.isdir(os.path.join(datap, name)) and name != "analyzed")]



def str_entropy(string):
	"Calculates the Shannon entropy of a string"
	# get probability of chars in string
	prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]
	# calculate the entropy
	entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])
	return entropy


def str_entropy_ideal(length):
	"Calculates the ideal Shannon entropy of a string with given length"
	prob = 1.0 / length
	return -1.0 * length * prob * math.log(prob) / math.log(2.0)




def map_angle(angle):
	if 0*np.pi / 4 <= angle <= 1*np.pi / 4:
		letter = "C"
	elif 1*np.pi / 4 <= angle <= 2*np.pi / 4:
		letter = "O"
	elif 2*np.pi / 4 <= angle <= 3*np.pi / 4:
		letter = "L"
	elif 3*np.pi / 4 <= angle <= 4*np.pi / 4:
		letter = "P"
	elif 4*np.pi / 4 <= angle <= 5*np.pi / 4:
		letter = "T"
	elif 5*np.pi / 4 <= angle <= 6*np.pi / 4:
		letter = "A"
	elif 6*np.pi / 4 <= angle <= 7*np.pi / 4:
		letter = "E"
	elif 7*np.pi / 4 <= angle <= 8*np.pi / 4:
		letter = "K"
	return letter
	


# myp = os.path.dirname(sys.argv[0])+os.sep

#Read Data
fi = 0

ymax = 500
ymin = -300


# ymin = -1000
# ymax = 500
xmin = -1200
xmax = 0


for subdir, dirs, files in os.walk(datap):
	for file in files:
		if file.endswith("csv"):
			print(file)	
			fi+=1
			with open(datap + file) as tempf:
				lendata = len(tempf.readlines())
			
			DATA = np.zeros([lendata,24*3+1])
			T_zero = ""

			dataArray, attributeNames = HF.getData(control = 2, dataFile = datap + file)

			# fig (ax,ax2) = plt.subplots(121)

			# fig = plt.figure()
			# ax = fig.add_subplot(121)
			# ax2 = fig.add_subplot(122)
			
			# fig.set_tight_layout(True)


			# fig, ax = plt.subplots()
			xdata, ydata = [], []
			# ln, = plt.plot([], [], 'ro', animated=True)
			# print('fig size: {0} DPI, size in inches {1}'.format(fig.get_dpi(), fig.get_size_inches()))

			# Plot a scatter that persists (isn't redrawn) and the initial line.
			# x = np.arange(0, 20, 0.1)
			# ax.scatter(x, x + np.random.normal(0, 3.0, len(x)))
			
			for ik in range(0,100,10):
				x,y,z = HF.getScopePos(ik, dataArray, spline_interpolate = False)
				x=np.array(x)				
				y=np.array(y)				
				z=np.array(z)				
				plt.subplot(131)
				plt.plot(-x,-y, 'k-', linewidth=2)
				plt.plot(-x[0],-y[0], 'ro')
				plt.plot(-x[1],-y[1], 'bo')
				
				plt.plot(-x[-1],-y[-1], 'go')
				
				word = ""
				plt.subplot(133,projection='polar')
				# ax = plt.subplot(111, )
				
				for coil in range(1,len(x)-1):
					a = np.array([-x[coil-1],-y[coil-1],z[coil-1]])
					b = np.array([-x[coil],-y[coil],z[coil]])
					c = np.array([-x[coil+1],-y[coil+1],z[coil+1]])
					ba = a - b
					bc = c - b

					cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
					angle = np.arccos(cosine_angle)
					word += map_angle(angle)
					plt.plot(1, angle)
					
				print(word,str_entropy(word),str_entropy_ideal(len(word)))
				

				plt.subplot(132)
				plt.plot(z,-y, 'k-', linewidth=2)
				
				plt.show()

			
			




# Loop Coils

# Loop Time



		

SS  = 30

gridy = np.arange(ymin,ymax,SS)
gridx = np.arange(xmin,xmax,SS)
bins =len(gridx)*len(gridy)


		
		# plt.show()

#Detect Anatomy
for x in gridx:
	plt.plot([x,x],[ymin,ymax],'k-',alpha=.3)

for y in gridy:	
	plt.plot([xmin,xmax],[y,y],'k-',alpha=.3)
	























