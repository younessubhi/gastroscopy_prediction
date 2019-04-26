#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 20:15:00 2019

@author: younessubhi
"""

import numpy as np
import os, csv

#%% defining functions

def ToMilSecs(dateVect):
    # input data vector, (hours, minutes, seconds)
    # output as miliseconds
    hours = dateVect[0]
    mins = dateVect[1]
    secs = dateVect[2]
    milsecs = dateVect[3]
    
    return milisecs + 1000 * (secs + 60 * (mins + hours * 60))

def ReformatRow(row):
    # reformat  the rows into desirable format
    out = []
    for k, dat in enumerate(row):
        if k == 0:
            dat = ToMilSecs(list(map(int, row[k].split(":"))))
        out.append(float(dat))

    return out

def getData(control = 0, dataFile = None):
    
    """ gets data from CSV. normal Olympus data-csv format is expected
    inputs:
        control, if a file path to specific csv file is given or not
        control = 0, stnd path is used
        control = 1, file dialog for choosing specific file
        control = 2, give datapath as str
        
    outputs:
        dataArray, the actual (x,y,z) points of the scope coils
        attributeNames, the names of the columns in dataArray
    """
    
    # data_path = "cd .. /gastroscopy_prediction/CoPS_data"
    if control == 0:
        files_names = os.listdir(data_path)
        data = []
        
        with open(data_path + files_names[1], newline = "") as csvfile:
            filereader = csv.reader(csvfile, delimiter = ",")
            for row in filereader:
                data.append(row)
    elif control == 1:
        root = tk.Tk()
        root.file_name = tk.filedialog.askopenfilename(initialdir = data_path)
        root.withdraw
        data = []
        
        with open(root.file_name, newline = "") as csvfile:
            filereader = csv.reader(csvfile, delimiter = ",")
            for row in filereader:
                data.append(row)
    elif control == 2 and dataFile != None:
        data = []
        # with open(dataFile, newline = "") as csvfile:
        with open(dataFile, "r") as csvfile:
            filereader = csv.reader(csvfile, delimiter = ",")
            for row in filereader:
                data.append(row)
                
    dataArray = np.empty((len(data) -1, len(data[0])))
    
    # first row is attributes and rest is data
    for i, row in enumerate(data):
        if i == 0:
            attributeNames = row
        else:
            #print(i)
            dataArray[i-1] = ReformatRow(row)
            
        refTime = dataArray[0,0]
        
        # shift so measurement starts at 0
        for i, dat in enumerate(dataArray):
            dataArray[i, 0] -= refTime
            
    return dataArray, attributeNames
    
def getCloud(dataArray):
        # get all (x,y,z), every coil, throughout whole procedure
        numCoils = 19
        n_data = len(dataArray[:,1])
        xData = np.zeros((numCoils, n_data))
        yData = np.zeros((numCoils, n_data))
        zData = np.zeros((numCoils, n_data))
        
        for nr in range(0, n_data - 1):
            for i in range(0,numcoils):
                xData[i, nr] = (dataArray[nr, i2])
                yData[i, nr] = (dataArray[nr, i2 + 1])
                zData[i, nr] = (dataArray[nr, i2 + 2])
    
        return xData, yData, zData

#%% Script
    
result_loc = "/CoPS_data"
result_ls = [result_loc + "\\" + name for name in os.listdir(result_loc) if (os.path.isdir(os.path.join(result_loc, name)) and name != "analyzed")]

print("folder " + result_loc + "\n")
print(result_loc)

no_series = len(result_list)

for i in range(0, no_series):
    file_no = int(no_series - i - 1)
    
    print(file_no)
    
    result_dir = result_list[file_no]
    result = result_dir.split("\\")[1]
    result_dir = result_loc + "\\" + result + "\\"
    
    # load corrected file
    data_dir = result_dir + "corrected.csv"
    
    scs_dir = result_dir + "/scs.txt"
    scs_file = open(scs_dir, 'r')
    
    content = scs_file.readline().split(";")
    
    scs.file.close()
    
    start = int(float(content[0]))
    cecum = int(float(content[2])) - start
    stop = int(float(content[4])) - start
    
    # print filename
    print("\n filepath:\n " + result + "\n")
    print("%.f of %.f" %(file_no + 1, no_series))
    
    # load data file
    dataArray, AttributeNames = getData(control = 2, dataFile = data_dir)
    
    # construct time vector
    time_vec = dataArray[:,0] - dataArray[0,0]
    # mean samp freq
    freq = len(time_vec) / (time_vec[-1] / 1000)
    print("%.2f Hz sample frequency" %freq)
    
    # max insertion index
    max_in = cecum
    # max insertion time
    max_in_time = (cecum) / freq
    print("%.f s to the max in " %max_in_time)
    print("%.f max in datapoint" %max_in)
    
    # recording time
    rec_time = time_vec[-1] / 1000
    # retraction time
    retract_time = rec_time - max_in_time
    
    print("%.f datapoints" %len(time_vec))
    print("%.f s recoding time" %rec_time)
    print("%.f s to the max in " %max_in_time)
    print("%.f s from max in " %retract_time)
    
    
    # construct individual matrices for coordinates (x,y,z)
    # with shape [numCoils, numDataPoints]
    x.y,z = getCloud(dataArray)

    