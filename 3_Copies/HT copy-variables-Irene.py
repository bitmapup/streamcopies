#!/usr/bin/env python
# coding: utf-8

# ## Este notebook funciona con los datos v6 (7 variables) y con el HT de Irene
# 

# In[1]:


import sys
import requests
import inspect
import pandas as pd
import random
import csv
import numpy as np
import os
from methods import HoeffdingTree
from sklearn.metrics import confusion_matrix
print ('\n'.join(sys.path))


# ## Dynamic windows

# In[4]:


def checkRate(y):
    rate = -1
    count = np.unique(y,return_counts=True)
    #print("y",y)
    #print("count",count)
    #print("count len",len(count[0]) )
    if (len(count) >= 1):
        if (len(count[0]) == 1):
            classall   = count[0]
            classA     = classall[0]
            classB     = 0
            classCount = count[1]
            classCountA= classCount[0]
            classCountB= 0
        else:
            classall   = count[0]
            classA     = classall[0]
            classB     = classall[1]
            classCount = count[1]
            classCountA= classCount[0]
            classCountB= classCount[1]
            
        if (classCountA<classCountB): # classCountA Minoritary class
            rate = float(classCountA/(classCountA+classCountB))
        else: # classCountB Minoritary class
            rate = float(classCountB/(classCountA+classCountB))
    
    return rate


# In[5]:


def computeMetrics (confusionMatrix):
    tn, fp, fn, tp = confusionMatrix.ravel()
    precision = tp/(tp+fp)
    recall    = tp/(tp+fn)
    return precision,recall


# In[11]:


models = False
aux_cfiers = None
numrows = 0
f= open("metrics_hti_baseline.csv","w+")
minorityClassRate = 0.12
min_windows_size = 10000
X_ = []
y_ = []
i = 0
cfiers = HoeffdingTree(split_criterion="mean")

f.write("size_win,precision0,recall0,precision1,recall1\n")
with open("../data_output/export_dataframe_0v6.csv") as infile:
    for line in infile:
        numrows += 1
        aux_list= (line.split(","))
        #print ("aux_list",aux_list)
        aux_y = int(aux_list.pop())
        aux_x = np.array(aux_list.copy(),dtype=np.float32) 
        y_.append(aux_y)        
        X_.append(aux_x)
        i += 1
        #print ("y_",y_)
        #print ("checkRate(y_) {} >= {} minorityClassRate: {}".format( checkRate(y_),  minorityClassRate, (checkRate(y_) >= minorityClassRate)))
        if ( (checkRate(y_) >= minorityClassRate) and len(y_)>=min_windows_size ):
            print ("Rows processed: {}".format(numrows))
            
            X = np.asarray(X_).copy()
            y = np.asarray(y_).copy()
            
            # Fit HT models
            if (models):
                print("Model true")
                yPred = aux_cfiers.predict(X)
                cm0 = confusion_matrix(y,yPred)
                p0,r0 = computeMetrics(cm0)
                
                
                tn, fp, fn, tp = cm0.ravel()
                print("{},{},{},{},{},{},{},{}\n".format(i,p0,r0,str(np.unique(yPred,return_counts=True)),tn, fp, fn, tp ))
                f.write("{},{},{},{},{},{},{},{}\n".format(i,p0,r0,str(np.unique(yPred,return_counts=True)),tn, fp, fn, tp ))
                f.flush()
                
                cfiers.fit(X,y)
                aux_cfiers = cfiers
                
                i = 0
                X = None
                y = None
                X_ = []
                y_ = []
            else:
                print("Model False")
                cfiers.fit(X,y)
                aux_cfiers = cfiers
                models = True
                i = 0
                X = None
                y = None
                X_ = []
                y_ = []
            
    X = np.asarray(X_).copy()
    y = np.asarray(y_).copy()
    
    yPred = aux_cfiers.predict(X)
    cm0 = confusion_matrix(y,yPred)
    p0,r0 = computeMetrics(cm0)


    tn, fp, fn, tp = cm0.ravel()
    print("{},{},{},{},{},{},{},{}\n".format(i,p0,r0,str(np.unique(yPred,return_counts=True)),tn, fp, fn, tp ))
    f.write("{},{},{},{},{},{},{},{}\n".format(i,p0,r0,str(np.unique(yPred,return_counts=True)),tn, fp, fn, tp ))
    f.flush()
    i = 0
    X = None
    y = None
    X_ = []
    y_ = []
    f.write("Total rows processed: {}".format(numrows))


# In[6]:





# In[ ]:





# In[ ]:




