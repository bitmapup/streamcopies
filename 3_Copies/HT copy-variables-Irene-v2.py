#!/usr/bin/env python
# coding: utf-8

# ## Este notebook funciona con los datos v6 (7 variables) y con el HT de Irene
# 

# In[6]:


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


# ## Data generation

# In[7]:





# ## Copy

# In[8]:


def generate_uniform_adnostic_sample():
    
    dict_aux = {}
    mu = 0.0 
    sigma = 1.0
    aux = random.uniform(mu, sigma)
    dict_aux['amount_usd']= aux*-1 if (aux<0) else aux
    aux = random.uniform(mu, sigma)
    dict_aux['client_age']= aux*-1 if (aux<0) else aux
    
    aux = random.uniform(mu, sigma)
    dict_aux['client_gender']=1 if (int(round(aux,0)) == 1 ) else 0
    
    aux = random.uniform(mu, sigma)
    dict_aux['debit_type']=1 if (int(round(aux,0)) == 1 ) else 0
        
    aux = random.uniform(mu, sigma)
    dict_aux['agency_region']= int(round(aux * 12,0))
    
    aux = random.uniform(mu, sigma)        
    dict_aux['merchant_departement']= int(round(aux * 25,0))
    
    aux = random.uniform(mu, sigma)        
    dict_aux['coicop']= int(round(aux * 12,0))
        
    return dict_aux
    


# In[9]:


def generate_gaussian_sample_from_original(pX,py,oracle,minority_label,original=False):
    size = len(py)
    #size = 2000
    print("generate_gaussian_original minority nb: ",size )    
    i = 0
    #mylist = []
    X_aux = []
    
    while (i<size):
        
        row = pX[i].copy()
        #print (type(row))
        i+=1
        if (not original):
            #print (row,i)
            if (np.random.randint(2) == 1):
                mu = 0.006903
                sigma = 0.017028
                aux = random.uniform(mu, sigma)
                row[0] += aux
            elif (np.random.randint(2) == 1):
                mu = 0.006903
                sigma = 0.017028
                aux = random.uniform(mu, sigma)
                row[0] += aux
                mu = 0.343192
                sigma = 0.144008
                aux = random.uniform( sigma,mu)
                row[1] += aux
            else:
                mu = 0.343192
                sigma = 0.144008
                #aux = random.gauss(mu, sigma)
                aux = random.uniform( sigma,mu)
                row[1] += aux
                
        
        y_iter = oracle.predict( np.asarray( [row] ) )

        if (y_iter[0]  == 1):       
            #print ("prediction: ",y_iter,i)
            row = np.append(row,[1],axis=0)
            X_aux.append (row)
    #y_res = oracle.predict( np.asarray(X_aux) )
    #print ("oracle.predict(X) : ", y_res)
    #print (np.unique(y_res,return_counts=True) )
    return pd.DataFrame(X_aux)


# In[10]:


def generate_uniform_sample_from_original(pX,py,oracle,minority_label,original=False):
    xmin = np.min(pX,axis=0)
    xman = np.max(pX,axis=0)
    x0_min =  xmin[0]
    x1_min =  xmin[1]
    x0_max =  xman[0]
    x1_max =  xman[1] 
    print (x0_min,x1_min,x0_max,x1_max)
    size = len(py)
    #size = 2000
    print("generate_uniform_from_original minority nb: ",size )    
    i = 0
    #mylist = []
    X_aux = []
    
    while (i<size):
        
        row = pX[i].copy()
        #print (type(row))
        i+=1
        if (not original):
            #print (row,i)
            if (np.random.randint(2) == 1):
                aux = random.uniform(x0_min, x0_max)
                row[0] += aux
            elif (np.random.randint(2) == 1):
                aux = random.uniform(x0_min, x0_max)
                row[0] += aux
                aux = random.uniform( x1_min,x1_max)
                row[1] += aux
            else:
                aux = random.uniform( x1_min,x1_max)
                row[1] += aux
                
        
        y_iter = oracle.predict( np.asarray( [row] ) )
        
        if (y_iter[0]  == 1):       
            #print ("prediction: ",y_iter,i)
            X_aux.append (row)
    #y_res = oracle.predict( np.asarray(X_aux) )
    #print ("oracle.predict(X) : ", y_res)
    #print (np.unique(y_res,return_counts=True) )
    return pd.DataFrame(X_aux)


# In[11]:


def generate_syntethic_data (nb_samples,pX,py,oracle,minority_label,pmax_iterations=10):
    
    newdf = pd.DataFrame()
    
    #Minority class
    print ("generating....")
    sampled_df = generate_gaussian_sample_from_original(pX,py,oracle,pmax_iterations)
    nb_sampled = sampled_df.shape[0]
    print("nb_sampled: ",sampled_df.shape)
    
    ii = 0
    print ("while....")
    if (sampled_df.shape[0]<=(nb_samples/2)):
        #print (sampled_df.shape[0]," < ",(nb_samples/2))
        #print ("max_iterations: ",pmax_iterations)
        while   (ii < pmax_iterations):    
            #print ('i : ',ii)
            sampled_df_aux = generate_gaussian_sample_from_original(pX,py,oracle,pmax_iterations)
            sampled_df = sampled_df.append(sampled_df_aux, ignore_index = True) 
            #print (sampled_df_aux.shape)
            #print (sampled_df.shape)
            ii+=1
        if (sampled_df.shape[0]<=(nb_samples/2)):
            ii = 0
            print ("second sampling")
            while   (ii < pmax_iterations  ):
                sampled_df_aux = generate_gaussian_sample_from_original(pX,py,oracle,pmax_iterations,original=True)
                sampled_df = sampled_df.append(sampled_df_aux, ignore_index = False) 
                print (sampled_df.shape)
                ii+=1
                if (sampled_df.shape[0]>=(nb_samples/2)):
                    print ("Break : ", sampled_df.shape)
                    break
    if (sampled_df.shape[0] >= int(nb_samples/2) ):
        newdf = sampled_df.iloc[0:int(nb_samples/2)]#, ignore_index = True)
    else :
        newdf = sampled_df
    column_names = ['amount_usd',
                    'client_age',
                    'client_gender',     
                    'debit_type',        
                    'agency_region',     
                    'merchant_departement',
                   'coicop',
                   'social_class']
    newdf.columns = column_names
    print ("newdf shape", newdf.shape)
    print ("newdf cols", newdf.columns)
    
    #All classes
    
    i = 0
    column_names = ['amount_usd',
                    'client_age',
                    'client_gender',     
                    'debit_type',        
                    'agency_region',     
                    'merchant_departement',
                   'coicop']
    df = pd.DataFrame(columns = column_names)
    while (i < (nb_samples/2)):
        row ={}
        row = generate_uniform_adnostic_sample()
        df = df.append(row  , ignore_index=True)
        i+=1
  
    aX=np.asarray(df)
    aY=oracle.predict(aX)
    df['social_class']=aY
    
    print ("df shape", df.shape)
    print ("df cols", df.columns)
    
    newdf = newdf.append(df)
    print ("newdf shape", newdf.columns)
    print ("newdf cols", newdf.columns)
    newdf = newdf.sample(frac = 1)
    
    return newdf #sampled_df


# df = pd.read_csv('../data_output/export_dataframe_0v6.csv',header=None,names=dataset_feature_names)
# oracle = cfiers[1]
# column_label = 'social_class'
# minority_label = 1
# nb_samples = 10000
# max_iterations = 30
# 
# minority_df = df.loc[df['social_class'] == 1]
# #print ("minority",np.unique(minority_df,return_counts=True))
# ay=np.asarray(minority_df['social_class'])
# aX=np.asarray(minority_df.iloc[:,0:-1])
# 
# 
# sampled_df = generate_syntethic_data (nb_samples,X,y,oracle,column_label,max_iterations)
# 
# #sampled_df = generate_uniform_adnostic_sample_from_original(X,y,oracle,minority_label,original=True)
# #np.unique(cfiers[0].predict(X_synthetic),return_counts=True)
# print ("Final ",sampled_df.shape)
# sampled_df.head()

# ## Dynamic windows

# In[12]:


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


# In[13]:


def computeMetrics (confusionMatrix):
    tn, fp, fn, tp = confusionMatrix.ravel()
    precision = tp/(tp+fp)
    recall    = tp/(tp+fn)
    return precision,recall


# In[14]:


models = []
no_first_model = False
numrows = 0
f= open("metrics_hti.csv","w+")
minorityClassRate = 0.12
min_windows_size = 10000
X_ = []
y_ = []
last_X = []
last_y = []
i = 0
cfiers = HoeffdingTree(split_criterion="mean")

f.write("size_win,precision,recall,count,tn, fp, fn, tp\n")
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
            if (no_first_model):
                # Semi copy
                minority_label = 1
                nb_samples = 10000
                max_iterations = 30
                oracle = models[-1]
                column_label = 'social_class'
                sampled_df = generate_syntethic_data (nb_samples,last_X,last_y,oracle,column_label,max_iterations)
                sampled_y=np.asarray(sampled_df['social_class'])
                sampled_X=np.asarray(sampled_df.iloc[:,0:-1])
                y = np.concatenate((sampled_y, y), axis=0)
                X = np.concatenate((sampled_X, X), axis=0)
                # Learn
                cfiers.fit(X,y)
                models.append(cfiers)
                
                if (len(models)>=3):
                    cfiers = models[-2]
                    yPred0 = cfiers.predict(X)
                    cm0 = confusion_matrix(y,yPred0)
                    p0,r0 = computeMetrics(cm0)
                    tn, fp, fn, tp = cm0.ravel()
                    f.write("{},{},{},{},{},{},{},{}\n".format(i,p0,r0,str(np.unique(yPred0,return_counts=True)),tn, fp, fn, tp ))
                    f.flush()
                i = 0
                last_X = X
                last_y = y
                X = None
                y = None
                X_ = []
                y_ = []
            else:
                cfiers.fit(X,y)
                models.append(cfiers)
                no_first_model = True
                i = 0
                last_X = X
                last_y = y
                X = None
                y = None
                X_ = []
                y_ = []
            
    X = np.asarray(X_).copy()
    y = np.asarray(y_).copy()
    
    if (len(models)>=3):
                    cfiers = models[-2]
                    yPred0 = cfiers.predict(X)
                    cm0 = confusion_matrix(y,yPred0)
                    p0,r0 = computeMetrics(cm0)
                    tn, fp, fn, tp = cm0.ravel()
                    f.write("{},{},{},{},{},{},{},{}\n".format(i,p0,r0,str(np.unique(yPred0,return_counts=True)),tn, fp, fn, tp ))
                    f.flush()
    i = 0
    X = None
    y = None
    X_ = []
    y_ = []
    f.write("Total rows processed: {}".format(numrows))


# In[ ]:





# In[ ]:





# In[ ]:




