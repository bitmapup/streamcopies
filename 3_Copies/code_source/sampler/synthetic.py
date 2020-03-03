#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SYNTHETIC

Created on Thu Apr 12 14:11:06 2018

@author: irene

Create a set of synthetic datasets
for each of the original problems. In each
case, perform a different sampling through
the learned class probability space.

"""

import os
import sys
import numpy as np
import joblib
import warnings
import logging
from os import path, remove
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

if path.isfile('synthetic.log'):
    remove('synthetic.log')

from sampler import RandomNormal

root = '/Users/irene/Documents/Projects/fat/results/' 
    
n_classifiers = ["xgboost"]

N_ITER = 10
MAX_QUERIES = 1000000
N_EXPLORE = int(3*MAX_QUERIES)

def sample_train(t, clf, X_train, y_train, dtypes, max_queries, n_explore):
    logger.info('TRAIN iteration: {}'.format(t+1))
    sampler = RandomNormal(max_queries=max_queries, n_explore=n_explore)
    X_train_, y_train_ = sampler.query(clf, X_train, y_train, dtypes)
    return {'X_train_': X_train_, 'y_train_': y_train_}

def sample_test(t, clf, X_train, y_train, dtypes, max_queries, n_explore):
    logger.info('TEST iteration')
    sampler = RandomNormal(max_queries=max_queries, n_explore=n_explore)
    X_test_, y_test_ = sampler.query(clf, X_train, y_train, dtypes)
    return {'X_test_': X_test_, 'y_test_': y_test_}

def sample(n_clf, path, n_iter, X_train, y_train, dtypes, max_queries, n_explore):

    clf =joblib.load(root+'open_'+n_clf+'.pkl')

    train_ = []
    test_ = []
    
    test_ = Parallel(n_jobs=1)(delayed(sample_test)(t, clf, X_train, y_train, dtypes, max_queries, n_explore) for t in range(1))
    train_ = Parallel(n_jobs=1)(delayed(sample_train)(t, clf, X_train, y_train, dtypes, max_queries, n_explore) for t in range(n_iter))
    
    #os.makedirs(path+'synthetic dataset/'+dataset+'/', exist_ok=True)
    #path = path+'synthetic dataset/'+dataset+'/'
    
    joblib.dump(test_, path+str(max_queries)+'_open_test.pkl')
    joblib.dump(train_, path+str(max_queries)+'_'+str(n_iter)+'_open_train.pkl')
    
    logger.info('{}: DONE!')

if __name__ == '__main__':
    
    # Create logger
    logger = logging.getLogger()
    
    # Create file handler
    fh = logging.FileHandler('open_synthetic.log')
    fh.setLevel(logging.DEBUG)
    
    # Create console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Create formatter and add it to both handlers
    formatter = logging.Formatter('%(asctime)s (%(filename)s) %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    
    logger.info('n_iter: {}'.format(N_ITER))
    logger.info('max_queries: {}'.format(MAX_QUERIES))
    logger.info('n_explore: {}'.format(N_EXPLORE))

    for n_clf in n_classifiers:

        logger.info('n_clf: {}'.format(n_clf))
        
        data = joblib.load(root+'open_data.pkl')
        X_train = data['X_train']
        y_train = data['y_train']
        dtypes = joblib.load(root+'dtypes.pkl')

        logger.info('n_classes: {}'.format(len(np.unique(y_train))))
        logger.info('classes: {}'.format(np.unique(y_train)))

        sample(n_clf, root, N_ITER, X_train, y_train, dtypes, MAX_QUERIES, N_EXPLORE)




