#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BASE_SAMPLER
Created on Thu Jan 25 16:25:12 2018

@author: irene

Different sampling strategies
to query pre-trained models and obtain new
synthetic datasets.
"""

import scipy
import random
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils import check_random_state

def check_estimator(estimator):
    """ Make sure that an estimator implements the necessary methods.
    """
    if not hasattr(estimator, "predict"):
        raise ValueError("The base estimator should implement predict")

def check_initial_samples(init_samples, init_labels, n_dim):
    """ Define initial samples.
    """
    if (init_samples is None) and (init_labels is None):
        return np.empty((0, n_dim)), None
    else:
        return init_samples, init_labels

def define_domain(X, domain):
    mins_ = np.percentile(X, 100-domain, axis=0) if not isinstance(X, pd.DataFrame) else X.quantile(1-domain/100.).values
    maxs_ = np.percentile(X, domain, axis=0) if not isinstance(X, pd.DataFrame) else X.quantile(domain/100.).values
    return mins_, maxs_

def define_uniques(X):
    uniques_ = [np.unique(X[:, col]) for col in range(X.shape[1])] if not isinstance(X, pd.DataFrame) else [X[col].unique() for col in X.columns]
    return uniques_

class BaseSampler(object):

    def __init__(self,
                 max_queries=1000,
                 n_exploit=10,
                 n_explore=100000,
                 n_batch=100,
                 step=1e-3,
                 random_state=None,
                 verbose=False):

        self.max_queries = max_queries
        self.n_exploit = n_exploit
        self.n_explore = n_explore
        self.n_batch = n_batch
        self.step = step
        self.random_state = random_state
        self.verbose = verbose

        self.logger = logging.getLogger(__name__)

    def _make_random_queries_2(self, n_samples):
        """ Randomly sample the class probability space
        """
        print ("BaseSampler","_make_random_queries 2")

        numeric = np.random.uniform(low=self.mins_, high=self.maxs_, size=(n_samples, self.n_numeric_))

        categorical = np.asarray([])
        for i in range(self.n_categorical_):
            categorical = np.append(categorical, np.random.choice(a=self.uniques_[i], size=n_samples, replace=True))
        categorical = np.reshape(categorical, (n_samples, self.n_categorical_), order='F')

        X_ = np.column_stack((numeric, categorical)) if not isinstance(self.X, pd.DataFrame) else pd.DataFrame(data=np.column_stack((numeric, categorical)), columns=self.cols_)
        
        #if isinstance(self.oracle, BaseEstimator):
        #y_ = self.oracle.predict(X_)
        #else:
        y_ = np.argmax(self.oracle.predict(X_), axis=0)

        return X_, y_.astype(int)

    def _make_random_queries(self):
        print ("BaseSampler","_make_random_queries")
        assert self.n_explore%self.n_batch == 0
        batch_size = int(self.n_explore/self.n_batch)

        X = np.empty(shape=(0, self.n_numeric_+self.n_categorical_)) if not isinstance(self.X, pd.DataFrame) else pd.DataFrame(columns=self.cols_)
        y = np.empty(shape=(0,))

        for i in range(self.n_batch):
            self.logger.info('batch {}'.format(i))
            X_hat, y_hat = self._make_random_queries_2(batch_size)
            X = np.vstack((X, X_hat)) if not isinstance(X, pd.DataFrame) else pd.concat([X, X_hat])
            y = np.append(y, y_hat)
            del X_hat, y_hat

        return X, y.astype(int)

    def _query(self):
        print ("BaseSampler ", " _query")
        pass

    def query(self,
              oracle,
              X,
              y=None,
              dtypes=None,
              domain=95):

        """ Query a trained model to obtain predicted class labels.

            Parameters
            ----------
            oracle : estimator object
            Trained primary model to use as an oracle
            to make queries. This is assumed to implement `predict` and
            `predict_proba` methods.

            X : array-like or sparse matrix, shape = [n_samples, n_features]
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features. Original data points
            (used to define feature domains).

            y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Set of labels, where n_samples is the number of samples.

            domain: int, optional (default=95)
            Range of original points to use for domain definition
            during sampling. If 100, the whole range of points is used.
            """
        print ("BaseSampler ", " query")
        self.domain= domain

        # Check estimator has necessary methods implemented
        if isinstance(oracle, BaseEstimator):
            check_estimator(oracle)
        self.oracle = oracle

        # Preprocessing.
        self.X = X
        self.y = y
        dtypes = check_array(dtypes, ensure_2d=False, ensure_min_samples=1, dtype=None)
        
        self.logger.info(type(self.X))

        if not isinstance(X, pd.DataFrame):
            check_consistent_length(self.X, self.y)
        else:
            self.cols_ = self.X.columns
        _, self.n_dim_ = self.X.shape
        self.classes_ = np.unique(self.y)
        self.n_classes_ = len(np.unique(self.y))
        self.shrink_ = False

        self.logger.info('n_classes: {}'.format(self.n_classes_))
        self.logger.info('classes: {}'.format(self.classes_))

        random_state = check_random_state(self.random_state)
        seed = random_state.randint(0, np.iinfo(np.int32).max)
        np.random.seed(seed)

        self.n_seeds_ = 0

        # Define feature domains
        self.mins_, self.maxs_ = define_domain(self.X[:, dtypes == 'numeric'], self.domain) if not isinstance(self.X, pd.DataFrame) else define_domain(self.X[self.cols_[np.where(dtypes=='numeric')[0]]], self.domain)

        self.uniques_ = define_uniques(self.X[:, dtypes == 'categorical']) if not isinstance(self.X, pd.DataFrame) else define_uniques(self.X[self.cols_[np.where(dtypes=='categorical')[0]]])
        self.dtypes_ = dtypes
        self.n_numeric_ = sum(self.dtypes_ == 'numeric')
        self.n_categorical_ = sum(self.dtypes_ == 'categorical')

        self.n_queries = 0

        return self._query()
