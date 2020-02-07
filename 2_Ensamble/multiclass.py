#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MULTICLASS CLASSIFIER

Created on Tue Mar 27 20:06:33 2018

@author: irene

Define ECOC strategy to implement
multiclass methods on copy classifiers
(currently: Online RBF SVM, Hoeffding Tree)

"""

import itertools
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone, copy
from sklearn.model_selection import train_test_split
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets
from joblib import Parallel, delayed

def check_estimator(estimator):
    """ Make sure that an estimator implements the necessary methods.
    """

    if not hasattr(estimator, "predict"):
        raise ValueError("The base estimator should implement predict")

def _fit_binary(estimator, X, y, classes=None):
    """ Fit a single binary estimator.
    """

    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn("Label %s is present in all training examples." %
                          str(classes[c]))
        estimator = _ConstantPredictor().fit(X, unique_y)
    else:
        estimator = clone(estimator)
        estimator.fit(X, y)

    return estimator

def _fit_ovo_binary(estimator, X, y):
    """Fit a single binary estimator (one-vs-one).
    """

    cond = np.logical_or(y == 1, y == -1)
    y_binary = y[cond]
    indcond = np.arange(X.shape[0])[cond]

    return _fit_binary(estimator,
                       _safe_split(estimator, X, None, indices=indcond)[0],
                       y_binary, classes=[1, -1]), indcond

def _predict_binary(estimator, X):
    """ Make predictions using a single binary estimator.
    """
    
    if hasattr(estimator, "predict_proba"):
        score = estimator.predict_proba(X)
        #score = score[:, 1]
    else:
        score = estimator.predict(X)
    return score

class _ConstantPredictor(BaseEstimator):

    def fit(self, X, y):
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self, 'y_')

        return np.repeat(self.y_, X.shape[0])

    def decision_function(self, X):
        check_is_fitted(self, 'y_')

        return np.repeat(self.y_, X.shape[0])

    def predict_proba(self, X):
        check_is_fitted(self, 'y_')

        return np.repeat([np.hstack([1 - self.y_, self.y_])],
                         X.shape[0], axis=0)

class ECOC(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self,
                 estimator,
                 n_jobs=1,
                 distance='euclidean'):

        """ Error-Correcting Output-Code multiclass strategy
            Output-code based strategies consist in representing each class with a
            binary code (an array of 0s and 1s). At fitting time, one binary
            classifier per bit in the code book is fitted.  At prediction time, the
            classifiers are used to project new points in the class space and the class
            closest to the points is chosen.

            Parameters
            ----------
                estimator : estimator object
                    An estimator object implementing `fit` and `predict` methods.

                n_jobs : int, optional (default=1)
                    The number of jobs to use for the computation. If -1 all CPUs are used.
                    If 1 is given, no parallel computing code is used at all, which is
                    useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
                    used. Thus for n_jobs = -2, all CPUs but one are used.

                distance: string, optional (default='euclidean')
                    Type of distance metric to use when computing the closest class
                    to each predicted point.
        """

        self.estimator = estimator
        self.n_jobs = n_jobs
        self.distance = distance

    def fit(self, X, y):
        """ Fit underlying estimators.

            Parameters
            ----------

                X : {array-like, sparse matrix}, shape (n_samples, n_features)
                    Set of samples, where n_samples is the number of samples and
                    n_features is the number of features.

                y : array-like, shape (n_samples,)
                    Set of labels, where n_samples is the number of samples.
                    provided in multiclass format
        """

        # Check estimator has necessary methods implemented
        check_estimator(self.estimator)

        # Check data has correct shape
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_classes_ = len(np.unique(y))

        #if n_classes_ < 3:
        #    sys.exit('There are no more than two classes, making the use of ECOC unneccesary. Exiting.')

        # Create code_book for all classifiers
        # h1, h2, ... and the corresponding class
        # identifiers C1, C2, ...
        self.code_book_ = np.zeros((n_classes_, int(n_classes_*(n_classes_-1)/2)))
        for idx, val in enumerate(itertools.combinations(range(n_classes_), 2)):
            self.code_book_[val[0], idx] = 1
            self.code_book_[val[1], idx] = -1

        classes_index = dict((c, i) for i, c in enumerate(self.classes_))

        # Create new array of labels, where its
        # class is represented by its corresponding
        # code Ci = (h1i, h2i,...)
        Y = np.array([self.code_book_[classes_index[y[i]]]
                      for i in range(X.shape[0])], dtype=np.int)
        
        # Train binary estimators
        estimators_indices = list(zip(*(Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_ovo_binary)
            (self.estimator, X, Y[:,i])
            for i in range(Y.shape[1])))))

        self.estimators_ = estimators_indices[0]
        self.indices_ = estimators_indices[1]

        return self

    def predict(self, X):
        """ Predict multi-class targets using underlying estimators.

            Parameters
            ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
                Set of samples, where n_samples is the number of samples and
                n_features is the number of features.
        """

        check_is_fitted(self, 'estimators_')
        X = check_array(X)
        Y = np.array([_predict_binary(e, X) for e in self.estimators_]).T

        if self.distance == 'euclidean':
            pred = euclidean_distances(Y, self.code_book_).argmin(axis=1)
        else:
            pred = manhattan_distances(Y, self.code_book_).argmin(axis=1)
        return self.classes_[pred]

