#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONLINE COPIES

Created on Thu Dec 14 16:33:45 2018

@author: irene
"""

import time
import random
import itertools
import scipy.spatial.distance
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_consistent_length

#from methods.base_HoeffdingTree import BaseHoeffdingTree
#from methods.base_SVM import BaseSVM
#from methods.base_MLP import BaseMLP

import sys
sys.path.insert(0, '/Users/irene/Documents/Projects/dual/scripts/methods/')

from base_HoeffdingTree import BaseHoeffdingTree
#from base_OnlineSVM import BaseOnlineSVM
#from base_CopyOnlineSVM import BaseCopyOnlineSVM

class HoeffdingTree(BaseHoeffdingTree):
    """ A Hoeffding tree classifier.

        Parameters
        ----------
        splitter : string, optional (default="mean")
            Strategy used to choose the split at each node. Supported
            strategies are "mean" to choose the mean split and "median" to choose
            the median split.

        min_samples_split: int, optional (default=100)
            Minimum number of samples required to split an internal node. In
            practice, it corresponde to the maximum number of minimum samples to
            split a node, since this value is decreased during each iteration.

        max_samples : int, optional (default=1000)
            Number of random samples drawn during each iteration of the method.
            Number of iterations to perform.
        
        max_depth : int or None, optional (default=None)
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves ocntain less than min_samples_split
            samples.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        """

    def __init__(self,
                 split_criterion="mean",
                 min_samples_split=100,
                 min_samples_split_lower = 2,
                 max_depth=None,
                 random_state=None):

        super(HoeffdingTree, self).__init__(split_criterion=split_criterion,
                                            min_samples_split=min_samples_split,
                                            min_samples_split_lower=min_samples_split_lower,
                                            max_depth=max_depth,
                                            random_state=random_state)

    def fit(self, X, y):
        """ Build a Hoeffding tree classifier from the training set (X, y).

            For each iteration t, evaluate a randomly selected
            set of samples of size equal to max_samples. Expand the tree until samples
            are classified and update internal parameters.

            Parameters
            ----------
            X : array-like or sparse matrix, shape = [n_samples, n_features]
                Set of samples, where n_samples is the number of samples and
                n_features is the number of features.

            y : array-like, shape = [n_samples] or [n_samples, n_outputs]
                Set of labels, where n_samples is the number of samples.

            Returns
            -------
            self : object
                Returns self.
        """

        super(HoeffdingTree, self).fit(X, y)

        return self
    
    def predict_proba(self, X):
        """ Compute a class probability estimator.

            Parameters
            ----------
            X : array-like or sparse matrix, shape = [n_samples, n_features]
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features. Original data points
            (used to define feature domains).

            Returns
            -------
            p : array of shape = [n_samples]
                The class probabilities of the input samples.
            """
        X = check_array(X, accept_sparse=True, ensure_min_samples=1, dtype=np.float64)
        all_proba = []
        for xi in X:
            node = self._traverse_tree(xi)[0]
            all_proba.append(node.classes_/sum(node.classes_))
        return np.array(all_proba)

